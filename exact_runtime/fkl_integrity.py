"""Independent, CPU-only validation of every sealed FKL shard and label binding."""
import hashlib,json,pathlib
from guards import SharedIntegrityError,require
import native_logits as codec
def sha(path):
 with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def lines(path):
 with open(path) as f:
  for line in f:
   if line.strip():yield json.loads(line)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def gate_jobs(jobs):
 chosen={}
 for split in ('train','calibration'):
  for kind in ('request_eos','request_truncated'):
   pool=[x for x in jobs.values() if x['split']==split and x[kind]]
   if pool:
    same=[x for x in pool if x['request_id']==pool[0]['request_id']]
    for x in (same[0],same[-1]):chosen[x['prefix_id']]=x
 require(len(chosen)>=4,'canonical gate fixture coverage incomplete');return chosen
def coverage_inputs(root,expected_occurrences=1370920,expected_prefixes=531692):
 root=pathlib.Path(root);c=json.loads((root/'completion.json').read_bytes())
 require(c['status']=='complete_CPU_mapping','label mapping not accepted')
 for e in c['files'].values():require(sha(e['path'])==e['sha256'],'frozen label mapping artifact SHA differs')
 jobs={}
 for row in lines(c['files']['all_unique_prefixes.jsonl']['path']):
  require(row['prefix_id'] not in jobs,'duplicate frozen prefix ID');jobs[row['prefix_id']]=row
 require(len(jobs)==expected_prefixes,'frozen prefix count differs')
 counts={};total=0;window_positions={}
 for row in lines(c['files']['all_occurrences.jsonl']['path']):
  key=row['prefix_id'];require(key in jobs,'label references unknown prefix');j=jobs[key]
  require(row['expected_id']==j['expected_id'] and row['split']==j['split'],'frozen label to prefix binding differs')
  require(row['valid'] is True,'invalid label occurrence in accepted map')
  w=(row['request_id'],row['window_index']);position=window_positions.get(w,0)
  require(row['label_position0']==position,'duplicate/missing label position');window_positions[w]=position+1
  counts[key]=counts.get(key,0)+1;total+=1
 require(total==expected_occurrences and set(counts)==set(jobs),'frozen all-label count or prefix coverage differs')
 return jobs,dict(mapping_receipt_sha256=sha(root/'completion.json'),occurrences=total,unique_prefixes=len(jobs),occurrence_counts_sha256=hashlib.sha256(canonical(counts)).hexdigest())
def validate_entry(entry,jobs,seen=None,hidden_dim=4096):
 import torch
 from safetensors import safe_open
 seen=set() if seen is None else seen
 # These calls MUST be separate. require() intentionally has no truth value.
 require(sha(entry['path'])==entry['sha256'],'FKL data SHA differs')
 require(sha(entry['metadata_path'])==entry['metadata_sha256'],'FKL metadata SHA differs')
 rows=list(lines(entry['metadata_path']));ids=[r['prefix_id'] for r in rows]
 require(entry['rows']==len(rows) and len(ids)==len(set(ids)),'FKL metadata row count or duplicate prefix differs')
 require(not seen.intersection(ids),'FKL prefix repeated across sealed shards')
 require(set(ids).issubset(jobs),'FKL prefix not in frozen mapping')
 with safe_open(entry['path'],framework='pt',device='cpu') as f:
  keys=set(ids)
  for row in rows:
   key=row['prefix_id'];info=row['native_logits'];keys.update(codec.tensor_names(key,info['mode']))
   codec.validate(key,info,f.get_tensor,codec.VOCAB)
  require(set(f.keys())==keys,'FKL data/metadata tensor keys differ')
  for row in rows:
   key=row['prefix_id'];job=jobs[key]
   require(all(row.get(k)==v for k,v in job.items()),'FKL metadata differs from frozen prefix/label mapping: '+key)
   require(row.get('full_vocab_reconstruction_bitwise_equal') is True and row.get('raw_top32_candidate_LSE_exact') is True,'FKL prefix has no canonical acceptance')
   h=f.get_tensor(key);require(h.dtype==torch.bfloat16 and h.shape==(hidden_dim,) and bool(torch.isfinite(h).all()),'FKL saved H shape/dtype/finiteness differs')
   digest=hashlib.sha256(h.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
   require(digest==row['H_sha256'],'FKL metadata H digest differs from data')
 return set(ids)
def validate_manifest(manifest,jobs,*,complete=False,terminal=None,hidden_dim=4096):
 manifest=pathlib.Path(manifest);seen=set();entries=[]
 if manifest.exists():
  for e in lines(manifest):
   seen.update(validate_entry(e,jobs,seen,hidden_dim));entries.append(e)
 if terminal is not None:
  require(terminal.get('status')=='complete','FKL terminal is not complete')
  require(terminal['manifest_sha256']==sha(manifest),'FKL terminal manifest SHA differs')
  require(terminal['prefixes']==len(seen),'FKL terminal prefix count differs')
 if complete:require(seen==set(jobs),'FKL completed prefix ID set differs from frozen map')
 return dict(prefixes=len(seen),prefix_ID_set_sha256=hashlib.sha256(canonical(sorted(seen))).hexdigest(),seen=seen,entries=entries)
