"""Bind every supervised occurrence to its exact confirmed Target prefix, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,hashlib,json,time
from common import *
def digest(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def main():
 out=ROOT/'labels';out.mkdir(exist_ok=True)
 if (out/'completion.json').exists():return
 started=time.monotonic();lock=read(MATRIX/'partition_lock.json');require(sha(TRAIN/'index.jsonl')==lock['source_index_sha256'],'source index differs')
 binding=read(WAVE/'freeze.json')['models_manifest_sha256'];seen={};counts=collections.Counter();missing_hash=hashlib.sha256();missing_unique=set();used=set()
 allmap=out/'all_occurrences.jsonl';unique=out/'all_unique_prefixes.jsonl';requestpath=out/'requests.jsonl'
 require(not any(p.exists() for p in (allmap,unique,requestpath)),'preserve incomplete label attempt; no append')
 with allmap.open('x') as mf,unique.open('x') as uf,requestpath.open('x') as rf:
  for r in lines(TRAIN/'index.jsonl'):
   if r['split'] not in ('train','calibration'):continue
   rid=r['request_id'];p=DATA/'shards'/rid;receipt=read(p/'receipt.json');require(sha(p/'trace.json')==receipt['files']['trace.json'],'RAW trace hash differs')
   t=read(p/'trace.json');require(t['trajectory_policy']=='RAW' and not t['repair_enabled'] and t['output_ids']==r['output_ids'],'RAW trajectory differs')
   prompt=r['prompt_ids'];output=r['output_ids'];source={};hpos={}
   for rr in t['rounds']:
    for j,pos in enumerate(rr['top32']['absolute_source_positions']):
     require(pos not in source,'repeated RAW validation position');source[pos]=(rr['round_id'],j,rr['top32']['greedy_ids'][j])
    if rr['hidden']['present']:hpos[rr['hidden']['absolute_position']]=rr['tensor_prefix']+'.hidden'
   rf.write(json.dumps(dict(request_id=rid,split=r['split'],family_id=r['family_id'],prompt_ids=prompt,output_ids=output,trace_path=str(p/'trace.json'),trace_sha256=sha(p/'trace.json')),separators=(',',':'))+'\n')
   for wi,w in enumerate(r['windows']):
    require(len(w['labels'])==w['length'] and 1<=w['length']<=8,'window label mask differs')
    require(w['labels']==output[w['offset']:w['offset']+w['length']],'window labels differ from committed Target output')
    require(151645 not in w['labels'][:-1],'labels continue beyond EOS')
    for j,label in enumerate(w['labels']):
     n=w['offset']+j;pos=len(prompt)+n-1;rr,row,expected=source[pos];require(label==expected,'RAW validation ID does not match label')
     prefix=prompt+output[:n];psha=digest(prefix);key=hashlib.sha256((binding+':BF16:eager:canonical_full_prefix:'+psha).encode()).hexdigest();available=pos in hpos
     ref=dict(request_id=rid,split=r['split'],window_index=wi,observation_key=w['observation_key'],label_position0=j,expected_id=label,prefix_id=key)
     if not available:missing_hash.update((json.dumps(ref,separators=(',',':'))+'\n').encode());missing_unique.add(key)
     mf.write(json.dumps(dict(ref,H_source='sealed_RAW_boundary' if available else 'canonical_backfill_required',valid=True,request_eos=t['eos'],request_truncated=t['truncated']),separators=(',',':'))+'\n')
     counts['label_occurrences']+=1;counts['available_H_occurrences' if available else 'missing_H_occurrences']+=1;counts[r['split']+'_occurrences']+=1;counts['EOS_label_occurrences']+=label==151645
     if key in seen:require(seen[key]==(r['split'],label,available),'cross split or supervision source collision')
     else:
      seen[key]=(r['split'],label,available)
      item=dict(prefix_id=key,prefix_sha256=psha,prefix_length=len(prefix),request_id=rid,split=r['split'],confirmed_output_tokens=n,absolute_source_position=pos,expected_id=label,raw_round=rr,raw_decision_row=row,request_eos=t['eos'],request_truncated=t['truncated'],target_binding_sha256=binding,precision='BF16',backend='eager_canonical_full_prefix',saved_H_path=str(p/'features.safetensors') if available else None,saved_H_key=hpos.get(pos))
      uf.write(json.dumps(item,separators=(',',':'))+'\n');counts['unique_prefixes']+=1;counts['unique_available' if available else 'unique_missing']+=1
   counts[r['split']+'_windows']+=len(r['windows']);used.add(r['split'])
  for f in (mf,uf,rf):f.flush();os.fsync(f.fileno())
 prior=read(V2/'fkl/dedup_receipt.json');require(counts['label_occurrences']==1370920 and counts['available_H_occurrences']==448788 and counts['missing_H_occurrences']==922132,'label acceptance counts differ')
 require(counts['train_windows']==156688 and counts['calibration_windows']==19736,'frozen window counts differ')
 require(missing_hash.hexdigest()==prior['files']['occurrence_map.jsonl']['sha256'],'all-label missing projection differs from sealed v2 mapping')
 require(missing_unique=={x['prefix_id'] for x in lines(V2/'fkl/unique_prefixes.jsonl')},'missing unique prefix coverage differs')
 write(out/'completion.json',dict(status='complete_CPU_mapping',counts=dict(counts),missing_projection_sha256=missing_hash.hexdigest(),v2_dedup_receipt_sha256=sha(V2/'fkl/dedup_receipt.json'),files={p.name:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in (allmap,unique,requestpath)},
  full_vocab_training_admitted=False,remaining_gate='GPU canonical full-vocabulary equality and every saved/backfilled H verification; CPU mapping is not logits acceptance',independent300_used=False,elapsed_seconds=time.monotonic()-started,unix=time.time()))
 print(json.dumps(read(out/'completion.json')),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:first_error(ROOT/'labels/first_error.json',e);raise
