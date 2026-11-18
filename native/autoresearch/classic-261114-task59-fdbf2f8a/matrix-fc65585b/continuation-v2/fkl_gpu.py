"""Canonical Target versus GPU saved-H reconstruction gate and resumable missing-H fill."""
import argparse,collections,hashlib,json,sys,time,traceback
from common import *
OUT=ROOT/'fkl'
class Mismatch(Exception):
 def __init__(self,detail):self.detail=detail;super().__init__(detail.get('check','mismatch'))
def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['gate','backfill'],required=True);args=a.parse_args()
 verify_freeze();out=OUT/args.phase;out.mkdir(exist_ok=True)
 if (out/'completion.json').exists():return
 if args.phase=='backfill':
  gate=read(OUT/'gate/completion.json')
  if gate['status']!='complete':write(out/'completion.json',{'status':'blocked','reason':'canonical GPU reconstruction gate did not pass'});return
 dedup=read(OUT/'dedup_receipt.json')
 for e in dedup['files'].values():assert sha(e['path'])==e['sha256']
 requests={x['request_id']:x for x in lines(OUT/'requests.jsonl')};jobs=lines(OUT/'unique_prefixes.jsonl')
 if args.phase=='gate':
  chosen=[]
  for split in ['train','calibration']:
   for kind in ['request_eos','request_truncated']:
    pool=[x for x in jobs if x['split']==split and x[kind]]
    if pool:
     first=pool[0];same=[x for x in pool if x['request_id']==first['request_id']]
     for x in [same[0],same[-1]]:
      if x['prefix_id'] not in {q['prefix_id'] for q in chosen}:chosen.append(x)
  jobs=chosen;assert len(jobs)>=4
 try:
  with lease(out):run(args.phase,out,requests,jobs)
 except Mismatch as e:
  write(out/'first_error.json',dict(e.detail,unix=time.time()))
  write(out/'completion.json',{'status':'blocked','reason':'exact canonical/reconstruction/raw comparison failed','first_error_sha256':sha(out/'first_error.json'),'full_vocab_training_admitted':False})
 except Exception as e:
  write(out/'first_error.json',{'error':repr(e),'traceback':traceback.format_exc(),'unix':time.time()});raise
def run(phase,out,requests,jobs):
 import torch
 from safetensors import safe_open
 from safetensors.torch import save_file
 f=read(WAVE/'freeze.json');sys.path.insert(0,f['source_root'])
 from feedback_codec.hf_backend import HFTargetBackend
 torch.set_num_threads(4)
 target=HFTargetBackend(str(pathlib.Path(f['models_root'])/'Qwen3-8B'),None,device='cuda:0',dtype=torch.bfloat16)
 head=target.model.get_output_embeddings();sealed=set();manifest=out/'manifest.jsonl';traces={};values={};metadata=[];examples=[]
 if manifest.exists():
  for e in lines(manifest):
   assert sha(e['path'])==e['sha256'] and sha(e['metadata_path'])==e['metadata_sha256']
   for row in lines(e['metadata_path']):assert row['prefix_id'] not in sealed;sealed.add(row['prefix_id'])
 def check(ok,name,job,**details):
  if not ok:raise Mismatch(dict(check=name,prefix_id=job['prefix_id'],request_id=job['request_id'],confirmed_output_tokens=job['confirmed_output_tokens'],**details))
 def seal():
  if not metadata:return
  number=len(lines(manifest)) if manifest.exists() else 0
  dest=out/'shards'/f'{number:06d}';tmp=out/'incomplete'/f'{number:06d}-{time.time_ns()}';tmp.mkdir(parents=True)
  save_file(values,str(tmp/'hidden.safetensors'))
  with (tmp/'metadata.jsonl').open('w') as z:
   for r in metadata:z.write(json.dumps(r,separators=(',',':'))+'\n')
   z.flush();os.fsync(z.fileno())
  dest.parent.mkdir(exist_ok=True);assert not dest.exists();tmp.rename(dest)
  e={'path':str(dest/'hidden.safetensors'),'sha256':sha(dest/'hidden.safetensors'),'metadata_path':str(dest/'metadata.jsonl'),'metadata_sha256':sha(dest/'metadata.jsonl'),'rows':len(metadata)}
  with manifest.open('a') as z:z.write(json.dumps(e)+'\n');z.flush();os.fsync(z.fileno())
  for r in metadata:sealed.add(r['prefix_id'])
  values.clear();metadata.clear();write(out/'status.json',{'sealed_prefixes':len(sealed),'total':len(jobs),'pid':os.getpid(),'unix':time.time()})
 started=time.perf_counter()
 for job in jobs:
  if job['prefix_id'] in sealed:continue
  req=requests[job['request_id']];prefix=req['prompt_ids']+req['output_ids'][:job['confirmed_output_tokens']]
  check(hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest()==job['prefix_sha256'],'prefix_binding',job)
  with torch.inference_mode():
   # Same full-prefix canonical path used by the accepted verifier, no KV reuse.
   output,final=target._forward_with_final_hidden(target._ids(prefix));canonical=output.logits[0,-1].detach().float()
   saved=final[0,-1].detach().to('cpu').contiguous();rebuilt=head(saved.to('cuda:0').reshape(1,-1))[0].float()
   if not torch.equal(canonical,rebuilt):
    index=int((canonical!=rebuilt).nonzero()[0]);top=canonical.topk(2)
    check(False,'full_vocab_GPU_reconstruction',job,first_vocab_id=index,canonical=float(canonical[index]),rebuilt=float(rebuilt[index]),max_abs_error=float((canonical-rebuilt).abs().max()),top2_ids=top.indices.tolist(),top2_logits=top.values.tolist())
   if phase=='gate':
    independent=target.next(prefix);check(independent.token_id==int(canonical.argmax()) and torch.equal(independent.hidden.cpu(),saved),'independent_canonical_next',job)
   check(int(canonical.argmax())==job['expected_id'],'expected_Target_ID',job,actual=int(canonical.argmax()),expected=job['expected_id'])
   rid=req['request_id'];p=DATA/'shards'/rid
   if rid not in traces:
    assert sha(req['trace_path'])==req['trace_sha256'];traces={rid:read(req['trace_path'])}
   rr=traces[rid]['rounds'][job['raw_round']];key=rr['tensor_prefix'];j=job['raw_decision_row']
   with safe_open(p/'features.safetensors',framework='pt',device='cpu') as sf:
    ids=sf.get_tensor(key+'.top32_ids')[j];logits=sf.get_tensor(key+'.top32_logits')[j].float();lse=float(sf.get_tensor(key+'.logsumexp')[j]);valid=bool(sf.get_tensor(key+'.candidate_valid')[j]);cand=float(sf.get_tensor(key+'.candidate_logits')[j])
   actual_top=canonical[ids.to('cuda:0')].cpu();check(torch.equal(actual_top,logits),'saved_raw_top32_values',job,max_abs_error=float((actual_top-logits).abs().max()))
   check(set(canonical.topk(32).indices.tolist())==set(ids.tolist()),'saved_raw_top32_ids',job)
   actual_lse=float(canonical.logsumexp(0));check(actual_lse==lse,'saved_raw_FP32_logsumexp',job,canonical=actual_lse,saved=lse)
   if valid:
    cid=rr['candidate_ids'][j];check(float(canonical[cid])==cand,'saved_raw_candidate_score',job,candidate_id=cid,canonical=float(canonical[cid]),saved=cand)
   values[job['prefix_id']]=saved
   metadata.append(dict(job,full_vocab_reconstruction_bitwise_equal=True,raw_top32_candidate_LSE_exact=True,H_sha256=hashlib.sha256(saved.view(torch.uint8).numpy().tobytes()).hexdigest(),FP32_logsumexp=actual_lse))
   if len(examples)<8:examples.append(metadata[-1])
  if len(metadata)>=256:seal()
 seal()
 write(out/'completion.json',{'status':'complete','phase':phase,'prefixes':len(sealed),'elapsed_seconds':time.perf_counter()-started,'manifest_sha256':sha(manifest),'samples':examples,
  'storage':'BF16 final norm H, losslessly reconstructable through frozen lm_head under proven exact GPU gate','all_filled_prefixes_individually_verified':True,
  'FKL_training_admitted':False,'remaining_for_FKL_training':'bind existing available-H positions and complete unified all-label coverage receipt before launching FKL loss','GPU_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'unix':time.time()})
if __name__=='__main__':main()
