"""Canonical Target versus GPU saved-H reconstruction gate and resumable missing-H fill."""
import argparse,collections,hashlib,json,sys,time,traceback
from common import *
import native_logits as codec
from fkl_integrity import coverage_inputs,validate_manifest,validate_entry,gate_jobs
from fkl_shards import recover,publish
OUT=ROOT/'fkl';INPUT=V3/'labels'
def strict_context():
 return dict(logit_format=codec.FORMAT,codec_sha256=sha(ROOT/'native_logits.py'),label_mapping_sha256=sha(INPUT/'completion.json'),runtime_freeze_sha256=sha(ROOT/'runtime_freeze.json'),writer_sha256=sha(ROOT/'fkl_shards.py'),integrity_sha256=sha(ROOT/'fkl_integrity.py'),worker_sha256=sha(ROOT/'fkl_all_verified_gpu.py'))
class Mismatch(FKLSpecificError):
 def __init__(self,detail):self.detail=detail;super().__init__(detail.get('check','mismatch'))
def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['gate','backfill'],required=True);args=a.parse_args()
 verify_freeze();out=OUT/args.phase;out.mkdir(parents=True,exist_ok=True)
 own=open(out/'worker.lock','a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)

 if args.phase=='backfill':
  gate=read(OUT/'gate/completion.json')
  if gate['status']!='complete':raise FKLSpecificError('canonical GPU reconstruction gate did not pass')
 frozen,coverage=coverage_inputs(INPUT)
 if args.phase=='backfill':
  require(gate.get('strict_context')==strict_context(),'gate strict writer/label context differs')
  probe=read(OUT/'gate/native_roundtrip.json')
  require(probe['strict_context']==strict_context() and probe['manifest_sha256']==gate['manifest_sha256'] and {r['prefix_id'] for r in probe['rows']}==set(gate_jobs(frozen)),'missing real disk roundtrip for gate prefixes')
  recover(OUT/'gate',gate_jobs(frozen),strict_context())
  require(gate['all_label_mapping_sha256']==coverage['mapping_receipt_sha256'],'gate label mapping changed');validate_manifest(OUT/'gate/manifest.jsonl',gate_jobs(frozen),complete=True,terminal=gate)
 requests={x['request_id']:x for x in lines(INPUT/'requests.jsonl')};jobs=list(frozen.values())
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
 expected={x['prefix_id']:x for x in jobs}
 manifest=out/'manifest.jsonl'
 context=strict_context();recovery=recover(out,expected,context)
 terminal=read(out/'completion.json') if (out/'completion.json').exists() else None
 accepted=validate_manifest(manifest,expected,complete=bool(terminal and terminal['status']=='complete'),terminal=terminal if terminal and terminal['status']=='complete' else None)
 if terminal:
  require(terminal.get('strict_context')==context,'strict terminal writer/mapping binding differs')
  require(terminal['status']=='complete' and terminal['all_label_mapping_sha256']==coverage['mapping_receipt_sha256'],'prior terminal or coverage binding differs');return
 if args.phase != 'gate' and accepted['seen']==set(expected):
  write(out/'completion.json',dict(status='complete',phase=args.phase,prefixes=len(expected),manifest_sha256=sha(manifest),all_label_mapping_sha256=coverage['mapping_receipt_sha256'],FKL_training_admitted=args.phase=='backfill',reused_only_after_strict_shard_recovery=True,strict_context=context,unix=time.time()));return
 try:
  with lease(out):run(args.phase,out,requests,jobs,context)
 except Exception as e:
  # Mismatch and explicit CUDA OOM are confined to the FKL gate/backfill.
  # Any other error, especially shared input integrity, blocks the queue.
  import torch
  if isinstance(e,torch.cuda.OutOfMemoryError):e=FKLSpecificError('FKL CUDA OOM: '+repr(e))
  first_error(out/'first_error.json',e,detail=getattr(e,'detail',None))
  if category(e)=='fkl_specific':
   write(out/'completion.json',{'status':'blocked','category':'fkl_specific','first_error_sha256':sha(out/'first_error.json'),'full_vocab_training_admitted':False})
  else:raise

def run(phase,out,requests,jobs,context):
 import torch
 f=read(WAVE/'freeze.json');sys.path.insert(0,f['source_root'])
 from feedback_codec.hf_backend import HFTargetBackend
 torch.set_num_threads(4)
 target=HFTargetBackend(str(pathlib.Path(f['models_root'])/'Qwen3-8B'),None,device='cuda:0',dtype=torch.bfloat16)
 run_target(phase,out,requests,jobs,context,target)

def run_target(phase,out,requests,jobs,context,target):
 import torch
 from safetensors import safe_open
 head=target.model.get_output_embeddings();device=head.weight.device;sealed=set();manifest=out/'manifest.jsonl';traces={};values={};metadata=[];examples=[]
 expected={job['prefix_id']:job for job in jobs}
 accepted=validate_manifest(manifest,expected);sealed=set(accepted['seen'])
 def check(ok,name,job,**details):
  if not ok:raise Mismatch(dict(check=name,prefix_id=job['prefix_id'],request_id=job['request_id'],confirmed_output_tokens=job['confirmed_output_tokens'],**details))
 def seal():
  if not metadata:return
  new_ids=publish(out,values,metadata,expected,context,sealed)
  sealed.update(new_ids);values.clear();metadata.clear();write(out/'status.json',{'sealed_prefixes':len(sealed),'total':len(jobs),'pid':os.getpid(),'unix':time.time(),'strict_context':context})
 started=time.perf_counter()
 for job in jobs:
  if job['prefix_id'] in sealed:continue
  req=requests[job['request_id']];prefix=req['prompt_ids']+req['output_ids'][:job['confirmed_output_tokens']]
  require(hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest()==job['prefix_sha256'],'shared prefix binding differs')
  with torch.inference_mode():
   # Same full-prefix canonical path used by the accepted verifier, no KV reuse.
   output,final=target._forward_with_final_hidden(target._ids(prefix));canonical=output.logits[0,-1].detach().float()
   check(bool(torch.isfinite(canonical).all()),'FKL_nonfinite_canonical_logits',job)
   saved=final[0,-1].detach().to('cpu').contiguous()
   if job.get('saved_H_path'):
    with safe_open(job['saved_H_path'],framework='pt',device='cpu') as sf:existing=sf.get_tensor(job['saved_H_key']).reshape(-1)
    check(torch.equal(existing,saved),'saved_RAW_final_norm_H_equals_canonical',job,max_abs_error=float((existing.float()-saved.float()).abs().max()))
   rebuilt=head(saved.to(device).reshape(1,-1))[0].float()
   key_id=job['prefix_id']
   info,extra=codec.encode(key_id,canonical,rebuilt)
   if phase=='gate' and info['replaced']:
    first=int((canonical.contiguous().view(torch.int32)!=rebuilt.contiguous().view(torch.int32)).nonzero()[0].item())
    info.update(first_vocab_id=first,first_native=float(canonical[first]),first_reconstructed=float(rebuilt[first]))
   restored=codec.decode(key_id,info,extra,rebuilt)
   check(torch.equal(canonical,restored),'full_vocab_native_roundtrip',job)
   values.update(extra)
   if phase=='gate':
    independent=target.next(prefix);require(independent.token_id==int(canonical.argmax()),'shared independent Target greedy differs');check(torch.equal(independent.hidden.cpu(),saved),'independent_canonical_hidden',job)
   require(int(canonical.argmax())==job['expected_id'],'shared canonical Target ID differs from frozen RAW label: '+job['prefix_id'])
   rid=req['request_id'];p=DATA/'shards'/rid
   if rid not in traces:
    require(sha(req['trace_path'])==req['trace_sha256'],'shared RAW trace SHA differs');traces={rid:read(req['trace_path'])}
   rr=traces[rid]['rounds'][job['raw_round']];key=rr['tensor_prefix'];j=job['raw_decision_row']
   with safe_open(p/'features.safetensors',framework='pt',device='cpu') as sf:
    ids=sf.get_tensor(key+'.top32_ids')[j];logits=sf.get_tensor(key+'.top32_logits')[j].float();lse=float(sf.get_tensor(key+'.logsumexp')[j]);valid=bool(sf.get_tensor(key+'.candidate_valid')[j]);cand=float(sf.get_tensor(key+'.candidate_logits')[j])
   actual_top=canonical[ids.to(device)].cpu();check(torch.equal(actual_top,logits),'saved_raw_top32_values',job,max_abs_error=float((actual_top-logits).abs().max()))
   check(set(canonical.topk(32).indices.tolist())==set(ids.tolist()),'saved_raw_top32_ids',job)
   actual_lse=float(canonical.logsumexp(0));check(actual_lse==lse,'saved_raw_FP32_logsumexp',job,canonical=actual_lse,saved=lse)
   if valid:
    cid=rr['candidate_ids'][j];check(float(canonical[cid])==cand,'saved_raw_candidate_score',job,candidate_id=cid,canonical=float(canonical[cid]),saved=cand)
   values[job['prefix_id']]=saved
   metadata.append(dict(job,native_logits=info,full_vocab_reconstruction_bitwise_equal=True,raw_top32_candidate_LSE_exact=True,H_sha256=hashlib.sha256(saved.view(torch.uint8).numpy().tobytes()).hexdigest(),FP32_logsumexp=actual_lse))
   if len(examples)<8:examples.append(metadata[-1])
  if len(metadata)>=256:seal()
 seal()
 accepted=validate_manifest(manifest,expected,complete=True)
 if phase=='gate':
  from safetensors.torch import load_file
  from fkl_integrity import lines as row_lines
  from full_vocab_supervision import reconstruct_row
  gate_rows=[]
  with torch.inference_mode():
   for entry in accepted['entries']:
    payload=load_file(entry['path'],device='cpu')
    for row in row_lines(entry['metadata_path']):
     logits=reconstruct_row(head,row['prefix_id'],row['native_logits'],payload)
     check(int(logits.argmax())==row['expected_id'],'gate_disk_teacher_greedy',row)
     gate_rows.append(dict(prefix_id=row['prefix_id'],**row['native_logits']))
  write(out/'native_roundtrip.json',dict(strict_context=context,rows=gate_rows,manifest_sha256=sha(manifest)))
 write(out/'completion.json',{'status':'complete','strict_context':context,'phase':phase,'prefixes':len(sealed),'elapsed_seconds':time.perf_counter()-started,'manifest_sha256':sha(manifest),'samples':examples,
  'storage':'BF16 final H and exact native logit replacements, verified by per-prefix FP32 digest','all_filled_prefixes_individually_verified':True,
  'FKL_training_admitted':phase=='backfill' and len(sealed)==531692,'all_label_mapping_sha256':sha(INPUT/'completion.json'),'label_occurrences':1370920,'all_unique_prefixes_expected':531692,'GPU_peak_reserved_mib':torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else 0,'unix':time.time()})
if __name__=='__main__':
 try:main()
 except Exception as e:
  phase=sys.argv[sys.argv.index('--phase')+1] if '--phase' in sys.argv else 'unknown'
  first_error(OUT/phase/'first_error.json',e)
  if category(e)=='fkl_specific':write(OUT/phase/'completion.json',dict(status='blocked',category='fkl_specific',first_error_sha256=sha(OUT/phase/'first_error.json'),GPU_work_started=(OUT/phase/'gpu_lease.json').exists(),full_vocab_training_admitted=False))
  raise
