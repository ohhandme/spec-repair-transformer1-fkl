"""Real subprocess exit after rename, durable partial-tail and orphan recovery regressions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,hashlib,json,pathlib,subprocess,sys,tempfile,time
import torch
from common import ROOT,read,write,sha
from guards import SharedIntegrityError
from fkl_integrity import canonical,validate_manifest
from fkl_shards import publish,recover

def fixture():
 jobs={};values={};rows=[]
 for i in range(3):
  key=hashlib.sha256(str(i).encode()).hexdigest();j=dict(prefix_id=key,expected_id=i+10,split='train',request_id='r'+str(i),prefix_sha256=hashlib.sha256(('p'+str(i)).encode()).hexdigest(),confirmed_output_tokens=i)
  h=(torch.arange(4096,dtype=torch.float32)+i).to(torch.bfloat16);jobs[key]=j;values[key]=h;rows.append(dict(j,full_vocab_reconstruction_bitwise_equal=True,raw_top32_candidate_LSE_exact=True,H_sha256=hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()))
 return jobs,values,rows,dict(label_mapping_sha256='fixture-lock',writer_sha256=sha(ROOT/'fkl_shards.py'))

def main():
 torch.set_num_threads(1);assert not torch.cuda.is_available();tests=[];details=[]
 jobs,values,rows,context=fixture()
 with tempfile.TemporaryDirectory() as td:
  base=pathlib.Path(td)
  def crash(name):
   out=base/name;cmd=[sys.executable,'-B',__file__,'--crash-after-rename',str(out)]
   child=subprocess.run(cmd,capture_output=True);assert child.returncode==37,(child.returncode,child.stderr)
   assert not (out/'manifest.jsonl').exists() and len(list((out/'shards').iterdir()))==1
   return out,next((out/'shards').iterdir())
  out,shard=crash('rename');before={p.name:sha(p) for p in shard.iterdir()};r=recover(out,jobs,context)
  assert r['seen']==set(jobs) and len(r['admitted'])==1 and before=={p.name:sha(p) for p in shard.iterdir()};tests.append('actual_exit_after_rename_orphan_validated_and_admitted_without_recompute')
  original=(out/'manifest.jsonl').read_bytes();again=recover(out,jobs,context);assert not again['admitted'] and (out/'manifest.jsonl').read_bytes()==original;tests.append('repeated_resume_is_idempotent_no_duplicate_prefix_or_manifest')
  keys=list(jobs);part={keys[0]:jobs[keys[0]]};out=base/'partial';out.mkdir();publish(out,{keys[0]:values[keys[0]]},[rows[0]],part,context,set())
  committed=(out/'manifest.jsonl').read_bytes();tail=b'{"path":"cut-off';(out/'manifest.jsonl').write_bytes(committed+tail)
  r=recover(out,part,context);assert r['seen']==set(part) and (out/'manifest.jsonl').read_bytes()==committed
  receipt=read(r['tail']['receipt_path']);preserved=pathlib.Path(r['tail']['receipt_path']).parent
  assert (preserved/'tail.bin').read_bytes()==tail and (preserved/'original_manifest.bin').read_bytes()==committed+tail;tests.append('half_final_record_preserved_then_atomic_trim_keeps_committed_prefix')
  again=recover(out,part,context);assert again['tail'] is None and (out/'manifest.jsonl').read_bytes()==committed;tests.append('repeated_tail_repair_does_not_rewrite_manifest')
  out,shard=crash('tail_orphan');e=read(shard/'seal.json')['entry'];(out/'manifest.jsonl').write_bytes(canonical(e)[:17]);r=recover(out,jobs,context)
  assert len(r['admitted'])==1 and r['tail'];validate_manifest(out/'manifest.jsonl',jobs,complete=True);tests.append('partial_append_after_rename_recovers_one_whole_verified_record')
  out,shard=crash('full_no_newline');e=read(shard/'seal.json')['entry'];(out/'manifest.jsonl').write_bytes(canonical(e));r=recover(out,jobs,context)
  assert len(r['admitted'])==1 and len((out/'manifest.jsonl').read_text().splitlines())==1;tests.append('complete_record_without_newline_preserved_and_recovered_once')
  for mode in ['data','invalid_resigned_safetensor','metadata','frozen_label','context','row_count','prefix_set','missing_receipt']:
   out,shard=crash('invalid_'+mode);seal=read(shard/'seal.json')
   if mode=='data':
    p=shard/'hidden.safetensors';p.write_bytes(p.read_bytes()+b'X')
   elif mode=='invalid_resigned_safetensor':
    p=shard/'hidden.safetensors';p.write_bytes(b'invalid tensor');seal['entry']['sha256']=sha(p);write(shard/'seal.json',seal)
   elif mode=='metadata':
    p=shard/'metadata.jsonl';p.write_bytes(p.read_bytes()+b' ')
   elif mode=='frozen_label':
    altered=copy.deepcopy(rows);altered[0]['expected_id']+=100;p=shard/'metadata.jsonl';p.write_bytes(b''.join(canonical(x)+b'\n' for x in altered));seal['entry']['metadata_sha256']=sha(p);write(shard/'seal.json',seal)
   elif mode=='context':seal['context']['label_mapping_sha256']='wrong-lock';write(shard/'seal.json',seal)
   elif mode=='row_count':seal['entry']['rows']+=1;write(shard/'seal.json',seal)
   elif mode=='prefix_set':
    p=shard/'metadata.jsonl';p.write_bytes(b''.join(canonical(x)+b'\n' for x in rows[:-1]));seal['entry'].update(metadata_sha256=sha(p),rows=2);write(shard/'seal.json',seal)
   else:(shard/'seal.json').rename(shard/'untrusted_legacy_receipt.json')
   before={p.name:sha(p) for p in shard.iterdir()};r=recover(out,jobs,context)
   assert not r['seen'] and len(r['quarantined'])==1 and not r['admitted'];q=pathlib.Path(r['quarantined'][0]['path']);assert before=={p.name:sha(p) for p in q.iterdir()};details.append(dict(case=mode,reason=r['quarantined'][0]['reason']))
   publish(out,values,rows,jobs,context,set());r2=recover(out,jobs,context);assert r2['seen']==set(jobs) and not r2['quarantined'];tests.append('orphan_'+mode+'_rejected_preserved_and_recomputed_without_duplicate')
  out=base/'duplicate';out.mkdir();publish(out,values,rows,jobs,context,set());before=sha(out/'manifest.jsonl')
  class SimulatedExit(Exception):pass
  def interrupt(_):raise SimulatedExit()
  try:publish(out,values,rows,jobs,context,set(),interrupt)
  except SimulatedExit:pass
  r=recover(out,jobs,context);assert len(r['quarantined'])==1 and sha(out/'manifest.jsonl')==before;tests.append('overlapping_orphan_quarantined_without_duplicate_manifest')
  shard=pathlib.Path(json.loads((out/'manifest.jsonl').read_text().splitlines()[0])['path']).parent;p=shard/'metadata.jsonl';p.write_bytes(p.read_bytes()+b' ')
  try:recover(out,jobs,context)
  except SharedIntegrityError:tests.append('committed_metadata_corruption_blocks_no_quarantine_or_silent_repair')
  else:raise AssertionError('committed corruption accepted')
  out=base/'bad_middle';out.mkdir();(out/'manifest.jsonl').write_bytes(b'bad-json\n')
  try:recover(out,jobs,context)
  except ValueError:tests.append('newline_terminated_invalid_record_blocks_no_tail_truncation')
  else:raise AssertionError('invalid complete record accepted')
 result=dict(status='pass',count=len(tests),tests=tests,quarantine_rejections=details,GPU_used=False,source_sha256={n:sha(ROOT/n) for n in ['fkl_shards.py','fkl_integrity.py','fkl_all_verified_gpu.py','fkl_shards_cpu_tests.py']},unix=time.time())
 write(ROOT/'FKL_shard_recovery_CPU_receipt.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:
  jobs,values,rows,context=fixture();publish(pathlib.Path(sys.argv[2]),values,rows,jobs,context,set(),lambda _:os._exit(37))
 else:main()
