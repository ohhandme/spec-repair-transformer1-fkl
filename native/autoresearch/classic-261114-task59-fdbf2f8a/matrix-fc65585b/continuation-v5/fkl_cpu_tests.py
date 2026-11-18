import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,hashlib,json,pathlib,tempfile,time
import torch
from safetensors.torch import save_file
from common import write,sha,ROOT
from guards import SharedIntegrityError
from fkl_integrity import validate_entry,validate_manifest,coverage_inputs
def main():
 tests=[];torch.set_num_threads(1);assert not torch.cuda.is_available()
 def reject(name,fn):
  try:fn()
  except SharedIntegrityError:tests.append(name);return
  raise AssertionError('accepted invalid fixture: '+name)
 with tempfile.TemporaryDirectory() as td:
  p=pathlib.Path(td);jobs={};tensors={};rows=[]
  for i in range(3):
   key=hashlib.sha256(str(i).encode()).hexdigest();j=dict(prefix_id=key,expected_id=10+i,split='train',request_id='req'+str(i),prefix_sha256=hashlib.sha256(('prefix'+str(i)).encode()).hexdigest(),confirmed_output_tokens=i);jobs[key]=j
   h=torch.arange(4096,dtype=torch.float32).to(torch.bfloat16)+i;tensors[key]=h
   rows.append(dict(j,full_vocab_reconstruction_bitwise_equal=True,raw_top32_candidate_LSE_exact=True,H_sha256=hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()))
  data=p/'hidden.safetensors';meta=p/'metadata.jsonl';manifest=p/'manifest.jsonl';save_file(tensors,str(data))
  def save_rows(rs):meta.write_text(''.join(json.dumps(r)+'\n' for r in rs))
  save_rows(rows);original=meta.read_bytes();entry=dict(path=str(data),sha256=sha(data),metadata_path=str(meta),metadata_sha256=sha(meta),rows=3)
  manifest.write_text(json.dumps(entry)+'\n');terminal=dict(status='complete',manifest_sha256=sha(manifest),prefixes=3)
  assert validate_manifest(manifest,jobs,complete=True,terminal=terminal)['seen']==set(jobs);tests.append('intact_resume_data_metadata_counts_prefixes_labels_pass')
  changed=copy.deepcopy(rows);changed[0]['expected_id']+=1;save_rows(changed)
  reject('metadata_only_replacement_rejected_with_data_unchanged',lambda:validate_entry(entry,jobs))
  e=dict(entry,metadata_sha256=sha(meta));reject('resigned_metadata_wrong_frozen_label_rejected',lambda:validate_entry(e,jobs))
  meta.write_bytes(original);reject('metadata_row_count_rejected',lambda:validate_entry(dict(entry,rows=2),jobs))
  save_rows(rows+[rows[0]]);reject('duplicate_prefix_ID_rejected',lambda:validate_entry(dict(entry,metadata_sha256=sha(meta),rows=4),jobs))
  save_rows(rows[:2]);reject('data_metadata_prefix_set_mismatch_rejected',lambda:validate_entry(dict(entry,metadata_sha256=sha(meta),rows=2),jobs));meta.write_bytes(original)
  reject('same_prefix_in_two_shards_rejected',lambda:validate_entry(entry,jobs,{rows[0]['prefix_id']}))
  reject('missing_frozen_prefix_on_complete_rejected',lambda:validate_manifest(manifest,dict(jobs,extra={'prefix_id':'extra'}),complete=True))
  reject('terminal_manifest_hash_rejected',lambda:validate_manifest(manifest,jobs,complete=True,terminal=dict(terminal,manifest_sha256='0'*64)))
  reject('terminal_prefix_count_rejected',lambda:validate_manifest(manifest,jobs,complete=True,terminal=dict(terminal,prefixes=2)))
  t=copy.deepcopy(tensors);t[rows[0]['prefix_id']][0]+=5;save_file(t,str(data))
  reject('data_only_replacement_rejected',lambda:validate_entry(entry,jobs))
  reject('resigned_data_H_digest_rejected',lambda:validate_entry(dict(entry,sha256=sha(data)),jobs));save_file(tensors,str(data))
  allp=p/'all_unique_prefixes.jsonl';allp.write_text(''.join(json.dumps(x)+'\n' for x in jobs.values()));occ=p/'all_occurrences.jsonl'
  refs=[dict(prefix_id=x['prefix_id'],expected_id=x['expected_id'],split='train',request_id=x['request_id'],window_index=0,label_position0=0,valid=True) for x in rows]
  occ.write_text(''.join(json.dumps(x)+'\n' for x in refs));write(p/'completion.json',{'status':'complete_CPU_mapping','files':{x.name:{'path':str(x),'sha256':sha(x)} for x in (allp,occ)}})
  accepted,b=coverage_inputs(p,3,3);assert set(accepted)==set(jobs);tests.append('all_label_frozen_mapping_pass')
  refs[0]['expected_id']+=1;occ.write_text(''.join(json.dumps(x)+'\n' for x in refs));write(p/'completion.json',{'status':'complete_CPU_mapping','files':{x.name:{'path':str(x),'sha256':sha(x)} for x in (allp,occ)}})
  reject('frozen_label_mapping_expected_ID_rejected',lambda:coverage_inputs(p,3,3))
 write(ROOT/'FKL_resume_CPU_receipt.json',dict(status='pass',tests=tests,count=len(tests),GPU_used=False,source_sha256=sha(ROOT/'fkl_integrity.py'),unix=time.time()))
 print(json.dumps({'status':'pass','tests':tests}),flush=True)
if __name__=='__main__':main()
