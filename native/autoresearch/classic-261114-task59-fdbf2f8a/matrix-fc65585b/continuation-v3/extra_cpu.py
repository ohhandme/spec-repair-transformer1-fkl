"""Exercise real runtime loading and supervisor observation plus intervention semantics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import gc,subprocess,sys,tempfile,time
from types import SimpleNamespace
from common import *
def main():
 import torch
 from safetensors import safe_open
 from runtime import PRIORITY,FrozenRows,load_head,loss_for
 from guards import initial_binding
 import remaining20_a100 as ext
 import supervisor_final as supervisor
 torch.set_num_threads(1);require(not torch.cuda.is_available(),'GPU visibility violated');tests=[]
 # The initial binding needs a runtime digest; use a private root, never the
 # production runtime freeze, which is only sealed after these tests pass.
 with tempfile.TemporaryDirectory() as td:
  testroot=pathlib.Path(td);write(testroot/'runtime_freeze.json',{'test_fixture':True})
  mr=pathlib.Path(read(WAVE/'freeze.json')['models_root'])/'Qwen3-0.6B'
  with safe_open(mr/'model.safetensors',framework='pt',device='cpu') as sf:weight=sf.get_tensor('model.embed_tokens.weight')
  embedding=torch.nn.Embedding.from_pretrained(weight,freeze=True);output=torch.nn.Linear(1024,151936,bias=False,device='meta');output.weight=embedding.weight
  draft=SimpleNamespace(model=SimpleNamespace(get_input_embeddings=lambda:embedding,get_output_embeddings=lambda:output));ds=FrozenRows('train')
  for group in PRIORITY:
   bound,state=initial_binding(testroot,MATRIX,group);head=load_head(group,draft,verified_state=state);sample=ds.sample(0,group,embedding,'cpu');value=loss_for(head,sample,read(MATRIX/'configs'/(group+'.json')))
   require(torch.isfinite(value),'priority CPU loss differs');tests.append({'case':'production_runtime_verified_bytes_load','group':group,'initial_sha256':bound['checkpoint_sha256'],'loss':float(value.detach())})
   del head,state,value;gc.collect()
  try:load_head('B',draft,checkpoint='/unverified/path.pt')
  except SharedIntegrityError:pass
  else:raise SharedIntegrityError('production runtime allowed unchecked checkpoint path')
  tests.append({'case':'production_runtime_rejects_unverified_path'})
  # Actual supervisor inspect_terminal, actual live process, fixture lease.
  job={'optional_fkl':True,'argv':[sys.executable,'-c','import time;time.sleep(30)'],'out':str(testroot/'stage')}
  out=pathlib.Path(job['out']);write(out/'first_error.json',{'category':'fkl_specific'});write(out/'gpu_lease.json',{'status':'released'})
  supervisor.kernel_gpu_free=lambda:True
  child=subprocess.Popen(job['argv'])
  try:require(supervisor.inspect_terminal(job)[0]=='wait_cleanup','supervisor continued while real child alive')
  finally:child.terminate();child.wait(timeout=10)
  require(supervisor.inspect_terminal(job,1)[0]=='blocked_fkl_continue_ce','supervisor failed clean FKL continuation')
  write(out/'gpu_lease.json',{'status':'held'});require(supervisor.inspect_terminal(job,1)[0]=='wait_cleanup','supervisor ignored held lease receipt')
  write(out/'gpu_lease.json',{'status':'released'});supervisor.kernel_gpu_free=lambda:False
  require(supervisor.inspect_terminal(job,1)[0]=='wait_cleanup','supervisor ignored kernel lease')
  supervisor.kernel_gpu_free=lambda:True;write(out/'first_error.json',{'category':'shared_integrity'})
  require(supervisor.inspect_terminal(job,1)[0]=='block_queue','supervisor ignored shared integrity error');tests.append({'case':'production_supervisor_real_child_cleanup_error_classification'})
  # Three payload interventions use exactly one frozen initialization/state.
  ri,wi=ds.windows[0];r=ds.records[ri];w=r['windows'][wi];trace=ds.trace(ri);cache=ds.cache(ri);tensors=ds.tensors(ri)
  head=ext.build('H-VQ64-weighted',output,embedding);head.eval();state,_=ext.prepare('H-VQ64-weighted',head,r,w,trace,tensors,cache,embedding)
  history=cache['draft_sequence'][w['offset']:w['offset']+w['length']].unsqueeze(0)
  with torch.inference_mode():baseline=ext.forward('H-VQ64-weighted',head,history,state,{},None)
  donorlock=read(ROOT/'remaining_cpu_v2/train_donors.json');di=donorlock['mapping']['H-VQ64-weighted-MISMATCH'][0];dri,dwi=ds.windows[di];dr=ds.records[dri];dw=dr['windows'][dwi];donor=(dr,dw,ds.trace(dri),ds.cache(dri))
  for group in sorted(ext.INTERVENTIONS):
   s,e=ext.prepare(group,head,r,w,trace,tensors,cache,embedding,donor,sha(ROOT/'remaining_cpu_v2/train_donors.json'))
   if group.endswith('ZERO-ID'):require(s.payload==bytes(8),'zero-ID differs')
   elif group.endswith('NO-CONTROL'):require(s.payload is None and torch.count_nonzero(head.message.decode_payload(None))==0,'no-control not zero')
   else:require(dr['family_id']!=r['family_id'] and dr['split']==r['split'] and dw['accepted']==w['accepted'],'mismatch donor illegal')
   with torch.inference_mode():changed=ext.forward(group,head,history,s,e,None)
   tests.append(dict(case='same_weight_intervention',group=group,payload_hex=s.payload.hex() if s.payload is not None else None,baseline_payload_hex=state.payload.hex(),max_logit_delta=float((changed-baseline).abs().max()),checkpoint_owner='H-VQ64-weighted',donor_family=dr['family_id'] if group.endswith('MISMATCH') else None))
  from full_vocab_supervision import FullVocabularySupervision
  try:FullVocabularySupervision(output)
  except SharedIntegrityError as e:require('not completed' in str(e),'unexpected full vocabulary gate failure')
  else:raise SharedIntegrityError('FKL accepted before actual GPU full-prefix completion')
  tests.append({'case':'full_vocab_loader_refuses_unadmitted_FKL'})
 write(ROOT/'extra_CPU_receipt.json',dict(status='pass',tests=tests,GPU_used=False,source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},unix=time.time()))
 print(json.dumps({'status':'pass','tests':len(tests),'receipt_sha256':sha(ROOT/'extra_CPU_receipt.json')}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:first_error(ROOT/'extra_CPU_first_error.json',e);raise
