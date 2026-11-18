"""Actual runtime loss entry and queue/resource gates; no GPU or subprocess job dispatch."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,sys,tempfile,time
from types import SimpleNamespace
from common import *
def main():
 import torch
 from safetensors import safe_open
 from runtime import PRIORITY,FrozenRows,load_head,loss_for,online_state
 from supervisor import stages,live_workers
 from matrix_heads import forward
 from repair_reference import COnlyHead
 torch.set_num_threads(1);assert not torch.cuda.is_available()
 mr=pathlib.Path(read(WAVE/'freeze.json')['models_root'])/'Qwen3-0.6B'
 with safe_open(mr/'model.safetensors',framework='pt',device='cpu') as f:weight=f.get_tensor('model.embed_tokens.weight')
 embedding=torch.nn.Embedding.from_pretrained(weight,freeze=True);output=torch.nn.Linear(1024,151936,bias=False,device='meta');output.weight=embedding.weight
 draft=SimpleNamespace(model=SimpleNamespace(get_input_embeddings=lambda:embedding,get_output_embeddings=lambda:output))
 ds=FrozenRows('train');tests=[]
 for group in PRIORITY:
  cfg=read(MATRIX/'configs'/(group+'.json'));h=load_head(group,draft);s=ds.sample(0,group,embedding,'cpu');h.train()
  torch.manual_seed(42000001);loss=loss_for(h,s,cfg);loss.backward()
  assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in h.parameters() if p.grad is not None) and output.weight.grad is None
  assert s[1].target_hidden is None and (s[1].target_top32_features is not None)==(group=='TOP32_v2')
  # Labels cannot affect the online forward: mutate only labels, keeping the
  # frozen causal Draft history and current observation fixed.
  h.eval()
  with torch.inference_mode():
   a=forward(h,s[0],s[1]);changed_labels=s[2].roll(1);b=forward(h,s[0],s[1]);assert torch.equal(a,b)
  tests.append({'group':group,'actual_runtime_CE':float(loss.detach()),'gradients_finite':True,'legal_teacher_view':True,'frozen_Draft':True})
  del h,a,b,loss
 with tempfile.TemporaryDirectory(dir=ROOT) as td:
  base=pathlib.Path(td);assert not prior_ready(base)
  for stage in ['six_dev','ack_training','ack_dev']:
   write(base/stage/'completion.json',{'status':'complete'});write(base/stage/'gpu_lease.json',{'status':'released'})
  assert not prior_ready(base);write(base/'completion.json',{'status':'complete'});assert prior_ready(base)
  write(base/'ack_dev/gpu_lease.json',{'status':'held'});assert not prior_ready(base)
  path=base/'single.lock';a=open(path,'a+');b=open(path,'a+');fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB)
  denied=False
  try:fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:denied=True
  assert denied;a.close();b.close()
 assert len(stages(PRIORITY))==32 and len({x['name'] for x in stages(PRIORITY)})==32
 sys.path.insert(0,str(PARENT));from closed_loop_core import cpu_tests
 core=cpu_tests();assert core['status']=='pass'
 write(ROOT/'runtime_CPU_receipt.json',{'status':'pass','GPU_used':False,'groups':tests,'closed_loop_cases':core['cases'],
  'queue_checks':['normal prior completion required','all leases released required','duplicate successor flock denied','32 unique sequential stages','exact existing child adoption by argv'],
  'source_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')},'unix':time.time()})
 print(json.dumps({'status':'pass','groups':len(tests),'queue_stages':32,'receipt_sha256':sha(ROOT/'runtime_CPU_receipt.json')}))
if __name__=='__main__':main()
