"""Actual CUDA entry for the remaining 20 conditions. CPU fixtures use the same engines."""
import argparse,sys,time
from common import *
import heads as H
import checkpoints as CP
from runtime20 import Rows,DonorPool
from engine20 import train,evaluate
class Production:
 fixture=False;interrupt_at=None;eval_interrupt_after=None;checkpoint_interval=256
 def __init__(self):
  self.root=ROOT;self.device='cuda:0';self.draft=None;self.target=None;self._rows={};self._teacher=None;self._donors=None;self.f=read(WAVE/'freeze.json')
  sys.path.insert(0,str(PARENT));from closed_loop_core import decode
  self.decode=decode
 def load_draft(self):
  if self.draft is None:
   import torch
   from feedback_codec.hf_backend import _FrozenHFBackend,HFDraftBackend
   class DraftOnly(_FrozenHFBackend):teacher_hidden=HFDraftBackend.teacher_hidden
   self.draft=DraftOnly(str(pathlib.Path(self.f['models_root'])/'Qwen3-0.6B'),None,device=self.device,dtype=torch.bfloat16)
 def head(self,group,weights=None):
  self.load_draft()
  if weights is None:_,weights=CP.binding(ROOT,group,'initial',initial=True)
  head=H.build(group,self.draft.model.get_output_embeddings(),self.draft.model.get_input_embeddings()).to(self.device);head.load_trainable_state_dict(weights);return head
 def eval_models(self):
  import torch
  from feedback_codec.hf_backend import HFTargetBackend,assert_compatible_pair
  self.load_draft()
  if self.target is None:self.target=HFTargetBackend(str(pathlib.Path(self.f['models_root'])/'Qwen3-8B'),None,device=self.device,dtype=torch.bfloat16)
  assert_compatible_pair(self.target,self.draft)
 def rows(self,split):
  if split not in self._rows:self._rows[split]=Rows(split)
  return self._rows[split]
 def orders(self,c):return read(MATRIX/'epoch_orders.json')
 def teacher(self,group):
  if group not in H.FKL:return None
  if self._teacher is None:
   import torch
   from safetensors import safe_open
   from full_vocab_supervision import FullVocabularySupervision
   folder=pathlib.Path(self.f['models_root'])/'Qwen3-8B';config=read(folder/'config.json');key='model.embed_tokens.weight' if config.get('tie_word_embeddings') else 'lm_head.weight'
   index=folder/'model.safetensors.index.json';file=folder/read(index)['weight_map'][key] if index.exists() else folder/'model.safetensors'
   require(read(ROOT/'runtime_freeze.json')['files'].get(str(file))==sha(file),'Target weight source not frozen')
   with safe_open(file,framework='pt',device='cpu') as f:weight=f.get_tensor(key)
   require(weight.dtype==torch.bfloat16 and weight.shape==(151936,4096),'Target output head precision/shape differs')
   head=torch.nn.Linear(4096,151936,bias=False,device='meta');head.weight=torch.nn.Parameter(weight.to(self.device),requires_grad=False)
   self._teacher=FullVocabularySupervision(head,dict(source_path=str(file),source_sha256=sha(file),tensor_key=key))
  return self._teacher
 def roster(self):return read(WAVE/'roster503.json')
 def roster_sha(self):return sha(WAVE/'roster503.json')
 def smoke_ids(self):return read(WAVE/'smoke_ids.json')
 def validate_request(self,req):
  from runtime20 import digest
  require(self.roster_sha()==self.f['roster_sha256'],'503 roster changed')
  oracle=read(WAVE/'oracles'/(req['request_id']+'.json'));require(oracle['output_ids']==req['frozen_target_ids'] and oracle['prompt_sha256']==digest(req['prompt_ids']) and oracle['freeze_sha256']==sha(WAVE/'freeze.json'),'independent Target oracle binding differs')
  require(sha(req['raw_trace_path'])==req['raw_trace_sha256'],'RAW source trace SHA differs');r=read(req['raw_trace_path'])
  require(r['trajectory_policy']=='RAW' and not r['repair_enabled'] and r['prompt_ids']==req['prompt_ids'] and r['output_ids']==req['frozen_target_ids'],'RAW/Target request binding differs')
 def raw_first_candidates(self,req):return read(req['raw_trace_path'])['rounds'][0]['candidate_ids']
 def target_step(self,group,prefix):return (self.target.next_with_logit_summary if group in ('RANK64-MARGIN','SCORE-VQ64-W') else self.target.next)(prefix)
 def independent(self,req):
  from feedback_codec.decoding import autoregressive_generate
  return list(autoregressive_generate(self.target,req['prompt_ids'],128,151645))
 def donors(self,group):
  if group not in ('R_SHUFFLED','H-VQ64-weighted-MISMATCH'):return None
  if self._donors is None:self._donors=DonorPool(self.rows('calibration'),read(ROOT/'online_donors.json'))
  return self._donors
 def donor_lock_sha(self,group):return sha(ROOT/'online_donors.json') if group in ('R_SHUFFLED','H-VQ64-weighted-MISMATCH') else None
 def resources(self):
  import torch
  return dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,Target_loaded=self.target is not None)
def main():
 p=argparse.ArgumentParser();p.add_argument('--group',choices=H.GROUPS,required=True);p.add_argument('--phase',choices=['smoke','train','eval'],required=True);a=p.parse_args()
 verify_freeze();require(read(ROOT/'CPU_E2E_receipt.json')['status']=='pass','CPU end-to-end not passed');c=read(ROOT/'configs'/(a.group+'.json'))
 require(c['training_windows']==156688 and c['calibration_windows']==19736 and c['epochs']==5 and c['updates']==783440 and c['max_new_tokens']==128,'production lock differs')
 require(not (c['evaluation_only'] and a.phase=='train'),'intervention training forbidden')
 out=ROOT/'runs'/a.group/a.phase;out.mkdir(parents=True,exist_ok=True);own=open(out/'worker.lock','a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if a.group in H.FKL:require(read(ROOT/'fkl/backfill/completion.json').get('FKL_training_admitted') is True,'five FKL conditions still await full vocabulary acceptance')
 if a.phase=='train':require(read(ROOT/'runs'/a.group/'smoke/completion.json')['status']=='complete','initial real smoke absent')
 if a.phase=='eval':
  for s in read(ROOT/'runs'/c['checkpoint_owner']/'train/selected.json'):CP.binding(ROOT,a.group,s)
 if c['evaluation_only']:
  for s in read(ROOT/'runs'/c['checkpoint_owner']/'train/selected.json'):CP.binding(ROOT,a.group,s)
 write(out/'identity.json',dict(identity(),group=a.group,phase=a.phase))
 try:
  with lease(out):
   env=Production()
   if a.phase=='train':train(a.group,out,env)
   else:evaluate(a.group,out,env,a.phase=='smoke')
 except Exception as e:first_error(out/'first_error.json',e);raise
if __name__=='__main__':main()
