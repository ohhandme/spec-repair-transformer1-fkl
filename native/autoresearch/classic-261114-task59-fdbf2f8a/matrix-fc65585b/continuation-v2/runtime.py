"""Shared actual CPU/GPU entry for the frozen priority ten; legal signals only."""
import dataclasses,functools,sys
from common import *
sys.path.insert(0,str(MATRIX));sys.path.insert(0,read(WAVE/'freeze.json')['source_root'])
import torch
from safetensors.torch import load_file
from matrix_heads import PRIORITY,build,prepare_cached,forward
from repair_reference import COnlyHead
from feedback_codec.v10_four_arm import V10PreparedState,semantic_current_top32
class FrozenRows:
 def __init__(self,split):
  assert split in ('train','calibration')
  self.records=[r for r in lines(TRAIN/'index.jsonl') if r['split']==split]
  self.windows=[(ri,wi) for ri,r in enumerate(self.records) for wi in range(len(r['windows']))]
 @functools.lru_cache(maxsize=32)
 def cache(self,ri):return load_file(TRAIN/'cache'/(self.records[ri]['request_id']+'.safetensors'),device='cpu')
 @functools.lru_cache(maxsize=64)
 def trace(self,ri):return read(DATA/'shards'/self.records[ri]['request_id']/'trace.json')
 @functools.lru_cache(maxsize=8)
 def tensors(self,ri):return load_file(DATA/'shards'/self.records[ri]['request_id']/'features.safetensors',device='cpu')
 def sample(self,index,group,embedding,device):
  ri,wi=self.windows[index];r=self.records[ri];w=r['windows'][wi];cache=self.cache(ri)
  state=prepare_cached(group,r,w,self.trace(ri),self.tensors(ri) if group=='TOP32_v2' else {},cache,embedding)
  state=dataclasses.replace(state,**{f.name:getattr(state,f.name).to(device) for f in dataclasses.fields(state) if isinstance(getattr(state,f.name),torch.Tensor)})
  history=cache['draft_sequence'][w['offset']:w['offset']+w['length']].unsqueeze(0).to(device)
  labels=torch.tensor(w['labels'],device=device)
  return history,state,labels,ri
def loss_for(head,sample,config,temperature=1.):
 h,s,y,_=sample;logits=forward(head,h,s,temperature=temperature)
 per=torch.nn.functional.cross_entropy(logits.float().reshape(len(y),-1),y,reduction='none')
 weights=per.new_tensor(config['position_weights'][:len(y)])
 return (per*weights).sum()/weights.sum()
def load_head(group,draft,checkpoint=None):
 head=build(group,draft.model.get_output_embeddings(),draft.model.get_input_embeddings())
 state=load_file(MATRIX/'initial'/(group+'.safetensors')) if checkpoint is None else torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
 head.load_trainable_state_dict(state);return head.to(draft.model.get_output_embeddings().weight.device)
def online_state(group,head,row,boundary,draft):
 valid=row['prefix_before_ids']+row['candidate_ids'][:row['accepted_count']]
 answer=row['correction_id'] if row['correction_id'] is not None else row['bonus_id']
 assert row['prefix_after_ids']==valid+[answer] and boundary.token_id==answer
 local=draft.teacher_hidden(row['prefix_after_ids'],[0]).detach()
 base=head.base if isinstance(head,COnlyHead) else head;mode=base.mode
 rejected=correction=accepted=feature=accounting=None
 if mode!='B':
  table=draft.model.get_input_embeddings();accepted=row['accepted_count']
  correction=table.weight[answer].detach().reshape(1,-1)
  if row['rejected_id'] is not None:rejected=table.weight[row['rejected_id']].detach().reshape(1,-1)
  else:assert accepted==len(row['candidate_ids'])
 if mode=='F-CURRENT-TOP32':
  summary=boundary.logit_summary;assert summary is not None and summary.greedy_token_id==answer
  accounting=semantic_current_top32(summary,draft.model.get_input_embeddings());feature=accounting.features
 return V10PreparedState(mode,tuple(row['prefix_after_ids']),local,rejected,correction,accepted,None,feature,accounting)
def online_next(head,prefix,state,position,draft):
 if state is None:return int(draft.model.get_output_embeddings()(draft.teacher_hidden(prefix,[0])).argmax(-1))
 n=len(state.corrected_prefix);assert tuple(prefix[:n])==state.corrected_prefix and len(prefix)-n==position-1
 history=draft.teacher_hidden(list(state.corrected_prefix),prefix[n:]+[0]).unsqueeze(0)
 return int(forward(head,history,state)[:,-1].argmax(-1).item())
