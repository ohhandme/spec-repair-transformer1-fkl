"""Frozen first-wave task23 heads; only the declared signal or local capacity changes."""
from dataclasses import replace
from pathlib import Path
import json
import torch
from torch import nn
from h_vq64_head import HTaskHead,MessageState
from h_vq64_codec import ConditionedVQ64
from feedback_codec.v10_four_arm import V10PreparedState
from causal_operands_task23_v1 import causal_inputs
ROOT=Path(__file__).resolve().parent
class VariableOfferedVQ64(ConditionedVQ64):
 def _inputs(self, target_hidden, accepted_count, rejected_embedding, correction_embedding):
  if rejected_embedding is None and 0 <= accepted_count < 8:
   rejected_embedding=torch.zeros_like(correction_embedding)
  return super()._inputs(target_hidden,accepted_count,rejected_embedding,correction_embedding)

def _base_prepare(head,obs,draft):
 if obs.rejected_token is not None or obs.accepted_count==8:
  return head.base.prepare_observation(obs,draft)
 if obs.rejected_token is None and obs.accepted_count==len(obs.proposals) and 0<=obs.accepted_count<8:
  if obs.target_hidden_prefix!=obs.valid_prefix or obs.valid_prefix!=obs.source_prefix+obs.proposals or obs.corrected_prefix!=obs.valid_prefix+(obs.correction_token,) or obs.remaining_budget<=0:
   raise ValueError('invalid short-full causal boundary')
  table=draft.model.get_input_embeddings()
  if table is None or any(p.requires_grad for p in table.parameters()):raise ValueError('Draft table not frozen')
  if head.base._draft_embedding_ref is not None and head.base._draft_embedding_ref() is not table:raise ValueError('Draft table binding differs')
  local=draft.teacher_hidden(list(obs.corrected_prefix),[0]).detach().clone()
  if local.shape!=(1,head.config.draft_hidden_size) or not bool(torch.isfinite(local).all()):raise ValueError('local hidden invalid')
  correction=head.base._protocol_embedding(table,obs.correction_token)
  return V10PreparedState('C',obs.corrected_prefix,local,None,correction,obs.accepted_count,None,None,None)
 raise ValueError('offered/accepted short-full mismatch')

def _short_full_forward(head,draft_hidden,state,position,temperature):
 base=state.base;local=base.local_hidden
 rejected=base.rejected_embedding if base.rejected_embedding is not None else local
 if temperature is None:codes=head.base.core.local_encode(local,rejected,base.correction_embedding,base.accepted_count)
 else:codes=head.base.core.local_sample_st(local,rejected,base.correction_embedding,base.accepted_count,temperature=temperature)
 control=head.base.core.repair._lookup(codes[:,0],1)+head.base.core.repair._lookup(codes[:,1],1)
 surrogate=head.base.local_hidden_lift(local.to(device=head.base.local_hidden_lift[0].weight.device,dtype=head.base.local_hidden_lift[0].weight.dtype))
 control=control+head.base.hidden_projection(surrogate)
 semantic=head.base.local_top32_surrogate(local.to(device=head.base.local_top32_surrogate.weight.device,dtype=head.base.local_top32_surrogate.weight.dtype))
 control=control+head.base.top32_projection(semantic)
 msg=head.message.train_control(*state.online_inputs,temperature=temperature or 1.) if head.training else head.message.decode_payload(state.payload)
 control=control+msg.to(device=control.device,dtype=control.dtype)
 logits=head.base.core.repair.repair_from_control(draft_hidden,control,position)
 if not bool(torch.isfinite(logits).all()):raise FloatingPointError('nonfinite short-full logits')
 return logits


class FixedPCAQ64(nn.Module):
 def __init__(self,initial_codebook,preprocess_path):
  super().__init__();p=torch.load(preprocess_path,map_location='cpu',weights_only=False)
  self.codebook=nn.Parameter(initial_codebook.detach().clone())
  for name,key in [('mean','mean'),('basis','basis'),('low','low'),('high','high'),('degenerate','degenerate')]:self.register_buffer(name,p[key].detach().clone())
  assert self.codebook.shape==(8,256,16) and self.basis.shape==(4096,8)
  self.target_hidden_size=4096
 def quantize(self,hidden):
  if hidden.shape!=(1,4096) or not torch.isfinite(hidden).all():raise ValueError('PCA input differs')
  with torch.no_grad():
   z=(hidden.to(self.mean.device,dtype=torch.float32)-self.mean)@self.basis
   den=(self.high-self.low).clamp_min(1e-8)
   q=torch.floor((torch.clamp((z-self.low)/den,0,1)*255)+.5).to(torch.long).flatten()
   q=torch.where(self.degenerate,torch.full_like(q,128),q)
  if q.shape!=(8,) or not bool(((q>=0)&(q<256)).all()):raise ValueError('PCA byte range')
  return q
 def encode_payload(self,hidden,*protocol):return bytes(self.quantize(hidden).cpu().tolist())
 def decode_payload(self,payload):
  if payload is None:return self.codebook.new_zeros((1,128))
  if type(payload) is not bytes or len(payload)!=8:raise ValueError('PCA wire must be 8 raw bytes')
  q=torch.tensor(list(payload),device=self.codebook.device,dtype=torch.long)
  return self.codebook[torch.arange(8,device=q.device),q].reshape(1,128)
 def train_control(self,hidden,*protocol,temperature=1.0):
  q=self.quantize(hidden);return self.codebook[torch.arange(8,device=q.device),q].reshape(1,128)

class PCAQHead(HTaskHead):
 def __init__(self,config,output_head,*,draft_embedding):
  super().__init__(config,output_head,draft_embedding=draft_embedding)
  cb=self.message.codebook.detach().clone();self.message=FixedPCAQ64(cb,ROOT/'pca_preprocess_v1.pt')
  self.mode='H-PCAQ64-W'
 def prepare_observation(self,observation,draft):
  base=_base_prepare(self,observation,draft)
  inputs=causal_inputs(observation,draft.model.get_input_embeddings(),target_hidden_size=self.config.target_hidden_size,draft_hidden_size=self.config.draft_hidden_size,eos=151645)
  return MessageState(base,inputs,self.message.encode_payload(*inputs),base.corrected_prefix)
 def forward_logits(self,draft_hidden,state,position,*,temperature=None):
  if state.base.rejected_embedding is None and state.base.accepted_count<8:return _short_full_forward(self,draft_hidden,state,position,temperature)
  return super().forward_logits(draft_hidden,state,position,temperature=temperature)
 def parameter_budget(self):
  base=self.base.parameter_budget();return {'matched_C_edge_trainable':int(base['edge_trainable']),'extra_edge_codebook_trainable':32768,'cloud_encoder_trainable':0,'physical_wire_bytes':8,'edge_codebook_control_dim':128}

class ScoreHead(HTaskHead):
 def __init__(self,config,output_head,*,draft_embedding):
  super().__init__(config,output_head,draft_embedding=draft_embedding)
  self.score_rows={}
  for split in ('train','val'):
   rows=json.loads((ROOT/('score_'+split+'_index_v1.json')).read_text())['rows']
   assert not (set(rows)&set(self.score_rows));self.score_rows.update(rows)
  assert len(self.score_rows)==9580
  msg=VariableOfferedVQ64(target_hidden_size=config.target_hidden_size,draft_hidden_size=config.draft_hidden_size,conditioned=True)
  msg.load_state_dict(self.message.state_dict());self.message=msg
  self.mode='SCORE-VQ64-W' 
 def prepare_observation(self,observation,draft):
  base=_base_prepare(self,observation,draft)
  original,accepted,rejected,correction=causal_inputs(observation,draft.model.get_input_embeddings(),target_hidden_size=self.config.target_hidden_size,draft_hidden_size=self.config.draft_hidden_size,eos=151645)
  key=observation.prompt_id+':'+str(observation.round_index)
  if key not in self.score_rows:raise ValueError('unlocked same-boundary SCORE row')
  padded=torch.zeros_like(original);padded[0,:4]=torch.tensor(self.score_rows[key],device=padded.device,dtype=padded.dtype)
  inputs=(padded,accepted,rejected,correction);payload=self.message.encode_payload(*inputs)
  return MessageState(base=base,online_inputs=inputs,payload=payload,corrected_prefix=base.corrected_prefix)
 def forward_logits(self,draft_hidden,state,position,*,temperature=None):
  if state.base.rejected_embedding is None and state.base.accepted_count<8:return _short_full_forward(self,draft_hidden,state,position,temperature)
  return super().forward_logits(draft_hidden,state,position,temperature=temperature)

class LocalCodebook(nn.Module):
 def __init__(self,initial_codebook):
  super().__init__();self.codebook=nn.Parameter(initial_codebook.detach().clone());assert self.codebook.shape==(8,256,16)
 def control(self,codes):
  cb=self.codebook;rows=torch.arange(8,device=cb.device)
  if codes.ndim==3:
   assert codes.shape==(1,2,8)
   indices=torch.stack([codes[:,g%2,g] for g in range(8)],dim=1)
   return cb[rows,indices[0]].reshape(1,128)
  if codes.ndim==4:
   assert codes.shape==(1,2,8,256)
   weights=torch.stack([codes[:,g%2,g,:] for g in range(8)],dim=1)
   return torch.einsum('bgc,gcd->bgd',weights,cb).reshape(1,128)
  raise ValueError('LOCAL code shape differs')

class LocalLUTHead(HTaskHead):
 def __init__(self,config,output_head,*,draft_embedding):
  super().__init__(config,output_head,draft_embedding=draft_embedding)
  cb=self.message.codebook.detach().clone();self.message=LocalCodebook(cb)
  self.mode='LOCAL-LUT-W'
 def prepare_observation(self,observation,draft):
  base=_base_prepare(self,observation,draft)
  return MessageState(base=base,online_inputs=(),payload=None,corrected_prefix=base.corrected_prefix)
 def with_payload(self,state,payload):
  if payload is not None:raise ValueError('LOCAL has no teacher wire')
  return replace(state,payload=None)
 def forward_logits(self,draft_hidden,state,position,*,temperature=None):
  if type(state) is not MessageState or state.base.mode!='C':raise TypeError('LOCAL state differs')
  base=state.base;local=base.local_hidden
  rejected=base.rejected_embedding if base.rejected_embedding is not None else local
  if temperature is None:codes=self.base.core.local_encode(local,rejected,base.correction_embedding,base.accepted_count)
  else:codes=self.base.core.local_sample_st(local,rejected,base.correction_embedding,base.accepted_count,temperature=temperature)
  control=self.base.core.repair._lookup(codes[:,0],1)+self.base.core.repair._lookup(codes[:,1],1)
  surrogate=self.base.local_hidden_lift(local.to(device=self.base.local_hidden_lift[0].weight.device,dtype=self.base.local_hidden_lift[0].weight.dtype))
  control=control+self.base.hidden_projection(surrogate)
  semantic=self.base.local_top32_surrogate(local.to(device=self.base.local_top32_surrogate.weight.device,dtype=self.base.local_top32_surrogate.weight.dtype))
  control=control+self.base.top32_projection(semantic)
  control=control+self.message.control(codes).to(device=control.device,dtype=control.dtype)
  logits=self.base.core.repair.repair_from_control(draft_hidden,control,position)
  if not torch.isfinite(logits).all():raise FloatingPointError('LOCAL logits nonfinite')
  return logits
 def parameter_budget(self):
  base=self.base.parameter_budget();return {'matched_C_edge_trainable':int(base['edge_trainable']),'extra_edge_codebook_trainable':32768,'cloud_encoder_trainable':0,'physical_wire_bytes':0,'edge_codebook_control_dim':128}
