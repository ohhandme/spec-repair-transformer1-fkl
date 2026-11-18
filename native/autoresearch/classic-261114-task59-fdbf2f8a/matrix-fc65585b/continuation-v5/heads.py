"""CPU preparation entry for remaining conditions; no implicit GPU admission."""
import dataclasses,sys
from common import *
sys.path[:0]=[str(V3/'originals20'),str(MATRIX),read(WAVE/'freeze.json')['source_root']]
import torch
from feedback_codec.codec import CodecConfig
from feedback_codec.v10_four_arm import V10FourArmHead
from matrix_heads import build as priority_build,prepare_cached,forward as priority_forward
from h_vq64_head import HTaskHead,MessageState
from task23_heads_v2 import FixedPCAQ64,LocalLUTHead,VariableOfferedVQ64,ScoreHead,_short_full_forward
from c_hard_st_head import CHardSTHead
from rank64_head import Rank64Head,RankState
from teacher_loop import TeacherRepairHead,RoundContext,TeacherMessage
from wire64 import encode_from_current_topk
import rank64_consumer
rank64_consumer.WIRE_SOURCE=V3/'originals20/wire64.py'
GROUPS=['F-H','F-H-weighted','MLP-C-FKL-I0','TR1-C-FKL-I0','TR2-W64-C-FKL','TR2-W64-C-CE25FKL75','MLP-C-FKL-EQUAL',
 'R_LOCAL_PROXY','R_FULL','R_SHUFFLED','RANK64-MARGIN','H-VQ64-TASK','C-HARD-ST','H-VQ64-weighted','H-VQ64-weighted-ZERO-ID',
 'H-VQ64-weighted-NO-CONTROL','H-VQ64-weighted-MISMATCH','H-PCAQ64-W','SCORE-VQ64-W','LOCAL-LUT-W']
FKL={'MLP-C-FKL-I0':'MLP-C-CE-I0','TR1-C-FKL-I0':'TR1-C-CE-I0','TR2-W64-C-FKL':'TR2-W64-C-CE','TR2-W64-C-CE25FKL75':'TR2-W64-C-CE','MLP-C-FKL-EQUAL':'MLP-C-CE-I0'}
INTERVENTIONS={'H-VQ64-weighted-ZERO-ID','H-VQ64-weighted-NO-CONTROL','H-VQ64-weighted-MISMATCH'}
class TaskHead(HTaskHead):
 def __init__(self,config,output_head,*,draft_embedding,kind):
  super().__init__(config,output_head,draft_embedding=draft_embedding);self.kind=kind
  if kind=='H-PCAQ64-W':self.message=FixedPCAQ64(self.message.codebook,V2/'preprocessing/pca_preprocess_v1.pt')
  else:
   msg=VariableOfferedVQ64(target_hidden_size=4096,draft_hidden_size=1024,conditioned=True);msg.load_state_dict(self.message.state_dict());self.message=msg
 def forward_logits(self,h,s,pos,*,temperature=None):
  if s.base.rejected_embedding is None and s.base.accepted_count<8:return _short_full_forward(self,h,s,pos,temperature)
  return super().forward_logits(h,s,pos,temperature=temperature)
def build(group,output,embedding):
 require(group in GROUPS,'unknown remaining condition')
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(42);c=CodecConfig(4096,1024,151936)
  if group in FKL:return priority_build(FKL[group],output,embedding)
  if group.startswith('F-H'):return V10FourArmHead(c,output,mode='F-H',draft_embedding=embedding)
  if group.startswith('R_'):
   base=V10FourArmHead(c,output,mode='C',draft_embedding=embedding)
   return TeacherRepairHead(base,'mlp',{'R_LOCAL_PROXY':'local_proxy','R_FULL':'matched','R_SHUFFLED':'shuffled'}[group])
  if group=='C-HARD-ST':return CHardSTHead(c,output,draft_embedding=embedding)
  if group=='RANK64-MARGIN':return Rank64Head(c,output,draft_embedding=embedding,payloads={})
  if group=='LOCAL-LUT-W':return LocalLUTHead(c,output,draft_embedding=embedding)
  return TaskHead(c,output,draft_embedding=embedding,kind=group)
def context(r,w):
 prefix=tuple(r['prompt_ids']+r['output_ids'][:w['offset']])
 return RoundContext(r['request_id'],r['family_id'],'train' if r['split']=='train' else 'dev',w['round'],prefix[:-1],prefix,read(WAVE/'freeze.json')['models_manifest_sha256'])
def legal_hidden(r,w,t,cache):
 rr=t['rounds'][w['round']];prefix=r['prompt_ids']+r['output_ids'][:w['offset']]
 require(rr['hidden']['present'] and rr['hidden']['absolute_position']==len(prefix)-2,'teacher H not at previous valid-prefix boundary')
 require(rr['observation_key']==w['observation_key'],'teacher observation differs')
 wi=r['windows'].index(w);return cache['target'][wi].reshape(1,4096)
def topk(r,w,t,tensors):
 rr=t['rounds'][w['round']];key=rr['tensor_prefix'];j=rr['top32']['used_rows']-1
 require(rr['top32']['greedy_ids'][j]==w['correction'] and rr['top32']['absolute_source_positions'][j]==len(r['prompt_ids'])+w['offset']-2,'Top32 boundary differs')
 ids=tensors[key+'.top32_ids'][j].tolist();values=tensors[key+'.top32_logits'][j].float().tolist();pairs=sorted(zip(ids,values),key=lambda x:(-x[1],x[0]))
 return dict(token_ids=[x[0] for x in pairs],logits=[x[1] for x in pairs],greedy_token_id=w['correction'],logsumexp=float(tensors[key+'.logsumexp'][j]),vocab_size=151936)
def intervention(head,state,group,donor=None):
 require(not head.training,'three interventions are evaluation only with identical weighted checkpoint')
 if group.endswith('ZERO-ID'):return head.with_payload(state,b'\0'*8)
 if group.endswith('NO-CONTROL'):return head.with_payload(state,None)
 require(group.endswith('MISMATCH') and donor is not None,'mismatch donor not bound')
 return head.with_payload(state,donor)
def prepare(group,head,r,w,t,tensors,cache,embedding,donor=None,pair_sha=None):
 s=prepare_cached('C',r,w,t,tensors,cache,embedding);extra={}
 if group in FKL or group=='C-HARD-ST':return s,extra
 if group.startswith('F-H'):return dataclasses.replace(s,mode='F-H',target_hidden=legal_hidden(r,w,t,cache)),extra
 if group.startswith('R_'):
  ctx=context(r,w);msg=None
  if group=='R_FULL':msg=TeacherMessage.full_hidden(legal_hidden(r,w,t,cache),ctx)
  if group=='R_SHUFFLED':
   require(donor is not None and pair_sha is not None,'locked R donor absent');dr,dw,dt,dc=donor
   msg=TeacherMessage.full_hidden(legal_hidden(dr,dw,dt,dc),context(dr,dw),ctx,pairing_lock_sha256=pair_sha)
  return s,dict(context=ctx,message=msg)
 if group=='RANK64-MARGIN':
  fit=read(V3/'remaining_cpu_v2/rank_fit.json');import rank64_head;rank64_head.MARGIN_CLIP=fit['train_p99_margin']
  payload=encode_from_current_topk(topk(r,w,t,tensors),protocol_b=w['correction'],margin_clip=fit['train_p99_margin'])
  return RankState(s,payload,s.corrected_prefix),extra
 if group=='LOCAL-LUT-W':return MessageState(s,(),None,s.corrected_prefix),extra
 h=legal_hidden(r,w,t,cache)
 if group=='SCORE-VQ64-W':
  import math
  k=topk(r,w,t,tensors);pairs=list(zip(k['token_ids'],k['logits']));b=w['correction'];alt=[x for x in pairs if x[0]!=b];byid=dict(pairs);lse=k['logsumexp']
  v=[math.exp(byid[b]-lse),math.exp(alt[0][1]-lse),math.exp(alt[1][1]-lse),sum(math.exp(z-lse) for z in k['logits'])]
  fit=read(V2/'preprocessing/score_preprocess_v1.json');normalized=[(x-m)/s for x,m,s in zip(v,fit['mean'],fit['scale'])]
  h=torch.zeros_like(h);h[0,:4]=torch.tensor(normalized,dtype=h.dtype)
 inputs=(h,w['accepted'],s.rejected_embedding,s.correction_embedding);payload=head.message.encode_payload(*inputs)
 state=MessageState(s,inputs,payload,s.corrected_prefix)
 if group in INTERVENTIONS:
  if group.endswith('MISMATCH'):
   require(donor is not None and pair_sha is not None,'mismatch donor absent');dr,dw,dt,dc=donor
   require(dr['split']==r['split'] and dr['family_id']!=r['family_id'] and dw['accepted']==w['accepted'] and (dw['rejected'] is None)==(w['rejected'] is None),'mismatch donor condition differs')
   ds=prepare_cached('C',dr,dw,dt,{},dc,embedding);dp=head.message.encode_payload(legal_hidden(dr,dw,dt,dc),dw['accepted'],ds.rejected_embedding,ds.correction_embedding)
  else:dp=None
  state=intervention(head,state,group,dp)
 return state,extra
def forward(group,head,history,state,extra,temperature=1.):
 if group in FKL:return priority_forward(head,history,state,temperature=temperature)
 if group.startswith('R_'):return head.forward_logits(history,state,**extra)
 return head.forward_logits(history,state,torch.arange(1,history.shape[1]+1,device=history.device),temperature=temperature)
def loss(group,logits,labels,teacher_logits=None,valid=None):
 z=logits.float().reshape(-1,logits.shape[-1]);y=labels.reshape(-1)
 require(z.shape[-1]==151936 and len(y)==len(z),'full vocabulary/label dimensions differ')
 valid=torch.ones(len(y),dtype=torch.bool,device=z.device) if valid is None else valid.reshape(-1)
 require(bool(valid.any()) and torch.isfinite(z).all(),'invalid logits or empty legal mask')
 weighted=group not in ('F-H','H-VQ64-TASK','C-HARD-ST','RANK64-MARGIN','MLP-C-FKL-EQUAL')
 weights=torch.arange(8,8-len(y),-1,device=z.device,dtype=torch.float32) if weighted else torch.ones(len(y),device=z.device)
 weights=weights*valid;ce=torch.nn.functional.cross_entropy(z,y,reduction='none')
 if group in FKL:
  require(teacher_logits is not None and teacher_logits.shape==logits.shape and torch.isfinite(teacher_logits).all(),'full finite teacher logits required; Top32 approximation forbidden')
  logp=teacher_logits.detach().float().reshape_as(z).log_softmax(-1);logq=z.log_softmax(-1);kl=(logp.exp()*(logp-logq)).sum(-1)
  per=.25*ce+.75*kl if group.endswith('CE25FKL75') else kl
 else:per=ce
 return (weights*per).sum()/weights.sum()
