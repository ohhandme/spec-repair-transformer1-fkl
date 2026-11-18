"""Actual cached training and each arm's own full-request legal-boundary adapter."""
import dataclasses,functools,hashlib,json,math,sys
from common import *
import heads as H
import torch
from safetensors.torch import load_file
from feedback_codec.v10_four_arm import V10PreparedState
from h_vq64_head import MessageState
from rank64_head import RankState
from teacher_loop import RoundContext,TeacherMessage
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def tensor_digest(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def move(x,device):
 if isinstance(x,torch.Tensor):return x.to(device)
 if dataclasses.is_dataclass(x):return dataclasses.replace(x,**{f.name:move(getattr(x,f.name),device) for f in dataclasses.fields(x)})
 if isinstance(x,tuple):return tuple(move(v,device) for v in x)
 if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
 return x
class Rows:
 def __init__(self,split):
  require(split in ('train','calibration'),'forbidden split')
  self.split=split;self.records=[r for r in lines(TRAIN/'index.jsonl') if r['split']==split]
  self.windows=[(ri,wi) for ri,r in enumerate(self.records) for wi in range(len(r['windows']))]
  self.donor_path=V3/'remaining_cpu_v2'/(split+'_donors.json');self.donors=read(self.donor_path)
 @functools.lru_cache(maxsize=24)
 def cache(self,ri):return load_file(TRAIN/'cache'/(self.records[ri]['request_id']+'.safetensors'),device='cpu')
 @functools.lru_cache(maxsize=32)
 def trace(self,ri):return read(DATA/'shards'/self.records[ri]['request_id']/'trace.json')
 @functools.lru_cache(maxsize=4)
 def tensors(self,ri):return load_file(DATA/'shards'/self.records[ri]['request_id']/'features.safetensors',device='cpu')
 def donor(self,index):
  ri,wi=self.windows[index];r=self.records[ri];return r,r['windows'][wi],self.trace(ri),self.cache(ri)
 def sample(self,index,group,head,embedding,device):
  ri,wi=self.windows[index];r=self.records[ri];w=r['windows'][wi];cache=self.cache(ri);donor=None;pair=None
  if group=='R_SHUFFLED':donor=self.donor(self.donors['mapping'][group][index]);pair=sha(self.donor_path)
  state,extra=H.prepare(group,head,r,w,self.trace(ri),self.tensors(ri) if group in ('SCORE-VQ64-W','RANK64-MARGIN') else {},cache,embedding,donor,pair)
  history=cache['draft_sequence'][w['offset']:w['offset']+w['length']].unsqueeze(0).to(device)
  return dict(history=history,state=move(state,device),extra=move(extra,device),labels=torch.tensor(w['labels'],device=device),record_index=ri,request_id=r['request_id'],window_index=wi)
def objective(group,head,sample,cfg,teacher=None,calibrate=False):
 z=H.forward(group,head,sample['history'],sample['state'],sample['extra'],None if calibrate else 1.)
 y=sample['labels'];weights=z.new_tensor(cfg['position_weights'][:len(y)],dtype=torch.float32)
 ce=torch.nn.functional.cross_entropy(z.float().reshape(len(y),-1),y,reduction='none')
 if cfg['loss']!='CE' and not calibrate:
  require(teacher is not None,'FKL requires admitted full vocabulary teacher')
  tz=teacher.logits(sample['request_id'],sample['window_index'],y)
  require(tz.shape==z.shape and z.shape[-1]==151936,'FKL is not full vocabulary')
  p=tz.detach().float().log_softmax(-1);q=z.float().log_softmax(-1);kl=(p.exp()*(p-q)).sum(-1).reshape(-1)
  per=.25*ce+.75*kl if cfg['loss']=='CE25FKL75' else kl
 else:per=ce
 require(bool(torch.isfinite(per).all()),'nonfinite training/calibration objective')
 return (per*weights).sum()/weights.sum()
class DonorPool:
 """R uses full-H arbitrary cross-family donors; 8B additionally matches ACK condition."""
 def __init__(self,rows,plan):self.rows=rows;self.plan=plan;self.by_key={w['observation_key']:(ri,wi) for ri,r in enumerate(rows.records) for wi,w in enumerate(r['windows'])}
 def get(self,group,request,row):
  rid=request['request_id'];ordinal=row['round'];require(0<=ordinal<127,'online donor round exceeds frozen budget')
  if group=='R_SHUFFLED':key=self.plan['R_SHUFFLED'][rid][ordinal]
  else:
   bucket=str(row['accepted_count'])+':'+str(row['rejected_id'] is None)
   require(bucket in self.plan['MISMATCH'][rid],'no same-condition 8B donor pool')
   choices=self.plan['MISMATCH'][rid][bucket];key=choices[ordinal%len(choices)]
  ri,wi=self.by_key[key];r=self.rows.records[ri];w=r['windows'][wi]
  require(r['split']=='calibration' and r['request_id']!=rid and r['family_id']!=request['family_id'],'online donor crosses partition/family boundary')
  if group!='R_SHUFFLED':require(w['accepted']==row['accepted_count'] and (w['rejected'] is None)==(row['rejected_id'] is None),'8B donor condition differs')
  return r,w,self.rows.trace(ri),self.rows.cache(ri),dict(kind='full_H_cross_family' if group=='R_SHUFFLED' else '8B_same_ACK_condition_cross_family',request_id=r['request_id'],family_id=r['family_id'],observation_key=key,split='calibration',accepted=w['accepted'],rejected_is_none=w['rejected'] is None)
class Online:
 def __init__(self,group,head,draft,request,donors=None,donor_lock_sha=None):
  self.group=group;self.head=head;self.draft=draft;self.request=request;self.donors=donors;self.donor_lock_sha=donor_lock_sha;self.signals={}
 def prepare(self,row,boundary):
  group=self.group;head=self.head;table=self.draft.model.get_input_embeddings()
  valid=row['prefix_before_ids']+row['candidate_ids'][:row['accepted_count']];answer=row['correction_id'] if row['correction_id'] is not None else row['bonus_id']
  require(boundary.token_id==answer and list(boundary.source_prefix)==valid and row['prefix_after_ids']==valid+[answer],'online teacher is not this arm actual correction/bonus boundary')
  require(row['valid_teacher_prefix_sha256']==digest(valid),'online teacher prefix digest differs')
  local=self.draft.teacher_hidden(row['prefix_after_ids'],[0]).detach();rejected=None if row['rejected_id'] is None else table.weight[row['rejected_id']].detach().reshape(1,-1);correction=table.weight[answer].detach().reshape(1,-1)
  s=V10PreparedState('C',tuple(row['prefix_after_ids']),local,rejected,correction,row['accepted_count'],None,None,None);extra={};payload=None;donor_meta=None
  h=boundary.hidden.detach().reshape(1,4096);require(bool(torch.isfinite(h).all()),'online Target hidden nonfinite')
  ctx=RoundContext(self.request['request_id'],self.request['family_id'],'dev',row['round'],tuple(valid),s.corrected_prefix,read(WAVE/'freeze.json')['models_manifest_sha256'])
  if group in H.FKL or group=='C-HARD-ST':state=s
  elif group.startswith('F-H'):
   require(h.dtype==torch.bfloat16,'frozen Target boundary dtype differs');state=dataclasses.replace(s,mode='F-H',target_hidden=h);payload=h.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
  elif group.startswith('R_'):
   msg=None
   if group=='R_FULL':msg=TeacherMessage.full_hidden(h.to(torch.bfloat16),ctx)
   if group=='R_SHUFFLED':
    dr,dw,dt,dc,donor_meta=self.donors.get(group,self.request,row);msg=TeacherMessage.full_hidden(H.legal_hidden(dr,dw,dt,dc),H.context(dr,dw),ctx,pairing_lock_sha256=self.donor_lock_sha)
   state=s;extra=dict(context=ctx,message=msg);payload=None if msg is None else msg.payload
  elif group=='LOCAL-LUT-W':state=MessageState(s,(),None,s.corrected_prefix)
  else:
   if group in ('RANK64-MARGIN','SCORE-VQ64-W'):
    summary=boundary.logit_summary;require(summary is not None and summary.greedy_token_id==answer,'online current Top32 summary missing')
    pairs=sorted(zip(summary.top_token_ids,summary.top_logits_fp16),key=lambda x:(-x[1],x[0]));ids=[x[0] for x in pairs];values=[x[1] for x in pairs];lse=float(summary.logsumexp_fp32)
    top=dict(token_ids=ids,logits=values,greedy_token_id=answer,logsumexp=lse,vocab_size=151936)
    self.signals[f'r{row["round"]:04}.top32_ids']=torch.tensor(ids);self.signals[f'r{row["round"]:04}.top32_logits']=torch.tensor(values);self.signals[f'r{row["round"]:04}.logsumexp']=torch.tensor([lse])
   if group=='RANK64-MARGIN':
    import rank64_head
    clip=read(V3/'remaining_cpu_v2/rank_fit.json')['train_p99_margin'];rank64_head.MARGIN_CLIP=clip
    payload=H.encode_from_current_topk(top,protocol_b=answer,margin_clip=clip);state=RankState(s,payload,s.corrected_prefix)
   else:
    feature=h
    if group=='SCORE-VQ64-W':
     by=dict(pairs);alt=[x for x in pairs if x[0]!=answer];v=[math.exp(by[answer]-lse),math.exp(alt[0][1]-lse),math.exp(alt[1][1]-lse),sum(math.exp(z-lse) for z in values)]
     fit=read(V2/'preprocessing/score_preprocess_v1.json');feature=torch.zeros_like(h);feature[0,:4]=torch.tensor([(x-m)/s for x,m,s in zip(v,fit['mean'],fit['scale'])],device=h.device,dtype=h.dtype)
    inputs=(feature,row['accepted_count'],rejected,correction);payload=head.message.encode_payload(*inputs);state=MessageState(s,inputs,payload,s.corrected_prefix)
    if group in H.INTERVENTIONS:
     donor_payload=None
     if group.endswith('MISMATCH'):
      dr,dw,dt,dc,donor_meta=self.donors.get(group,self.request,row);ds=H.prepare_cached('C',dr,dw,dt,{},dc,table)
      donor_payload=head.message.encode_payload(H.legal_hidden(dr,dw,dt,dc),dw['accepted'],ds.rejected_embedding,ds.correction_embedding)
     state=H.intervention(head,state,group,donor_payload);payload=state.payload
  self.signals[f'r{row["round"]:04}.boundary_hidden']=h.detach().cpu().flatten().clone()
  row['feedback_evidence']=dict(valid_prefix_sha256=digest(valid),corrected_prefix_sha256=digest(row['prefix_after_ids']),Target_boundary_H_sha256=tensor_digest(h),physical_teacher_bytes=0 if payload is None else len(payload),payload_sha256=None if payload is None else hashlib.sha256(payload).hexdigest(),payload_hex=payload.hex() if payload is not None and len(payload)<=8 else None,donor=donor_meta,donor_lock_sha256=self.donor_lock_sha if donor_meta else None)
  return move((state,extra),table.weight.device)
 def next(self,prefix,prepared,position):
  if prepared is None:return int(self.draft.model.get_output_embeddings()(self.draft.teacher_hidden(prefix,[0])).argmax(-1))
  state,extra=prepared;n=len(state.corrected_prefix);require(tuple(prefix[:n])==state.corrected_prefix and len(prefix)-n==position-1,'Draft history not on own corrected prefix')
  history=self.draft.teacher_hidden(list(state.corrected_prefix),prefix[n:]+[0]).unsqueeze(0)
  z=H.forward(self.group,self.head,history,state,extra,None);return int(z[:,-1].argmax(-1))
