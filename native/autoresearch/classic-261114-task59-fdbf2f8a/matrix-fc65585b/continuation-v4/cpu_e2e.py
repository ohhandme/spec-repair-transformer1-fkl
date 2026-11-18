"""Actual 20-group engines with small CPU fixtures; explicitly NOT GPU/Target acceptance."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,gc,hashlib,json,pathlib,random,sys,time
from types import SimpleNamespace
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from common import *
import heads as H
import checkpoints as CP
from runtime20 import Rows,DonorPool,digest,Online
from engine20 import train,evaluate,InjectedCPUInterruption
from feedback_codec.decoding import TargetLogitSummary
class Subset:
 def __init__(self,base):self.base=base;self.windows=[base.windows[0]];self.records=[base.records[0]]
 def sample(self,index,*args):
  require(index==0,'fixture subset index differs');x=self.base.sample(0,*args);x['record_index']=0;return x
class SyntheticFullTeacher:
 def __init__(self,rows,output):self.rows=rows;self.output=output
 def logits(self,rid,wi,labels):
  r=self.rows.records[0];w=r['windows'][0];require(r['request_id']==rid and wi==0,'synthetic fixture mapping differs')
  h=self.rows.cache(0)['draft_sequence'][w['offset']:w['offset']+w['length']].unsqueeze(0)
  with torch.inference_mode():return self.output(h).float().roll(1,-1)
class Fixture:
 fixture=True;device='cpu';checkpoint_interval=1;interrupt_at=None;eval_interrupt_after=None
 def __init__(self,root,embedding,output,train_rows,cal_rows,roster,donor_plan):
  self.root=root;self.output=output;self.embedding=embedding;self.tr=train_rows;self.cal=cal_rows;self.original_roster=roster;self.plan=donor_plan;self._donors=DonorPool(cal_rows,donor_plan);self._active=None;self.fresh_calls=0;self.target_calls=0;self.draft_calls=0
  cache=train_rows.cache(0);r=train_rows.records[0];w=r['windows'][0];self.h=cache['draft_sequence'][w['offset']].reshape(1,1024);self.th=cache['target'][0].reshape(1,4096)
  self.raw_id=int(output(self.h).argmax(-1));self.other_id=(self.raw_id+13)%151000
  self.draft=SimpleNamespace(model=SimpleNamespace(get_input_embeddings=lambda:embedding,get_output_embeddings=lambda:output),teacher_hidden=self.hidden)
  sys.path.insert(0,str(PARENT));from closed_loop_core import decode
  self.decode=decode
 def hidden(self,prefix,tokens):self.draft_calls+=1;return self.h.repeat(len(tokens),1)
 def head(self,g,weights=None):
  if weights is None:_,weights=CP.binding(self.root,g,'initial',initial=True)
  h=H.build(g,self.output,self.embedding);h.load_trainable_state_dict(weights);return h
 def rows(self,s):return Subset(self.tr if s=='train' else self.cal)
 def orders(self,c):return [[0] for _ in range(c['epochs'])]
 def teacher(self,g):return SyntheticFullTeacher(self.tr,self.output) if g in H.FKL else None
 def eval_models(self):pass
 def roster(self):
  values=[]
  for i,r in enumerate(self.original_roster[:2]):
   seq=[self.other_id,151645] if i==0 else [self.other_id]*3
   values.append(dict(request_id=r['request_id'],family_id=r['family_id'],prompt_ids=r['prompt_ids'],frozen_target_ids=seq,eos=i==0,truncated=i==1))
  return values
 def smoke_ids(self):return [x['request_id'] for x in self.roster()]
 def roster_sha(self):return digest(self.roster())
 def validate_request(self,r):require(r in self.roster(),'CPU fixture roster differs');self._active=r
 def raw_first_candidates(self,r):return [self.raw_id]*3
 def independent(self,r):self._active=r;self.fresh_calls+=1;return list(r['frozen_target_ids'])
 def target_step(self,g,prefix):
  r=self._active;n=len(prefix)-len(r['prompt_ids']);seq=r['frozen_target_ids'];require(prefix==r['prompt_ids']+seq[:n],'fixture Target called on unconfirmed prefix');token=seq[n];self.target_calls+=1
  ids=[token]+[i for i in range(32) if i!=token][:31];values=[2.-i*.1 for i in range(32)]
  summary=TargetLogitSummary.from_json(dict(greedy_token_id=token,top_token_ids=ids,top_logits_fp16=values,logsumexp_fp32=5.,top2_margin_fp32=.1,vocab_size=151936))
  return SimpleNamespace(token_id=token,hidden=self.th.flatten(),logit_summary=summary)
 def donors(self,g):return self._donors if g in ('R_SHUFFLED','H-VQ64-weighted-MISMATCH') else None
 def donor_lock_sha(self,g):return sha(ROOT/'online_donors.json') if self.donors(g) else None
 def resources(self):return dict(CPU_fixture=True,GPU_used=False,synthetic_Target_only=True)
def main():
 torch.set_num_threads(1);require(not torch.cuda.is_available(),'CPU GPU visibility violated');started=time.monotonic()
 root=ROOT/'cpu_e2e_attempts'/str(time.time_ns());root.mkdir(parents=True);(root/'configs').mkdir();write(root/'runtime_freeze.json',dict(CPU_fixture=True,production_source={p.name:sha(p) for p in ROOT.glob('*.py')}))
 for g in H.GROUPS:
  c=read(ROOT/'configs'/(g+'.json'));c.update(training_windows=1,calibration_windows=1,epochs=2,updates=2,max_new_tokens=3,CPU_fixture=True);write(root/'configs'/(g+'.json'),c)
 mr=pathlib.Path(read(WAVE/'freeze.json')['models_root'])/'Qwen3-0.6B'
 with safe_open(mr/'model.safetensors',framework='pt',device='cpu') as f:weight=f.get_tensor('model.embed_tokens.weight')
 embedding=torch.nn.Embedding.from_pretrained(weight,freeze=True);output=torch.nn.Linear(1024,151936,bias=False,device='meta');output.weight=embedding.weight
 tr=Rows('train');cal=Rows('calibration');roster=read(WAVE/'roster503.json');plan=read(ROOT/'online_donors.json');results=[]
 order=[g for g in H.GROUPS if g not in H.INTERVENTIONS]+sorted(H.INTERVENTIONS)
 for g in order:
  env=Fixture(root,embedding,output,tr,cal,roster,plan);run=root/'runs'/g
  # Intervention smoke waits for and loads the actual same weighted checkpoint.
  smoke=run/'smoke';smoke.mkdir(parents=True);evaluate(g,smoke,env,True)
  if g not in H.INTERVENTIONS:
   out=run/'train';out.mkdir();env.interrupt_at=1
   try:train(g,out,env)
   except InjectedCPUInterruption:pass
   else:raise AssertionError('training interruption not exercised')
   require(read(out/'latest.json')['updates']==1,'resume pointer not committed at injected stop');env.interrupt_at=None;train(g,out,env)
   require(read(out/'completion.json')['updates']==2 and read(out/'resume_receipt.json')['updates']==1,'training resume duplicated/lost an update')
   CP.binding(root,g,'main')
  out=run/'eval';out.mkdir();env.eval_interrupt_after=1
  try:evaluate(g,out,env,False)
  except InjectedCPUInterruption:pass
  else:raise AssertionError('evaluation interruption not exercised')
  partial=read(out/'sealed_index.json');require(len(partial)==1,'fixture first request not sealed');first=next(iter(partial.values()));first_sha=first['receipt_sha256'];env.eval_interrupt_after=None
  evaluate(g,out,env,False);require(sha(pathlib.Path(first['path'])/'receipt.json')==first_sha,'resume rewrote already sealed request')
  before=sha(out/'completion.json');calls=(env.target_calls,env.draft_calls);evaluate(g,out,env,False);require(sha(out/'completion.json')==before and calls==(env.target_calls,env.draft_calls),'complete resume redispatched or changed terminal')
  terminal=read(out/'completion.json');require(all(x['requests']==2 for x in terminal['summaries'].values()),'fixture eval lost request')
  entries=read(out/'sealed_index.json');evidence=[]
  for e in entries.values():
   for row in lines(pathlib.Path(e['path'])/'rounds.jsonl'):
    if 'feedback_evidence' in row:evidence.append(row['feedback_evidence'])
  require(evidence,'no actual boundary feedback exercised')
  if g=='R_SHUFFLED':require(all(x['physical_teacher_bytes']==8192 and x['donor']['kind']=='full_H_cross_family' for x in evidence),'R_SHUFFLED not full H')
  if g.endswith('MISMATCH'):require(all(x['physical_teacher_bytes']==8 and x['donor']['kind']=='8B_same_ACK_condition_cross_family' for x in evidence),'MISMATCH not conditioned 8B')
  if g in H.INTERVENTIONS:
   for label in read(root/'runs/H-VQ64-weighted/train/selected.json'):
    a,_=CP.binding(root,g,label);b,_=CP.binding(root,'H-VQ64-weighted',label);require(a['checkpoint_sha256']==b['checkpoint_sha256'] and a['training_completion_sha256']==b['training_completion_sha256'],'interventions do not share same actual weighted checkpoint')
  result=dict(group=g,status='pass_CPU_E2E',training_resume_tested=g not in H.INTERVENTIONS,eval_resume_and_no_redispatch=True,checkpoint_owner=read(root/'configs'/(g+'.json'))['checkpoint_owner'],eval_terminal_sha256=sha(out/'completion.json'),own_boundary_feedback=len(evidence),synthetic_Target=True,synthetic_full_vocab_teacher=g in H.FKL)
  results.append(result);print(json.dumps(result),flush=True);del env;gc.collect()
 # New checkpoint guards, tested against actual engine-produced artifacts.
 victim='F-H';selected=root/'runs'/victim/'train/selected.json';original=selected.read_bytes();x=read(selected);terminalpath=selected.parent/'completion.json';oldterminal=terminalpath.read_bytes();term=read(terminalpath)
 x['main']['sha256']='0'*64;write(selected,x);term['selected']=x;write(terminalpath,term)
 try:CP.binding(root,victim,'main')
 except SharedIntegrityError:pass
 else:raise AssertionError('wrong SHA accepted in production checkpoint guard')
 selected.write_bytes(original);terminalpath.write_bytes(oldterminal)
 write(ROOT/'CPU_E2E_receipt.json',dict(status='pass',groups=results,conditions=20,training_conditions=17,interventions_shared_actual_weighted_checkpoint=True,checkpoint_wrong_SHA_rejected=True,artifact_root=str(root),
  scope='real production engines, real frozen cache/heads and CPU checkpoint/resume; small fixture budgets and synthetic Target/full-vocab supervision test interfaces only',GPU_admission=False,GPU_used=False,independent300_used=False,elapsed_seconds=time.monotonic()-started,source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},unix=time.time()))
 print(json.dumps({'status':'pass','conditions':20,'receipt_sha256':sha(ROOT/'CPU_E2E_receipt.json')}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:first_error(ROOT/'CPU_E2E_first_error.json',e);raise
