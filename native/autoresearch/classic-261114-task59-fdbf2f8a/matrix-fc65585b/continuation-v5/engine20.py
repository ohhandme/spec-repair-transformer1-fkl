"""Shared production/CPU-fixture train, resume, smoke and full-request execution."""
import gc,json,math,os,pathlib,time
from types import SimpleNamespace
import torch
from common import read,write,sha,require
import checkpoints as CP
from runtime20 import objective,Online
class InjectedCPUInterruption(RuntimeError):pass
def cfg(env,g):return read(env.root/'configs'/(g+'.json'))
def train(group,out,env):
 c=cfg(env,group);require(not c['evaluation_only'],'intervention has no training job');lock=sha(env.root/'configs'/(group+'.json'));freeze=sha(env.root/'runtime_freeze.json')
 if (out/'completion.json').exists():
  for selection in read(out/'selected.json'):CP.binding(env.root,group,selection)
  return
 head=env.head(group);embedding=env.draft.model.get_input_embeddings();ds=env.rows('train');cal=env.rows('calibration');teacher=env.teacher(group)
 require(len(ds.windows)==c['training_windows'] and len(cal.windows)==c['calibration_windows'],'actual data budget differs')
 opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],**{k:v for k,v in c['optimizer'].items() if k!='kind'})
 state=dict(epoch=0,position=0,updates=0,loss_sum=0.,train_seconds=0.,calibration_seconds=0.,history=[],calibration=[],best=None,best_epoch=None)
 def save(tag):return CP.save(out,dict(state,group=group,config_sha256=lock,runtime_freeze_sha256=freeze,model=head.trainable_state_dict(),optimizer=opt.state_dict()),tag)
 if (out/'latest.json').exists():
  old=CP.load(read(out/'latest.json'),group,lock,freeze);head.load_trainable_state_dict(old['model']);opt.load_state_dict(old['optimizer'])
  require(old['updates']==old['epoch']*c['training_windows']+old['position'],'resume update/epoch/position invariant differs')
  for k in state:state[k]=old[k]
  write(out/'resume_receipt.json',dict(updates=state['updates'],epoch=state['epoch'],position=state['position'],checkpoint_sha256=read(out/'latest.json')['sha256']))
 else:save('latest')
 orders=env.orders(c);require(len(orders)==c['epochs'] and all(sorted(x)==list(range(c['training_windows'])) for x in orders),'training order not frozen permutation')
 while state['epoch']<c['epochs']:
  ep=state['epoch'];head.train()
  while state['position']<c['training_windows']:
   pos=state['position'];started=time.perf_counter();sample=ds.sample(orders[ep][pos],group,head,embedding,env.device)
   torch.manual_seed(42000000+ep*c['training_windows']+pos+1);opt.zero_grad(set_to_none=True);value=objective(group,head,sample,c,teacher);value.backward()
   require(bool(torch.isfinite(value)) and all(bool(torch.isfinite(p.grad).all()) for p in head.parameters() if p.grad is not None),'nonfinite training loss/gradient')
   opt.step();state['position']+=1;state['updates']+=1;state['loss_sum']+=float(value.detach());state['train_seconds']+=time.perf_counter()-started
   if state['updates']==1:write(out/'first_update.json',dict(group=group,loss=float(value.detach()),config_sha256=lock,initial_sha256=c['initial_sha256'],window_index=orders[ep][pos],CPU_fixture=env.fixture))
   if state['updates']==1 or state['position']%env.checkpoint_interval==0:save('latest');write(out/'status.json',dict(state,phase='train',unix=time.time()))
   if env.fixture and env.interrupt_at==state['updates']:save('latest');raise InjectedCPUInterruption('CPU fixture interruption after durable checkpoint')
  head.eval();sums=[0.]*len(cal.records);counts=[0]*len(cal.records);started=time.perf_counter()
  with torch.inference_mode():
   for i in range(len(cal.windows)):
    sample=cal.sample(i,group,head,embedding,env.device);v=float(objective(group,head,sample,c,calibrate=True));require(math.isfinite(v),'nonfinite calibration CE');ri=sample['record_index'];sums[ri]+=v;counts[ri]+=1
  require(all(counts),'calibration request missing legal windows');metric=sum(s/n for s,n in zip(sums,counts))/len(counts)
  record=dict(epoch=ep+1,calibration_request_equal_CE=metric,requests=[dict(request_id=r['request_id'],windows=n,mean_CE=s/n) for r,s,n in zip(cal.records,sums,counts)])
  write(out/f'calibration_epoch{ep+1}.json',record);state['calibration_seconds']+=time.perf_counter()-started;state['calibration'].append(dict(epoch=ep+1,value=metric,receipt_sha256=sha(out/f'calibration_epoch{ep+1}.json')))
  state['history'].append(dict(epoch=ep+1,mean_training_objective=state['loss_sum']/c['training_windows'],updates=state['updates']))
  if state['best'] is None or metric<state['best']:state['best']=metric;state['best_epoch']=ep+1
  state.update(epoch=ep+1,position=0,loss_sum=0.);save('epoch'+str(ep+1));save('latest')
 require(state['updates']==c['updates'],'completed training update count differs')
 selected={'main':dict(read(out/('epoch'+str(c['epochs'])+'.json')),rule='frozen_final_epoch'),'secondary':dict(read(out/('epoch'+str(state['best_epoch'])+'.json')),rule='minimum_calibration_request_equal_CE')}
 write(out/'selected.json',selected);write(out/'completion.json',dict(status='complete',group=group,updates=state['updates'],selected=selected,config_sha256=lock,runtime_freeze_sha256=freeze,train_seconds=state['train_seconds'],calibration_seconds=state['calibration_seconds'],CPU_fixture=env.fixture,resources=env.resources(),unix=time.time()))
def validate_request_files(dest,req,bound):
 r=read(dest/'receipt.json');require(r['checkpoint_binding']==bound,'sealed request checkpoint binding differs')
 for name,digest in r['files'].items():require(sha(dest/name)==digest,'sealed request file SHA differs: '+name)
 require(r['output_ids']==req['frozen_target_ids'] and r['eos']==req['eos'] and r['truncated']==req['truncated'],'sealed request ID/EOS/truncation differs')
 return r
def evaluate(group,out,env,smoke):
 from safetensors.torch import save_file
 c=cfg(env,group);initial=smoke and not c['evaluation_only'];owner=c['checkpoint_owner'];selections=['initial'] if initial else list(read(env.root/'runs'/owner/'train/selected.json'))
 roster=env.roster();smoke_ids=env.smoke_ids();ordered=[r for rid in smoke_ids for r in roster if r['request_id']==rid]
 require(len(ordered)==len(smoke_ids),'smoke roster differs')
 if not smoke:ordered += [r for r in roster if r['request_id'] not in smoke_ids]
 index={};aliases=[];visited={};summaries={};already_complete=(out/'completion.json').exists()
 if already_complete:
  terminal=read(out/'completion.json');require(terminal['status']=='complete' and terminal['config_sha256']==sha(env.root/'configs'/(group+'.json')) and terminal['runtime_freeze_sha256']==sha(env.root/'runtime_freeze.json'),'evaluation terminal binding differs')
 for selection in selections:
  bound,weights=CP.binding(env.root,group,selection,initial=initial)
  if bound['checkpoint_sha256'] in visited:aliases.append(dict(selection=selection,equivalent_to=visited[bound['checkpoint_sha256']]));continue
  visited[bound['checkpoint_sha256']]=selection;head=None;sealed=[]
  for req in ordered:
   CP.binding(env.root,group,selection,expected=bound,initial=initial);env.validate_request(req);dest=out/selection/req['request_id']
   if (dest/'receipt.json').exists():
    r=validate_request_files(dest,req,bound);sealed.append(r)
   else:
    require(not already_complete,'completed evaluation lacks sealed request')
    if head is None:env.eval_models();head=env.head(group,weights);head.eval()
    # Independent greedy is re-run on smoke cases before any repaired decoding.
    fresh=req['request_id'] in smoke_ids
    if fresh:require(env.independent(req)==req['frozen_target_ids'],'fresh independent Target differs')
    tmp=out/'incomplete'/(selection+'-'+req['request_id']+'-'+str(time.time_ns()));tmp.mkdir(parents=True);rows=[]
    online=Online(group,head,env.draft,req,env.donors(group),env.donor_lock_sha(group));started=time.perf_counter()
    def target_step(prefix):
     step=env.target_step(group,prefix)
     return SimpleNamespace(token_id=step.token_id,hidden=step.hidden,logit_summary=getattr(step,'logit_summary',None),source_prefix=tuple(prefix))
    def capture(row,boundary):
     rows.append(row)
     with (tmp/'rounds.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
    with torch.inference_mode():result=env.decode(req['prompt_ids'],draft_token=online.next,target_step=target_step,prepare=online.prepare,on_round=capture,oracle_ids=req['frozen_target_ids'],max_new=c['max_new_tokens'],window=8,eos=151645)
    require(result['eos']==req['eos'] and result['truncated']==req['truncated'],'live ID/EOS/truncation differs')
    require(rows[0]['candidate_ids']==env.raw_first_candidates(req),'first round original Draft differs')
    save_file(online.signals,str(tmp/'signals.safetensors'))
    CP.binding(env.root,group,selection,expected=bound,initial=initial)
    r={k:v for k,v in result.items() if k!='rounds'};r.update(status='pass',request_id=req['request_id'],family_id=req['family_id'],group=group,selection=selection,checkpoint_binding=bound,files={n:sha(tmp/n) for n in ['rounds.jsonl','signals.safetensors']},fresh_independent_Target_checked=fresh,elapsed_seconds=time.perf_counter()-started,CPU_fixture=env.fixture,unix=time.time())
    write(tmp/'receipt.json',r);dest.parent.mkdir(parents=True,exist_ok=True);require(not dest.exists(),'request already sealed');tmp.rename(dest);sealed.append(r)
   index[selection+'/'+req['request_id']]=dict(path=str(dest),receipt_sha256=sha(dest/'receipt.json'))
   # Atomic index also adopts a sealed request orphaned between rename/index.
   if not already_complete:write(out/'sealed_index.json',index);write(out/'status.json',dict(group=group,selection=selection,sealed=len(sealed),total=len(ordered),unix=time.time()))
   if env.fixture and env.eval_interrupt_after==len(sealed):raise InjectedCPUInterruption('CPU eval interruption after request seal')
  require(all(r['fresh_independent_Target_checked'] for r in sealed if r['request_id'] in smoke_ids),'smoke independent Target receipt absent')
  summary=dict(requests=len(sealed),accepted_draft_tokens=sum(r['accepted_draft_tokens'] for r in sealed),verification_rounds=sum(r['verification_rounds'] for r in sealed),elapsed_seconds=sum(r['elapsed_seconds'] for r in sealed))
  summary['accepted_tokens_per_round']=summary['accepted_draft_tokens']/summary['verification_rounds'];summaries[selection]=summary
  if not already_complete:write(out/(selection+'_summary.json'),summary)
  if not already_complete:write(out/(selection+'_smoke.json'),dict(status='pass',requests=smoke_ids,ID_EOS_truncation_exact=True,first_round_RAW_exact=True,checkpoint_binding=bound,CPU_fixture=env.fixture))
  del head,weights;gc.collect()
 if already_complete:
  require(sha(out/'sealed_index.json')==terminal['sealed_index_sha256'] and read(out/'sealed_index.json')==index,'completed sealed index differs');return
 write(out/'completion.json',dict(status='complete',group=group,phase='smoke' if smoke else 'eval',selection_aliases=aliases,summaries=summaries,sealed_index_sha256=sha(out/'sealed_index.json'),roster_sha256=env.roster_sha(),config_sha256=sha(env.root/'configs'/(group+'.json')),runtime_freeze_sha256=sha(env.root/'runtime_freeze.json'),CPU_fixture=env.fixture,resources=env.resources(),independent300_used=False,unix=time.time()))
