"""One priority group, one GPU lease, durable training and strict full requests."""
import argparse,collections,gc,math,sys,time,traceback
from common import *
from guards import training_binding,initial_binding
def train(group,out):
 import torch
 from runtime import FrozenRows,load_head,loss_for
 from feedback_codec.hf_backend import _FrozenHFBackend
 config=read(MATRIX/'configs'/(group+'.json'));torch.set_num_threads(4)
 draft=_FrozenHFBackend(str(pathlib.Path(read(WAVE/'freeze.json')['models_root'])/'Qwen3-0.6B'),None,device='cuda:0',dtype=torch.bfloat16)
 head=load_head(group,draft);embedding=draft.model.get_input_embeddings()
 ds=FrozenRows('train');cal=FrozenRows('calibration');assert len(ds.windows)==156688 and len(cal.windows)==19736
 opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
 lock_sha=sha(MATRIX/'configs'/(group+'.json'));source_sha=sha(ROOT/'runtime_freeze.json')
 state={'epoch':0,'position':0,'updates':0,'loss_sum':0.,'history':[],'calibration':[],'train_seconds':0.,'calibration_seconds':0.,'best':None,'best_epoch':None}
 def checkpoint(path):
  payload=dict(state,group=group,model=head.trainable_state_dict(),optimizer=opt.state_dict(),config_sha256=lock_sha,runtime_freeze_sha256=source_sha,torch_cpu_rng=torch.get_rng_state(),torch_cuda_rng=torch.cuda.get_rng_state_all())
  temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path)
 current=out/'latest.pt'
 if current.exists():
  old=torch.load(current,map_location='cpu',weights_only=False);assert old['config_sha256']==lock_sha and old['runtime_freeze_sha256']==source_sha
  head.load_trainable_state_dict(old['model']);opt.load_state_dict(old['optimizer'])
  for k in state:state[k]=old[k]
 else:
  initial=head.trainable_state_dict();smoke=ds.sample(0,group,embedding,'cuda:0');losses=[]
  for i in range(2):
   torch.manual_seed(42000001+i);opt.zero_grad(set_to_none=True);loss=loss_for(head,smoke,config);loss.backward()
   assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
   opt.step();losses.append(float(loss.detach()))
  head.load_trainable_state_dict(initial);opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
  assert all(p.grad is None for p in draft.model.parameters())
  write(out/'GPU_training_smoke.json',{'status':'pass','losses':losses,'updates_discarded':2,'initial_restored':True,'Target_loaded':False,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
  checkpoint(current)
 orders=read(MATRIX/'epoch_orders.json');assert len(orders)==5
 while state['epoch']<5:
  ep=state['epoch'];head.train()
  while state['position']<156688:
   pos=state['position'];sample=ds.sample(orders[ep][pos],group,embedding,'cuda:0')
   torch.manual_seed(42*1000000+ep*156688+pos+1);t0=time.perf_counter();opt.zero_grad(set_to_none=True)
   loss=loss_for(head,sample,config);loss.backward()
   grads=[p.grad for p in head.parameters() if p.requires_grad and p.grad is not None]
   assert grads and torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads)
   opt.step();state['loss_sum']+=float(loss.detach());state['position']+=1;state['updates']+=1;state['train_seconds']+=time.perf_counter()-t0
   if state['updates']==1:write(out/'first_update.json',{'status':'pass','loss':float(loss.detach()),'window_index':orders[ep][pos],'initial_sha256':config['initial_sha256'],'config_sha256':lock_sha,'source_sha256':source_sha,'unix':time.time()})
   if state['position']%256==0:checkpoint(current);write(out/'status.json',dict(state,phase='train',pid=os.getpid(),unix=time.time()))
  head.eval();sums=[0.]*503;counts=[0]*503;t0=time.perf_counter()
  with torch.inference_mode():
   for i in range(19736):
    sample=cal.sample(i,group,embedding,'cuda:0');value=float(loss_for(head,sample,config,temperature=None));ri=sample[3]
    assert math.isfinite(value);sums[ri]+=value;counts[ri]+=1
  assert all(counts);metric=sum(s/n for s,n in zip(sums,counts))/503
  state['calibration_seconds']+=time.perf_counter()-t0
  state['history'].append({'epoch':ep+1,'updates':state['updates'],'train_window_CE':state['loss_sum']/156688})
  calrec={'epoch':ep+1,'calibration_request_equal_CE':metric,'request_values':[{'request_id':r['request_id'],'mean_CE':s/n,'windows':n} for r,s,n in zip(cal.records,sums,counts)]}
  write(out/f'calibration_epoch{ep+1}.json',calrec);state['calibration'].append({'epoch':ep+1,'value':metric,'receipt_sha256':sha(out/f'calibration_epoch{ep+1}.json')})
  if state['best'] is None or metric<state['best']:state['best']=metric;state['best_epoch']=ep+1
  state['epoch']+=1;state['position']=0;state['loss_sum']=0.;checkpoint(out/f'epoch{state["epoch"]}.pt');checkpoint(current)
 assert state['updates']==783440
 selected={'main':{'rule':'original_epoch5','path':str(out/'epoch5.pt'),'sha256':sha(out/'epoch5.pt')},'secondary':{'rule':'minimum_503_request_equal_CE','epoch':state['best_epoch'],'path':str(out/f'epoch{state["best_epoch"]}.pt'),'sha256':sha(out/f'epoch{state["best_epoch"]}.pt')}}
 write(out/'selected.json',selected);write(out/'completion.json',{'status':'complete','group':group,'updates':state['updates'],'train_seconds':state['train_seconds'],'calibration_seconds':state['calibration_seconds'],'config_sha256':lock_sha,'runtime_freeze_sha256':source_sha,'selected':selected,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'unix':time.time()})

def evaluate(group,out,smoke):
 import torch
 from runtime import load_head,online_state,online_next
 from feedback_codec.hf_backend import HFTargetBackend,HFDraftBackend,_FrozenHFBackend,assert_compatible_pair
 from feedback_codec.decoding import autoregressive_generate
 sys.path.insert(0,str(PARENT));from closed_loop_core import decode,digest
 class DraftOnly(_FrozenHFBackend):teacher_hidden=HFDraftBackend.teacher_hidden
 torch.set_num_threads(4);f=read(WAVE/'freeze.json');roster=read(WAVE/'roster503.json');assert sha(WAVE/'roster503.json')==f['roster_sha256']
 target=HFTargetBackend(str(pathlib.Path(f['models_root'])/'Qwen3-8B'),None,device='cuda:0',dtype=torch.bfloat16)
 draft=DraftOnly(str(pathlib.Path(f['models_root'])/'Qwen3-0.6B'),None,device='cuda:0',dtype=torch.bfloat16);assert_compatible_pair(target,draft)
 selected={'initial':{'path':None,'sha256':sha(MATRIX/'initial'/(group+'.safetensors'))}} if smoke else read(ROOT/'runs'/group/'train/selected.json')
 refs={};duplicates=[]
 for label,a in selected.items():
  if a['sha256'] in refs:duplicates.append({'selection':label,'equivalent_to':refs[a['sha256']]});continue
  refs[a['sha256']]=label
  bound,verified=(initial_binding(ROOT,MATRIX,group) if smoke else training_binding(ROOT,MATRIX,group,label))
  head=load_head(group,draft,verified_state=verified);del verified;head.eval()
  smoke_ids=read(WAVE/'smoke_ids.json');ordered=[r for rid in smoke_ids for r in roster if r['request_id']==rid]
  if not smoke:ordered+=[r for r in roster if r['request_id'] not in smoke_ids]
  sealed=[]
  for req in ordered:
   # Revalidate actual checkpoint, config and terminal before every reuse/new seal.
   checked,_=(initial_binding(ROOT,MATRIX,group,bound) if smoke else training_binding(ROOT,MATRIX,group,label,bound))
   dest=out/label/req['request_id'];receipt=dest/'receipt.json'
   if receipt.exists():
    r=read(receipt);require(r.get('checkpoint_binding')==bound,'sealed request checkpoint binding differs');assert r['checkpoint_sha256']==a['sha256'] and r['runtime_freeze_sha256']==sha(ROOT/'runtime_freeze.json') and sha(dest/'rounds.jsonl')==r['rounds_sha256']
    assert r['output_ids']==req['frozen_target_ids'] and r['eos']==req['eos'] and r['truncated']==req['truncated'];sealed.append(r);continue
   oracle=read(WAVE/'oracles'/(req['request_id']+'.json'));assert oracle['output_ids']==req['frozen_target_ids'] and oracle['prompt_sha256']==digest(req['prompt_ids']) and oracle['freeze_sha256']==sha(WAVE/'freeze.json')
   raw=read(req['raw_trace_path']);assert sha(req['raw_trace_path'])==req['raw_trace_sha256'] and raw['prompt_ids']==req['prompt_ids'] and raw['output_ids']==req['frozen_target_ids'] and raw['trajectory_policy']=='RAW'
   if req['request_id'] in smoke_ids:
    with torch.inference_mode():fresh=list(autoregressive_generate(target,req['prompt_ids'],128,151645))
    assert fresh==req['frozen_target_ids'],'fresh independent Target differs'
   rows=[];t0=time.perf_counter();tmp=out/'incomplete'/f'{label}-{req["request_id"]}-{time.time_ns()}';tmp.mkdir(parents=True)
   def capture(row,boundary):
    rows.append(row)
    with (tmp/'rounds.jsonl').open('a') as z:z.write(json.dumps(row,separators=(',',':'))+'\n')
   with torch.inference_mode():result=decode(req['prompt_ids'],draft_token=lambda p,s,i:online_next(head,p,s,i,draft),target_step=target.next_with_logit_summary if group=='TOP32_v2' else target.next,
    prepare=lambda row,b:online_state(group,head,row,b,draft),on_round=capture,oracle_ids=req['frozen_target_ids'],max_new=128,window=8,eos=151645)
   assert result['eos']==req['eos'] and result['truncated']==req['truncated'] and rows[0]['candidate_ids']==raw['rounds'][0]['candidate_ids']
   r={k:v for k,v in result.items() if k!='rounds'};r.update(checkpoint_binding=bound,status='pass',request_id=req['request_id'],family_id=req['family_id'],group=group,selection=label,checkpoint_sha256=a['sha256'],runtime_freeze_sha256=sha(ROOT/'runtime_freeze.json'),raw_trace_sha256=req['raw_trace_sha256'],oracle_receipt_sha256=sha(WAVE/'oracles'/(req['request_id']+'.json')),rounds_sha256=sha(tmp/'rounds.jsonl'),elapsed_seconds=time.perf_counter()-t0,ID_EOS_truncation_exact=True,unix=time.time())
   checked,_=(initial_binding(ROOT,MATRIX,group,bound) if smoke else training_binding(ROOT,MATRIX,group,label,bound))
   write(tmp/'receipt.json',r);dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();tmp.rename(dest);sealed.append(r)
   with (out/'manifest.jsonl').open('a') as z:z.write(json.dumps({'request_id':req['request_id'],'selection':label,'path':str(dest),'receipt_sha256':sha(dest/'receipt.json'),'unix':time.time()})+'\n');z.flush();os.fsync(z.fileno())
   write(out/'status.json',{'group':group,'selection':label,'sealed':len(sealed),'total':len(ordered),'pid':os.getpid(),'unix':time.time()})
   if len(sealed)==len(smoke_ids):
    write(out/(label+'_GPU_smoke.json'),{'status':'pass','requests':smoke_ids,'checkpoint_sha256':a['sha256'],
     'ID_EOS_truncation_exact':True,'first_round_RAW_exact':True,'fresh_independent_Target_checked':True,
     'receipt_sha256':{x:sha(out/label/x/'receipt.json') for x in smoke_ids},
     'Target_loaded':True,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
  write(out/(label+'_summary.json'),{'requests':len(sealed),'accepted_draft_tokens':sum(x['accepted_draft_tokens'] for x in sealed),'verification_rounds':sum(x['verification_rounds'] for x in sealed),'accepted_tokens_per_round':sum(x['accepted_draft_tokens'] for x in sealed)/sum(x['verification_rounds'] for x in sealed),'seconds':sum(x['elapsed_seconds'] for x in sealed)})
  del head;gc.collect();torch.cuda.empty_cache()
 write(out/'completion.json',{'status':'complete','group':group,'phase':'smoke' if smoke else 'dev503','selection_aliases':duplicates,'roster_sha256':f['roster_sha256'],'smoke_ID_EOS_truncation_exact':True,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'independent300_used':False,'unix':time.time()})

def main():
 p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--phase',choices=['smoke','train','eval'],required=True);a=p.parse_args()
 verify_freeze();assert a.group in read(ROOT/'runtime_freeze.json')['priority']
 out=ROOT/'runs'/a.group/a.phase;out.mkdir(parents=True,exist_ok=True)
 if (out/'completion.json').exists():
  require(read(out/'completion.json')['status']=='complete','noncomplete worker terminal')
  if a.phase=='eval':
   for label in read(ROOT/'runs'/a.group/'train/selected.json'):training_binding(ROOT,MATRIX,a.group,label)
  return
 own=open(out/'worker.lock','a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if a.phase=='train':assert read(ROOT/'runs'/a.group/'smoke/completion.json')['status']=='complete'
 if a.phase=='eval':assert read(ROOT/'runs'/a.group/'train/completion.json')['status']=='complete'
 write(out/'identity.json',dict(identity(),group=a.group,phase=a.phase))
 try:
  with lease(out):
   if a.phase=='train':train(a.group,out)
   else:evaluate(a.group,out,a.phase=='smoke')
 except Exception as e:first_error(out/'first_error.json',e);raise
if __name__=='__main__':main()
