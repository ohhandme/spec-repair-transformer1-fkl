"""Freeze A100-only donor/quantizer fits, initializations and exercise 20 CPU entries."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,dataclasses,gc,hashlib,json,random,sys,time
from common import *
def main():
 import torch
 from safetensors import safe_open
 from safetensors.torch import save_file,load_file
 from runtime import FrozenRows
 import remaining20_a100 as ext
 torch.set_num_threads(1);require(not torch.cuda.is_available(),'CPU visibility violated')
 out=ROOT/'remaining_cpu_v2';out.mkdir(exist_ok=True);(out/'initial').mkdir(exist_ok=True);(out/'configs').mkdir(exist_ok=True)
 for name,e in read(ROOT/'originals20_manifest.json').items():require(sha(ROOT/'originals20'/name)==e['sha256'],'original source changed '+name)
 require(read(V2/'preprocessing/completion.json')['status']=='complete','PCA/SCORE fit absent')
 fit=read(V2/'preprocessing/completion.json')
 for n,e in fit['artifacts'].items():require(sha(V2/'preprocessing'/n)==e['sha256'],'PCA/SCORE fit artifact changed '+n)
 ds=FrozenRows('train');cal=FrozenRows('calibration');datasets={'train':ds,'calibration':cal};mapping={};unmatched={};started=time.monotonic()
 for split,data in datasets.items():
  rows=[(r['family_id'],r['request_id'],w['round'],ri,wi,w['accepted'],w['rejected'] is None) for ri,r in enumerate(data.records) for wi,w in enumerate(r['windows'])]
  buckets=collections.defaultdict(list)
  for i,row in enumerate(rows):buckets[(row[5],row[6])].append(i)
  # Original Task34 same-split family derangement and an extra matched-boundary
  # lock for the eight-byte mismatch intervention.
  def derange(indices):
   order=sorted(indices,key=lambda i:rows[i][:3]);counts=collections.Counter(rows[i][0] for i in order);maximum=max(counts.values());shift=maximum+random.Random(42).randrange(maximum)
   require(2*shift<=len(order),'insufficient different-family donors for frozen condition')
   pairs={i:order[(pos+shift)%len(order)] for pos,i in enumerate(order)}
   require(all(rows[i][0]!=rows[j][0] for i,j in pairs.items()) and len(set(pairs.values()))==len(order),'donor permutation invalid')
   return pairs
  ordinary=derange(list(range(len(rows))));matched={}
  for indices in buckets.values():matched.update(derange(indices))
  mapping[split]={'R_SHUFFLED':[ordinary[i] for i in range(len(rows))],'H-VQ64-weighted-MISMATCH':[matched[i] for i in range(len(rows))]}
  immutable(out/(split+'_donors.json'),dict(split=split,index_sha256=sha(TRAIN/'index.jsonl'),rows=[{'request_id':x[1],'family_id':x[0],'round':x[2],'record_index':x[3],'window_index':x[4]} for x in rows],mapping=mapping[split],seed=42,fit_split=split))
 # RANK64 margin retains the original train-p99 target, refitted on A100 train.
 if (ROOT/'remaining_cpu/rank_fit.json').exists():immutable(out/'rank_fit.json',read(ROOT/'remaining_cpu/rank_fit.json'))
 if not (out/'rank_fit.json').exists():
  margins=[]
  for ri,r in enumerate(ds.records):
   t=ds.trace(ri);tensors=ds.tensors(ri)
   for w in r['windows']:
    k=ext.topk(r,w,t,tensors);by=dict(zip(k['token_ids'],k['logits']));alt=max(z for i,z in by.items() if i!=w['correction']);margins.append(by[w['correction']]-alt)
  require(len(margins)==156688,'rank train fit count differs');clip=float(torch.tensor(margins,dtype=torch.float32).quantile(.99));require(clip>0,'degenerate RANK margin')
  immutable(out/'rank_fit.json',dict(train_p99_margin=clip,fit_train_windows=len(margins),quantile=.99,index_sha256=sha(TRAIN/'index.jsonl'),train_fit_rows_sha256=sha(MATRIX/'train_fit_rows.jsonl'),dev_fit=False))
 mr=pathlib.Path(read(WAVE/'freeze.json')['models_root'])/'Qwen3-0.6B'
 with safe_open(mr/'model.safetensors',framework='pt',device='cpu') as sf:weight=sf.get_tensor('model.embed_tokens.weight')
 embedding=torch.nn.Embedding.from_pretrained(weight,freeze=True);output=torch.nn.Linear(1024,151936,bias=False,device='meta');output.weight=embedding.weight
 ri,wi=ds.windows[0];r=ds.records[ri];w=r['windows'][wi];trace=ds.trace(ri);cache=ds.cache(ri);tensors=ds.tensors(ri);history=cache['draft_sequence'][w['offset']:w['offset']+w['length']].unsqueeze(0);labels=torch.tensor(w['labels']);results=[]
 weighted_init=None
 for group in ext.GROUPS:
  head=ext.build(group,output,embedding);evaluation=group in ext.INTERVENTIONS
  if evaluation:
   require(weighted_init is not None,'intervention base initialization absent');head.load_trainable_state_dict(weighted_init)
  initial=head.trainable_state_dict();init_path=out/'initial'/(( 'H-VQ64-weighted' if evaluation else group)+'.safetensors')
  if not init_path.exists():save_file(initial,str(init_path))
  else:
   old=load_file(init_path);require(set(old)==set(initial) and all(torch.equal(old[n],initial[n]) for n in old),'initial changed')
  if group=='H-VQ64-weighted':weighted_init=initial
  head.eval() if evaluation else head.train();donor=None;pairsha=None
  if group in ('R_SHUFFLED','H-VQ64-weighted-MISMATCH'):
   idx=mapping['train'][group][0];dri,dwi=ds.windows[idx];dr=ds.records[dri];donor=(dr,dr['windows'][dwi],ds.trace(dri),ds.cache(dri));pairsha=sha(out/'train_donors.json')
  state,extra=ext.prepare(group,head,r,w,trace,tensors,cache,embedding,donor,pairsha)
  torch.manual_seed(42000001);logits=ext.forward(group,head,history,state,extra,temperature=None if evaluation else 1.)
  teacher=None
  if group in ext.FKL:
   # Full-dimensional synthetic teacher tests loss algebra only, NOT FKL admission.
   teacher=logits.detach().float().roll(1,-1)
  value=ext.loss(group,logits,labels,teacher);require(torch.isfinite(value),'CPU loss nonfinite')
  if not evaluation:
   value.backward();require(all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None),'CPU gradient nonfinite')
   require(output.weight.grad is None,'frozen Draft output received gradient')
  if group in ext.FKL:
   lp=teacher.float().log_softmax(-1);lq=logits.float().log_softmax(-1);per=(lp.exp()*(lp-lq)).sum(-1).flatten()
   if group.endswith('CE25FKL75'):per=.25*torch.nn.functional.cross_entropy(logits.float().flatten(0,1),labels,reduction='none')+.75*per
   weights=torch.ones(len(labels)) if group.endswith('EQUAL') else torch.arange(8,8-len(labels),-1,dtype=torch.float32)
   require(torch.allclose(value,(per*weights).sum()/weights.sum(),atol=1e-6,rtol=1e-6),'FKL/mixed reference differs')
   try:ext.loss(group,logits,labels,teacher[...,:32])
   except SharedIntegrityError:pass
   else:raise SharedIntegrityError('Top32 accepted as full FKL')
  cfg=dict(group=group,stage='CPU_entry_prepared',entry=str(ROOT/'remaining20_a100.py'),entry_sha256=sha(ROOT/'remaining20_a100.py'),
   initial_path=str(init_path),initial_sha256=sha(init_path),parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),parameter_bytes=sum(p.numel()*p.element_size() for p in head.parameters() if p.requires_grad),
   training_windows=156688,calibration_windows=19736,index_sha256=sha(TRAIN/'index.jsonl'),partition_sha256=sha(MATRIX/'partition_lock.json'),
   loss='CE25FKL75' if group.endswith('CE25FKL75') else 'FKL' if group in ext.FKL else 'CE',position_weights=[1]*8 if group in ('F-H','H-VQ64-TASK','C-HARD-ST','RANK64-MARGIN','MLP-C-FKL-EQUAL') else list(range(8,0,-1)),position_normalization='valid positions only',temperature=1.,seed=42,batch_size=1,epochs=5,updates=783440,optimizer={'kind':'AdamW','lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':.01},
   evaluation_only=evaluation,checkpoint_owner='H-VQ64-weighted' if evaluation else group,main_epoch=5,secondary='minimum calibration request-equal CE',
   requires_GPU_smoke=True,full_vocab_training_admitted=False if group in ext.FKL else None,independent300_used=False)
  immutable(out/'configs'/(group+'.json'),cfg)
  results.append(dict(group=group,status='pass_CPU',loss=float(value.detach()),gradient_checked=not evaluation,parameters=cfg['parameters'],parameter_bytes=cfg['parameter_bytes'],initial_sha256=cfg['initial_sha256'],config_sha256=sha(out/'configs'/(group+'.json')),synthetic_teacher_loss_test_only=group in ext.FKL,evaluation_only=evaluation))
  print(json.dumps(results[-1]),flush=True);del head,initial,logits,value;gc.collect()
 require(len(results)==20,'remaining condition count differs')
 write(out/'completion.json',dict(status='pass_CPU',groups=results,elapsed_seconds=time.monotonic()-started,donor_sha256={s:sha(out/(s+'_donors.json')) for s in mapping},rank_fit_sha256=sha(out/'rank_fit.json'),PCA_SCORE_reused_completion_sha256=sha(V2/'preprocessing/completion.json'),
  full_vocab_training_admitted=False,remaining_GPU_gate='real independent Target ID/EOS/truncation smoke, shared resource admission, canonical full-vocab coverage for FKL; no GPU jobs dispatched by this entry',independent300_used=False,GPU_used=False,source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},prior_attempt_first_error_sha256=sha(ROOT/'remaining_cpu/first_error.json'),resolved_prior_error='original rank consumer absolute path rebound to exact SHA-verified local wire source',unix=time.time()))
 print(json.dumps({'status':'pass_CPU','groups':len(results),'completion_sha256':sha(out/'completion.json')}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:first_error(ROOT/'remaining_cpu_v2/first_error.json',e);raise
