"""Execute the audited numerical main bodies with only A100 data/count adapters."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gc,hashlib,importlib.util,json,math,sys,time,traceback
from types import SimpleNamespace
from common import *
OUT=ROOT/'preprocessing';OUT.mkdir(exist_ok=True)
def main():
 if (OUT/'completion.json').exists():print('already_complete_no_refit');return
 import torch
 from safetensors import safe_open
 torch.set_num_threads(4);assert not torch.cuda.is_available()
 source=ROOT/'task23_originals';manifest=read(source/'audit_package_manifest_v1.json')
 for n,e in manifest['files'].items():assert sha(source/n)==e['sha256'] and (source/n).stat().st_size==e['bytes']
 audit=read(ROOT/'original_source_audit.json');assert audit['status']=='all_56_verified'
 lock=read(MATRIX/'partition_lock.json');assert sha(TRAIN/'index.jsonl')==lock['source_index_sha256']
 assert sha(MATRIX/'train_fit_rows.jsonl')==lock['train_fit_rows_sha256']
 records=lines(TRAIN/'index.jsonl');selected={s:[r for r in records if r['split']==s] for s in ['train','calibration']}
 keys={s:[w['observation_key'] for r in rs for w in r['windows']] for s,rs in selected.items()}
 assert len(keys['train'])==156688 and len(keys['calibration'])==19736
 assert keys['train']==[x['observation_key'] for x in lines(MATRIX/'train_fit_rows.jsonl')]
 for name,s in [('train','train'),('val','calibration')]:
  immutable(OUT/(name+'_acceptance.json'),{'requests':len(selected[s]),'windows':len(keys[s]),'observation_keys_sha256':hashlib.sha256(json.dumps(keys[s],separators=(',',':')).encode()).hexdigest(),'index_sha256':lock['source_index_sha256']})
 def module(name,adapt_counts=False):
  src=(source/'new_source'/name).read_text()
  if adapt_counts:
   # Only dataset size literals change. The PCA solver, sign, quantiles and
   # scalar quantizer statements remain byte-for-byte as in the audited source.
   for old,new in [('7982','156688'),('1598','19736'),('1000','3986'),('200','503')]:src=src.replace(old,new)
  adapted=OUT/('a100_'+name)
  if adapted.exists():assert adapted.read_text()==src
  else:adapted.write_text(src)
  spec=importlib.util.spec_from_file_location('a100_'+name[:-3],adapted);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);m.ROOT=OUT
  return m
 def load_split(s):
  split='train' if s=='train' else 'calibration';windows=[]
  for r in selected[split]:
   with safe_open(TRAIN/'cache'/(r['request_id']+'.safetensors'),framework='pt',device='cpu') as f:target=f.get_tensor('target')
   assert target.shape==(len(r['windows']),4096)
   for i,w in enumerate(r['windows']):windows.append(SimpleNamespace(key=w['observation_key'],observation=SimpleNamespace(target_hidden=target[i])))
  return None,selected[split],windows
 pca=module('preprocess_pca_v1.py',True)
 pca.load_anchor=lambda:SimpleNamespace(load_split=load_split,previous=SimpleNamespace(causal_window_id=lambda w:w.key))
 if not (OUT/'pca_preprocess_manifest_v1.json').exists():pca.main()
 gc.collect()
 def score_selected(s):
  split='train' if s=='train' else 'calibration';rows={};counts=collections.Counter();h=hashlib.sha256()
  for r in selected[split]:
   trace=read(DATA/'shards'/r['request_id']/'trace.json')
   with safe_open(DATA/'shards'/r['request_id']/'features.safetensors',framework='pt',device='cpu') as f:
    for w in r['windows']:
     rr=trace['rounds'][w['round']];key=rr['tensor_prefix'];j=rr['top32']['used_rows']-1;b=w['correction']
     assert rr['observation_key']==w['observation_key'] and rr['top32']['greedy_ids'][j]==b
     assert rr['top32']['absolute_source_positions'][j]==rr['prefix_after_commit_length']-2
     ids=f.get_tensor(key+'.top32_ids')[j].tolist();logits=f.get_tensor(key+'.top32_logits')[j].float().tolist();lse=float(f.get_tensor(key+'.logsumexp')[j])
     assert len(ids)==len(logits)==32 and len(set(ids))==32 and b in ids
     pair=sorted(zip(ids,logits),key=lambda z:(-z[1],z[0]));alt=[z for z in pair if z[0]!=b][:2];byid=dict(zip(ids,logits))
     vec=[math.exp(byid[b]-lse),math.exp(alt[0][1]-lse),math.exp(alt[1][1]-lse),sum(math.exp(z-lse) for z in logits)]
     assert all(math.isfinite(z) and z>=0 for z in vec) and vec[3]<=1.05,(r['request_id'],w['round'],vec)
     rows[w['observation_key']]=vec;counts[r['topic']]+=1
     h.update(json.dumps((r['request_id'],w['round'],ids,logits,lse),separators=(',',':')).encode())
  assert len(rows)==len(keys[split]);return rows,dict(counts),h.hexdigest()
 score=module('preprocess_score_v2.py');score.selected=score_selected
 if not (OUT/'score_preprocess_v1.json').exists():score.main()
 p=torch.load(OUT/'pca_preprocess_v1.pt',map_location='cpu',weights_only=False)
 assert p['fit_train_windows']==156688 and p['quantiles']==[.001,.999] and p['basis'].shape==(4096,8)
 assert all(p['basis'][int(p['basis'][:,j].abs().argmax()),j]>=0 for j in range(8))
 fit=read(OUT/'score_preprocess_v1.json');assert fit['train_count']==156688 and fit['val_count']==19736 and all(x>0 for x in fit['scale'])
 cfg={'source_audit_manifest_sha256':sha(source/'audit_package_manifest_v1.json'),
  'fit_rows_sha256':sha(MATRIX/'train_fit_rows.jsonl'),'fit_split':'train','fit_windows':156688,'dev_apply_only_windows':19736,
  'PCA':{'q':16,'niter':4,'seed':42,'dimensions':8,'direction':'largest absolute loading positive','quantiles':[.001,.999],'quantizer':'floor(clamp((z-low)/(high-low),0,1)*255+0.5)','degenerate':'width < 1e-8 => byte128'},
  'SCORE':{'features':['p_b','p_alt1','p_alt2','top32_mass'],'dtype':'float64','scale':'std(unbiased=False)','zero_scale':'assert >0'},
  'codebook_training':{'objective':'8to1_valid_position_normalized_CE_only','PCA_and_LOCAL_codebook_shape':[8,256,16],'SCORE':'original ConditionedVQ64 straight-through task CE; no added reconstruction loss','optimizer':{'kind':'AdamW','lr':.0001,'betas':[.9,.999],'eps':1e-8,'weight_decay':.01},'epochs':5,'seed':42,'old_fitted_weights_reused':False},
  'CPU_only':True,'original_source_sha256':{n:sha(source/'new_source'/n) for n in ['preprocess_pca_v1.py','preprocess_score_v2.py','run_group_v3.py','task23_heads_v2.py']}}
 immutable(OUT/'fit_contract.json',cfg)
 write(OUT/'completion.json',{'status':'complete','contract_sha256':sha(OUT/'fit_contract.json'),
  'artifacts':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.is_file() and p.name not in ['completion.json']},'completed_unix':time.time()})
 print(json.dumps({'status':'complete','PCA':read(OUT/'pca_preprocess_manifest_v1.json'),'SCORE':{k:v for k,v in fit.items() if k not in ['rows']},'completion_sha256':sha(OUT/'completion.json')}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write(OUT/'first_error.json',{'error':repr(e),'traceback':traceback.format_exc(),'unix':time.time()});raise
