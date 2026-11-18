"""Deduplicate missing legal prefixes while retaining every supervised occurrence."""
import collections,hashlib,json,sys,time,traceback
from common import *
OUT=ROOT/'fkl';OUT.mkdir(exist_ok=True)
def digest(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def main():
 if (OUT/'dedup_receipt.json').exists():print('dedup_already_sealed');return
 start=time.monotonic();lock=read(MATRIX/'partition_lock.json');assert sha(TRAIN/'index.jsonl')==lock['source_index_sha256']
 wave=read(WAVE/'freeze.json');binding=wave['models_manifest_sha256']
 records=[r for r in lines(TRAIN/'index.jsonl') if r['split'] in ('train','calibration')]
 seen={};totals=collections.Counter();request_rows=[];refs={};first=None
 upath=OUT/'unique_prefixes.jsonl';mpath=OUT/'occurrence_map.jsonl'
 # Unsealed attempt files are preserved, never appended to a sealed manifest.
 utemp=OUT/f'unique_prefixes.attempt-{os.getpid()}.jsonl';mtemp=OUT/f'occurrence_map.attempt-{os.getpid()}.jsonl'
 with utemp.open('x') as uf,mtemp.open('x') as mf:
  for r in records:
   t=read(DATA/'shards'/r['request_id']/'trace.json');prompt=r['prompt_ids'];output=r['output_ids']
   assert t['trajectory_policy']=='RAW' and t['output_ids']==output and t['repair_enabled'] is False
   hpos={x['hidden']['absolute_position'] for x in t['rounds'] if x['hidden']['present']};source={}
   for rr in t['rounds']:
    for j,p in enumerate(rr['top32']['absolute_source_positions']):
     assert p not in source;source[p]=(rr['round_id'],j,rr['top32']['greedy_ids'][j])
   needed={len(prompt)+w['offset']+i-1 for w in r['windows'] for i in range(w['length'])}-hpos
   position_map={}
   for pos in sorted(needed):
    n=pos-len(prompt)+1;assert 0<=n<len(output)
    prefix=prompt+output[:n];prefix_sha=digest(prefix)
    key=hashlib.sha256((binding+':BF16:eager:canonical_full_prefix:'+prefix_sha).encode()).hexdigest()
    rr,j,label=source[pos];assert label==output[n]
    if key in seen:
     prev=seen[key];assert prev['prefix_length']==len(prefix) and prev['expected_id']==label
     totals['duplicate_prefixes_across_requests']+=1
     assert prev['split']==r['split'], 'cross-split identical prefix; inspect independence'
    else:
     item={'prefix_id':key,'prefix_sha256':prefix_sha,'prefix_length':len(prefix),'request_id':r['request_id'],
      'split':r['split'],'confirmed_output_tokens':n,'absolute_source_position':pos,'expected_id':label,
      'raw_round':rr,'raw_decision_row':j,'request_eos':t['eos'],'request_truncated':t['truncated'],
      'target_binding_sha256':binding,'precision':'BF16','backend':'eager_canonical_full_prefix'}
     seen[key]=item;uf.write(json.dumps(item,separators=(',',':'))+'\n');totals['unique_missing_prefixes']+=1
     totals['unique_'+r['split']]+=1
    position_map[pos]=key
   for wi,w in enumerate(r['windows']):
    for j,label in enumerate(w['labels']):
     pos=len(prompt)+w['offset']+j-1
     if pos in hpos:continue
     key=position_map[pos];ref={'request_id':r['request_id'],'split':r['split'],'window_index':wi,'observation_key':w['observation_key'],'label_position0':j,'expected_id':label,'prefix_id':key}
     mf.write(json.dumps(ref,separators=(',',':'))+'\n');totals['missing_occurrences']+=1;totals['occurrences_'+r['split']]+=1
     refs[key]=refs.get(key,0)+1
   request_rows.append({'request_id':r['request_id'],'split':r['split'],'family_id':r['family_id'],'prompt_ids':prompt,'output_ids':output,'trace_path':str(DATA/'shards'/r['request_id']/'trace.json'),'trace_sha256':sha(DATA/'shards'/r['request_id']/'trace.json')})
  for f in [uf,mf]:f.flush();os.fsync(f.fileno())
 assert totals['missing_occurrences']==922132 and sum(refs.values())==922132
 assert set(refs)==set(seen)
 for tmp,dest in [(utemp,upath),(mtemp,mpath)]:assert not dest.exists();tmp.rename(dest)
 with (OUT/'requests.jsonl').open('x') as f:
  for r in request_rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
 write(OUT/'dedup_receipt.json',{'status':'complete','counts':dict(totals),'source_missing_sha256':sha(MATRIX/'FKL_missing_legal_prefixes.jsonl'),
  'files':{p.name:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in [upath,mpath,OUT/'requests.jsonl']},
  'prefix_key':'Target binding + BF16 + eager canonical full-prefix + exact token-list SHA',
  'full_vocab_training_admitted':False,'required_gate':'canonical Target logits == GPU lm_head(saved final norm H) for all vocabulary entries; then validate every backfilled prefix',
  'elapsed_cpu_seconds':time.monotonic()-start,'unix':time.time()})
 print(json.dumps(read(OUT/'dedup_receipt.json')),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write(OUT/'first_error.json',{'error':repr(e),'traceback':traceback.format_exc(),'unix':time.time()});raise
