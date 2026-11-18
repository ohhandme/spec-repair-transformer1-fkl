import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,hashlib,json,time
from common import *
import heads as H
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def main():
 (ROOT/'configs').mkdir(exist_ok=True)
 for group in H.GROUPS:
  old=V3/'remaining_cpu_v2/configs'/(group+'.json');c=read(old)
  require(sha(c['initial_path'])==c['initial_sha256'],'initial weight SHA changed')
  c.update(stage='actual_smoke_train_eval_entry',entry=str(ROOT/'worker20.py'),entry_sha256=sha(ROOT/'worker20.py'),parent_CPU_config_sha256=sha(old),max_new_tokens=128,
   checkpoint_owner='H-VQ64-weighted' if group in H.INTERVENTIONS else group,online_boundary='own realized valid prefix that predicts correction/bonus; local Draft at committed corrected prefix',
   training_sampler='deterministic hard ordinary C' if group.startswith('R_') else 'original temperature1 sampler',checkpoint_retention='two alternating resume slots plus immutable epoch checkpoints',CPU_fixture=False)
  immutable(ROOT/'configs'/(group+'.json'),c)
 path=ROOT/'online_donors.json'
 if not path.exists():
  rs=[r for r in lines(TRAIN/'index.jsonl') if r['split']=='calibration'];roster=read(WAVE/'roster503.json');require(len(rs)==len(roster)==503,'calibration donor partition differs')
  donors=[dict(key=w['observation_key'],request_id=r['request_id'],family_id=r['family_id'],split='calibration',round=w['round'],accepted=w['accepted'],rejected_is_none=w['rejected'] is None) for r in rs for w in r['windows']]
  require(len(donors)==19736 and len({x['key'] for x in donors})==19736,'donor legal-window lock differs')
  full={};mismatch={};counts={};started=time.monotonic()
  for r in roster:
   eligible=[d for d in donors if d['family_id']!=r['family_id'] and d['request_id']!=r['request_id']]
   # Exact original Task34 hash-ranking rule for full-H donors, new A100 pool.
   stem=['task34_dynamic_shuffle_v1',42,'dev',r['request_id']]
   ordered=sorted(eligible,key=lambda d:(hashlib.sha256(canonical([*stem,d['key']])).hexdigest(),d['key']))
   require(len(ordered)>=127,'R_SHUFFLED has insufficient distinct donors');full[r['request_id']]=[d['key'] for d in ordered[:127]]
   buckets=collections.defaultdict(list)
   for d in eligible:buckets[str(d['accepted'])+':'+str(d['rejected_is_none'])].append(d)
   mismatch[r['request_id']]={};counts[r['request_id']]={}
   for bucket,pool in buckets.items():
    stem=['task59_8B_mismatch_condition_v1',42,'dev',r['request_id'],bucket]
    order=sorted(pool,key=lambda d:(hashlib.sha256(canonical([*stem,d['key']])).hexdigest(),d['key']))
    mismatch[r['request_id']][bucket]=[d['key'] for d in order[:127]];counts[r['request_id']][bucket]=len(order)
  immutable(path,dict(version=1,R_SHUFFLED=full,MISMATCH=mismatch,eligible_8B_counts=counts,roster_sha256=sha(WAVE/'roster503.json'),partition_sha256=sha(MATRIX/'partition_lock.json'),
   donor_index_sha256=sha(TRAIN/'index.jsonl'),donor_metadata_sha256=hashlib.sha256(canonical(donors)).hexdigest(),
   R_contract='same calibration split, different request/family; 127 distinct hash-ranked full BF16 H donors per request; 8192 bytes',
   MISMATCH_contract='same calibration split and accepted-count/reject-vs-allaccept condition, different request/family; deterministic round-index selection from bucket; encode only 8 bytes with the bound weighted checkpoint',
   training_R_donors_sha256=sha(V3/'remaining_cpu_v2/train_donors.json'),calibration_R_donors_sha256=sha(V3/'remaining_cpu_v2/calibration_donors.json'),elapsed_seconds=time.monotonic()-started))
 write(ROOT/'prepare_CPU_receipt.json',dict(status='pass',conditions=20,training_conditions=17,evaluation_only=sorted(H.INTERVENTIONS),online_donors_sha256=sha(path),
  configs={g:sha(ROOT/'configs'/(g+'.json')) for g in H.GROUPS},independent300_used=False,GPU_used=False,unix=time.time()))
 print(json.dumps({'status':'pass','conditions':20,'online_donors_sha256':sha(path)}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:first_error(ROOT/'prepare_CPU_first_error.json',e);raise
