"""Verify all frozen inputs, seal executable queue, launch at most one waiting successor."""
import subprocess,sys,time,traceback
from common import *
def main():
 from supervisor import live_workers,stages
 cpu=read(ROOT/'runtime_CPU_receipt.json');assert cpu['status']=='pass' and len(cpu['groups'])==10
 assert read(ROOT/'preprocessing/completion.json')['status']=='complete'
 assert read(ROOT/'fkl/dedup_receipt.json')['counts']['missing_occurrences']==922132
 priority=read(MATRIX/'queue_registration.json')['priority_order'];files={}
 def bind(p,expected=None):
  p=pathlib.Path(p);h=sha(p)
  if expected is not None:assert h==expected, 'binding mismatch: '+str(p)
  files[str(p)]=h
 for n,h in cpu['source_sha256'].items():bind(ROOT/n,h)
 bind(ROOT/'seal_and_start.py')
 wave=read(WAVE/'freeze.json');bind(WAVE/'freeze.json')
 bind(WAVE/'roster503.json',wave['roster_sha256']);bind(WAVE/'smoke_ids.json')
 bind(pathlib.Path(wave['models_manifest_path']),wave['models_manifest_sha256'])
 modelmanifest=read(wave['models_manifest_path'])
 for model,entries in modelmanifest.items():
  for name,e in entries.items():bind(pathlib.Path(wave['models_root'])/model/name,e['sha256'])
 for name,h in wave['source_sha256'].items():bind(pathlib.Path(wave['source_root'])/name,h)
 for name,h in wave['reference_sources'].items():bind(name,h)
 for n in ['matrix_heads.py','repair_reference.py','arch_head.py','program.md','partition_lock.json','train_fit_rows.jsonl','epoch_orders.json','CPU_preflight.json']:bind(MATRIX/n)
 for g in priority:
  bind(MATRIX/'configs'/(g+'.json'));bind(MATRIX/'initial'/(g+'.safetensors'),read(MATRIX/'configs'/(g+'.json'))['initial_sha256'])
 partition=read(MATRIX/'partition_lock.json');bind(TRAIN/'index.jsonl',partition['source_index_sha256'])
 cachepath=PARENT/'first4-1a34c7c8/cache_manifest.json';bind(cachepath);cache=read(cachepath)
 selected=[r for r in lines(TRAIN/'index.jsonl') if r['split'] in ('train','calibration')]
 for r in selected:
  e=cache[r['request_id']];bind(e['path'],e['sha256']);root=DATA/'shards'/r['request_id'];receipt=read(root/'receipt.json');bind(root/'receipt.json')
  assert receipt['trajectory_policy']=='RAW' and receipt['repair_enabled'] is False
  for n in ['trace.json','features.safetensors']:bind(root/n,receipt['files'][n])
 for n in ['runtime_CPU_receipt.json','original_source_audit.json','preprocessing/completion.json','preprocessing/fit_contract.json','preprocessing/pca_preprocess_v1.pt','preprocessing/score_preprocess_v1.json','fkl/dedup_receipt.json']:bind(ROOT/n)
 for e in read(ROOT/'fkl/dedup_receipt.json')['files'].values():bind(e['path'],e['sha256'])
 freeze={'program_sha256':sha(MATRIX/'program.md'),'priority':priority,'files':files,'execution':'one sequential child, one GPU lease','all_models_and_selected_train_dev_caches_verified':True,'independent300_used':False}
 immutable(ROOT/'runtime_freeze.json',freeze)
 immutable(ROOT/'persistent_queue.json',{'stages':stages(priority),'required_prior':str(WAVE),'no_runtime_cap':True,'independent300_used':False})
 argv=[sys.executable,'-B','-u',str(ROOT/'supervisor.py')]
 live=live_workers(argv);assert len(live)<=1
 if live:owner=live[0];launched=False
 else:
  prior=ROOT/'supervisor_identity.json'
  if prior.exists():assert not alive(read(prior)), 'live successor has unexpected argv'
  assert not (ROOT/'supervisor_first_error.json').exists(), 'preserved successor failure requires diagnosis'
  with (ROOT/'supervisor.log').open('ab') as log:
   child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  owner=identity(child.pid);assert owner;launched=True
 write(ROOT/'launch_receipt.json',{'status':'successor_started' if launched else 'existing_successor_adopted','owner':owner,'argv':argv,'runtime_freeze_sha256':sha(ROOT/'runtime_freeze.json'),'persistent_queue_sha256':sha(ROOT/'persistent_queue.json'),'prior_ready_at_launch':prior_ready(),'existing_GPU_jobs_untouched':True,'unix':time.time()})
 print(json.dumps(read(ROOT/'launch_receipt.json')),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write(ROOT/'seal_first_error.json',{'error':repr(e),'traceback':traceback.format_exc(),'unix':time.time()});raise
