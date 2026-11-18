"""Seal independent revision; replace only the verified idle waiting predecessor."""
import signal,subprocess,sys,time
from common import *
from supervisor_final import live_workers,stages
def main():
 gate=open(MATRIX/'successor_adoption.lock','a+');fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
 require(read(ROOT/'audit_final_CPU_receipt.json')['status']=='pass','final CPU regression missing')
 require(read(ROOT/'remaining_cpu_v2/completion.json')['status']=='pass_CPU','20 CPU entries not passed')
 require(read(ROOT/'extra_CPU_receipt.json')['status']=='pass','entry integration regression missing')
 require(read(ROOT/'labels/completion.json')['counts']['label_occurrences']==1370920,'label mapping incomplete')
 prior=read(V2/'runtime_freeze.json');files=dict(prior['files'])
 def bind(p):files[str(p)]=sha(p)
 for p in ROOT.glob('*.py'):bind(p)
 for p in (ROOT/'originals20').glob('*.py'):bind(p)
 for n in ['originals20_manifest.json','audit_final_CPU_receipt.json','extra_CPU_receipt.json','remaining_cpu_v2/completion.json','remaining_cpu_v2/rank_fit.json','remaining_cpu_v2/train_donors.json','remaining_cpu_v2/calibration_donors.json','remaining_cpu/rank_fit.json','remaining_cpu/first_error.json','labels/completion.json']:bind(ROOT/n)
 for p in (ROOT/'remaining_cpu_v2/configs').glob('*.json'):bind(p)
 for p in (ROOT/'remaining_cpu_v2/initial').glob('*.safetensors'):bind(p)
 for e in read(ROOT/'labels/completion.json')['files'].values():require(sha(e['path'])==e['sha256'],'all-label artifact changed');bind(e['path'])
 for p,h in files.items():require(sha(p)==h,'shared source/model/data hash differs: '+p)
 freeze=dict(program_sha256=sha(MATRIX/'program.md'),revision=3,priority=prior['priority'],files=files,prior_freeze_sha256=sha(V2/'runtime_freeze.json'),independent300_used=False,
  checkpoint_gate='actual bytes SHA + configuration + training terminal before every initial/resume/new request seal',failure_policy='only typed FKL-specific failure after dead child and released receipt and free kernel lease may continue independent CE')
 immutable(ROOT/'runtime_freeze.json',freeze)
 prepared=[dict(group=x['group'],status='CPU_prepared_GPU_not_admitted',config_path=str(ROOT/'remaining_cpu_v2/configs'/(x['group']+'.json')),config_sha256=x['config_sha256']) for x in read(ROOT/'remaining_cpu_v2/completion.json')['groups']]
 queue=dict(stages=stages(prior['priority']),prepared_remaining20=prepared,required_prior=str(WAVE),no_runtime_cap=True,independent300_used=False)
 immutable(ROOT/'persistent_queue.json',queue)
 # Never replace a controller that has started work or taken the common lock.
 active=read(MATRIX/'active_successor.json');require(pathlib.Path(active['root'])==V2,'active successor changed; inspect instead of duplicate adoption')
 old=read(V2/'supervisor_identity.json');require(old['pid']==active['pid'] and old['start_tick']==active['start_tick'] and alive(old),'predecessor identity changed')
 status=read(V2/'supervisor_status.json');require(status['phase']=='waiting_original_queue' and status.get('GPU_dispatches')==0,'predecessor no longer idle waiting')
 require(not prior_ready(),'original queue already complete; refuse unsafe waiting-only handoff')
 require(not (V2/'queue_state.json').exists(),'predecessor has an execution state; do not replace')
 matching=live_workers([sys.executable,'-B','-u',str(V2/'supervisor.py')]);require(len(matching)==1 and matching[0]['pid']==old['pid'],'predecessor process argv differs')
 descendants=[]
 for p in pathlib.Path('/proc').glob('[0-9]*'):
  try:args=p.joinpath('cmdline').read_bytes().split(b'\0')
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
  if any(x.decode(errors='replace') in (str(V2/'priority_worker.py'),str(V2/'fkl_gpu.py')) for x in args):descendants.append(int(p.name))
 require(not descendants,'predecessor child exists; cannot replace')
 immutable(ROOT/'adoption_precheck.json',dict(old_identity=old,old_status=status,original_controller=read(WAVE/'controller_identity.json'),CPU_sha256=sha(ROOT/'audit_final_CPU_receipt.json'),queue_sha256=sha(ROOT/'persistent_queue.json'),unix=time.time()))
 os.kill(old['pid'],signal.SIGTERM)
 until=time.monotonic()+20
 while alive(old) and time.monotonic()<until:time.sleep(.1)
 require(not alive(old),'waiting predecessor did not exit; no new successor launched')
 with open(V2/'supervisor.lock','a+') as old_lock:fcntl.flock(old_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 argv=[sys.executable,'-B','-u',str(ROOT/'supervisor_final.py')];require(not live_workers(argv),'v3 already alive; avoid duplicate')
 with (ROOT/'supervisor.log').open('ab') as log:child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 owner=identity(child.pid);require(owner is not None,'successor identity absent')
 reg=PARENT/'followup_registry'/('matrix-'+sha(MATRIX/'program.md')+'.json')
 immutable(ROOT/'prior_registration.json',read(reg));oldreg=read(reg)
 nextreg=dict(oldreg,execution_revision=3,execution_root=str(ROOT),supervisor_identity=owner,prior_registration_sha256=sha(ROOT/'prior_registration.json'),runtime_freeze_sha256=sha(ROOT/'runtime_freeze.json'),persistent_queue_sha256=sha(ROOT/'persistent_queue.json'),updated_unix=time.time())
 write(reg,nextreg);write(MATRIX/'active_successor.json',dict(root=str(ROOT),**owner,revision=3,entry=argv[-1],prior_root=str(V2),unix=time.time()))
 write(ROOT/'launch_receipt.json',dict(status='single_successor_started_waiting_original',owner=owner,argv=argv,replaced_waiting_pid=old['pid'],replaced_pid_exited=True,old_workers_untouched=True,
  runtime_freeze_sha256=sha(ROOT/'runtime_freeze.json'),persistent_queue_sha256=sha(ROOT/'persistent_queue.json'),registration_sha256=sha(reg),GPU_dispatches=0,unix=time.time()))
 print(json.dumps(read(ROOT/'launch_receipt.json')),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:first_error(ROOT/'adoption_first_error.json',e);raise
