"""Freeze reviewed v5, then retire only paused/rechecked zero-dispatch v3/v4 waiters."""
import subprocess,sys,time
from common import *
from handoff_guard import snapshot,retire_waiters,process
from queue20 import stages,live_workers
import heads as H
PROGRAM='fc65585bce083e6c95a3237c081abe91484273639db4d0527130389dce5ff71b'
SPECS=[dict(root=str(V3),entry='supervisor_final.py',phase='waiting_original_queue'),dict(root=str(V4),entry='queue20.py',phase='waiting_v3_normal_terminal')]

def lock_owners(path):
 st=pathlib.Path(path).stat();owners=[]
 for line in pathlib.Path('/proc/locks').read_text().splitlines():
  fields=line.split()
  for i,x in enumerate(fields):
   if x.count(':')!=2:continue
   try:a,b,c=x.split(':');match=(int(a,16),int(b,16),int(c))==(os.major(st.st_dev),os.minor(st.st_dev),st.st_ino)
   except ValueError:continue
   if match:owners.append(int(fields[i-1]))
 return owners

def original_guard():
 owner=read(WAVE/'controller_identity.json');require(alive(owner),'original controller not alive')
 require(not original_ready(),'original queue advanced to complete; re-audit dispatch before handoff')
 require(lock_owners(CONTROLLER_LOCK)==[owner['pid']],'original shared controller lock is not exclusively held by original owner')
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
 priorpids={read(pathlib.Path(x['root'])/'supervisor_identity.json')['pid'] for x in SPECS}
 require(not priorpids.intersection(int(x.strip()) for x in active if x.strip()),'prior successor owns GPU')
 return dict(controller=process(owner['pid']),controller_lock_owners=lock_owners(CONTROLLER_LOCK),GPU_lock_owners=lock_owners(GPU_LOCK),compute_PIDs=active,read_only=True)

def main():
 lock=open(MATRIX/'strict_FKL_handoff.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 require(sha(MATRIX/'program.md')==PROGRAM,'matrix program changed')
 if not (ROOT/'handoff_receipt.json').exists():
  write(ROOT/'pre_handoff_readonly.json',dict(prior=[snapshot(s) for s in SPECS],original=original_guard(),unix=time.time()))
 for name,status in [('CPU_reuse_receipt.json','pass'),('FKL_resume_CPU_receipt.json','pass'),('FKL_shard_recovery_CPU_receipt.json','pass'),('queue_CPU_receipt.json','pass'),('handoff_CPU_receipt.json','pass'),('audit_final_CPU_receipt.json','pass')]:require(read(ROOT/name)['status']==status,'CPU regression missing: '+name)
 require(read(ROOT/'FKL_resume_CPU_receipt.json')['source_sha256']==sha(ROOT/'fkl_integrity.py'),'integrity CPU source differs')
 for n,h in read(ROOT/'FKL_shard_recovery_CPU_receipt.json')['source_sha256'].items():require(sha(ROOT/n)==h,'shard CPU source differs: '+n)
 require(read(ROOT/'queue_CPU_receipt.json')['source_sha256']==sha(ROOT/'queue20.py'),'queue CPU source differs')
 require(read(ROOT/'handoff_CPU_receipt.json')['source_sha256']==sha(ROOT/'handoff_guard.py'),'handoff CPU source differs')
 for n,h in read(ROOT/'CPU_reuse_receipt.json')['unchanged_engine_sources'].items():require(sha(ROOT/n)==h,'reused engine source changed')
 require(sha(V3/'runtime_freeze.json')=='4227da7ed2a34d89d809d182b75801c6f02c6e6b06f627e4e09498b733cc21e4','v3 freeze changed')
 require(sha(V4/'runtime_freeze.json')=='9fc8d7f1b72bf07c067fd4604aa823bd15f61e7f1197cc84463380a1e36ace96','v4 freeze changed')
 files=dict(read(V4/'runtime_freeze.json')['files'])
 for p in ROOT.glob('*.py'):files[str(p)]=sha(p)
 for p in (ROOT/'configs').glob('*.json'):files[str(p)]=sha(p)
 for n in ['CPU_E2E_receipt.json','CPU_reuse_receipt.json','FKL_resume_CPU_receipt.json','FKL_shard_recovery_CPU_receipt.json','queue_CPU_receipt.json','handoff_CPU_receipt.json','audit_final_CPU_receipt.json','label_coverage_CPU_receipt.json','online_donors.json']:files[str(ROOT/n)]=sha(ROOT/n)
 for r in [V3,V4]:
  for n in ['runtime_freeze.json','persistent_queue.json']:files[str(r/n)]=sha(r/n)
 for p,h in files.items():require(sha(p)==h,'frozen model/data/source changed: '+p)
 immutable(ROOT/'runtime_freeze.json',dict(revision=5,program_sha256=PROGRAM,priority=read(V3/'runtime_freeze.json')['priority'],remaining20=H.GROUPS,files=files,independent300_used=False))
 jobs=stages();require(len(jobs)==89 and jobs[0]['name']=='fkl_gate' and jobs[73]['name']=='fkl_backfill','strict queue order differs')
 immutable(ROOT/'persistent_queue.json',dict(revision=5,stages=jobs,stage_count=89,strict_FKL_first_position1=1,full_backfill_position1=74,nonFKL_positions1=[2,73],FKL_group_positions1=[75,89],original_six_ACK_must_complete=True,no_runtime_cap=True,independent300_used=False))
 # Snapshot old registration bytes; mutation below is only the reviewed handoff pointer.
 registry=PARENT/'followup_registry'/('matrix-'+PROGRAM+'.json')
 for path in [MATRIX/'active_successor.json',MATRIX/'remaining20_successor.json',registry]:
  if path.exists():
   target=ROOT/'legacy_registrations'/(path.name+'.'+sha(path)+'.json')
   target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():target.write_bytes(path.read_bytes())
   require(sha(target)==sha(path),'prior registration archive differs')
 if not (ROOT/'handoff_receipt.json').exists():
  receipt=retire_waiters(SPECS,original_guard)
  receipt.update(runtime_freeze_sha256=sha(ROOT/'runtime_freeze.json'),queue_sha256=sha(ROOT/'persistent_queue.json'),original_after=original_guard())
  immutable(ROOT/'handoff_receipt.json',receipt)
 else:
  old=read(ROOT/'handoff_receipt.json');require(old['status']=='safe_zero_dispatch_handoff','prior handoff failed')
  require(old['runtime_freeze_sha256']==sha(ROOT/'runtime_freeze.json') and old['queue_sha256']==sha(ROOT/'persistent_queue.json'),'handoff freeze changed')
  require(all(not alive(x) for x in old['retired']),'retired prior owner is live')
 argv=[sys.executable,'-B','-u',str(ROOT/'queue20.py')];existing=live_workers(argv);require(len(existing)<=1,'duplicate v5 successor')
 if existing:owner=existing[0];launched=False
 else:
  require(not (ROOT/'supervisor_first_error.json').exists(),'preserved supervisor error requires diagnosis')
  with (ROOT/'supervisor.log').open('ab') as log:child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
  owner=identity(child.pid);require(owner is not None,'v5 process identity absent');launched=True
 declaration=dict(revision=5,scope='priority10_plus_remaining20',root=str(ROOT),entry=str(ROOT/'queue20.py'),owner=owner,required_original_root=str(WAVE),runtime_freeze_sha256=sha(ROOT/'runtime_freeze.json'),queue_sha256=sha(ROOT/'persistent_queue.json'),handoff_sha256=sha(ROOT/'handoff_receipt.json'),stage_count=89,strict_FKL_first_position1=1,full_backfill_position1=74,new_GPU_dispatches=0)
 write(PARENT/'followup_registry'/('matrix-'+PROGRAM+'-strict-v5.json'),declaration);write(registry,declaration)
 write(MATRIX/'active_successor.json',declaration);write(MATRIX/'remaining20_successor.json',dict(declaration,alias_of_combined_queue=True))
 write(ROOT/'launch_receipt.json',dict(status='launched_safe_strict_v5' if launched else 'adopted_existing_v5',**declaration,argv=argv,unix=time.time()))
 print(json.dumps(read(ROOT/'launch_receipt.json')),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:first_error(ROOT/'handoff_first_error.json',e);raise
