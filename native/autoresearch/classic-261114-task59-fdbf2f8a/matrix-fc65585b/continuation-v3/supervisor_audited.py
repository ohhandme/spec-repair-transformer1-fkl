"""Single successor; no FKL-specific failure can bypass cleanup or shared integrity."""
import subprocess,sys,time,traceback
from common import *
from guards import dispatch_decision,terminal_decision
def live_workers(argv):
 found=[]
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=[x.decode() for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x];owner=identity(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if args==argv and owner and owner['state']!='Z':found.append(owner)
 return found
def stages(priority):
 jobs=[dict(name='fkl_gate',argv=[sys.executable,'-B','-u',str(ROOT/'fkl_all_gpu.py'),'--phase','gate'],out=str(ROOT/'fkl/gate'),optional_fkl=True)]
 for group in priority:
  for phase in ('smoke','train','eval'):
   jobs.append(dict(name=group+'_'+phase,argv=[sys.executable,'-B','-u',str(ROOT/'priority_worker.py'),'--group',group,'--phase',phase],out=str(ROOT/'runs'/group/phase)))
 jobs.append(dict(name='fkl_backfill',argv=[sys.executable,'-B','-u',str(ROOT/'fkl_all_gpu.py'),'--phase','backfill'],out=str(ROOT/'fkl/backfill'),optional_fkl=True))
 return jobs
def inspect_terminal(job,rc=None):
 out=pathlib.Path(job['out']);terminal=read(out/'completion.json') if (out/'completion.json').exists() else None
 error=read(out/'first_error.json') if (out/'first_error.json').exists() else None
 lease=read(out/'gpu_lease.json') if (out/'gpu_lease.json').exists() else None
 live=live_workers(job['argv'])
 return terminal_decision(job,terminal,error,bool(live),bool(lease and lease['status']=='released'),kernel_gpu_free(),rc),terminal,error,lease
def main():
 own=open(ROOT/'supervisor.lock','a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 freeze=verify_freeze();require(read(ROOT/'audit_final_CPU_receipt.json')['status']=='pass','CPU audit not passed')
 write(ROOT/'supervisor_identity.json',dict(identity(),source_sha256=sha(ROOT/'supervisor_audited.py'),freeze_sha256=sha(ROOT/'runtime_freeze.json'),role='single_waiting_successor',unix=time.time()))
 jobs=stages(freeze['priority']);require(jobs==read(ROOT/'persistent_queue.json')['stages'],'queue changed')
 while not prior_ready():
  old=read(WAVE/'controller_identity.json')
  if not alive(old):
   if prior_ready():break
   raise SharedIntegrityError('original controller exited without complete prior queue; do not redispatch')
  write(ROOT/'supervisor_status.json',dict(phase='waiting_original_queue',prior_pid=old['pid'],prior_alive=True,GPU_dispatches=0,pid=os.getpid(),unix=time.time()));time.sleep(15)
 control=open(CONTROLLER_LOCK,'a+')
 while True:
  try:fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(15)
 require(prior_ready(),'original queue is not complete')
 write(PARENT/'continuation_active.json',dict(version=5,root=str(ROOT),**identity(),prior_completion_sha256=sha(WAVE/'completion.json'),unix=time.time()))
 state=read(ROOT/'queue_state.json') if (ROOT/'queue_state.json').exists() else {'stages':{}}
 for job in jobs:
  name=job['name'];out=pathlib.Path(job['out']);out.mkdir(parents=True,exist_ok=True);child=None
  if name=='fkl_backfill' and state['stages']['fkl_gate']['status']!='complete':
   state['stages'][name]={'status':'blocked_by_fkl_gate'};write(ROOT/'queue_state.json',state);continue
  # Read live processes first, then terminal: a completion racing process exit
  # becomes a terminal decision, never another launch.
  existing=live_workers(job['argv']);terminal=(out/'completion.json').exists();error=(out/'first_error.json').exists()
  action=dispatch_decision(existing,terminal,error,state['stages'].get(name))
  if action=='block':raise SharedIntegrityError(name+' exited without terminal; diagnosis required')
  if action=='launch':
   while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip() or not kernel_gpu_free():time.sleep(15)
   # A controller crash after Popen is recovered by exact argv adoption;
   # an exclusive worker lock also prohibits two actual executions.
   state['stages'][name]=dict(status='launching',argv=job['argv'],unix=time.time());write(ROOT/'queue_state.json',state)
   with (ROOT/(name+'.log')).open('ab') as log:child=subprocess.Popen(job['argv'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   owner=identity(child.pid);require(owner is not None,'child identity absent')
  elif action=='adopt':owner=existing[0]
  else:owner=None
  if owner:
   state['stages'][name]=dict(owner,status='running',argv=job['argv']);write(ROOT/'queue_state.json',state)
   while alive(owner):
    write(ROOT/'supervisor_status.json',dict(phase='child_running',stage=name,child_pid=owner['pid'],pid=os.getpid(),unix=time.time()));time.sleep(15)
  rc=child.wait() if child else None
  decision,terminal,error,lease=inspect_terminal(job,rc)
  # No automatic force release. A dead child with an unreleased lease receipt
  # stays blocked for diagnosis; normal worker finally publishes release.
  if decision not in ('complete','blocked_fkl_continue_ce'):
   write(ROOT/'cleanup_block.json',dict(stage=name,decision=decision,terminal=terminal,error=error,lease=lease,unix=time.time()))
   raise SharedIntegrityError(name+': '+decision)
  state['stages'][name]=dict(status=decision,terminal_sha256=sha(out/'completion.json') if terminal else None,first_error_sha256=sha(out/'first_error.json') if error else None,cleanup_verified=True,unix=time.time());write(ROOT/'queue_state.json',state)
  event('child_terminal',stage=name,status=decision,rc=rc)
 write(ROOT/'completion.json',dict(status='complete',stages=state['stages'],remaining20='CPU prepared; separate admission required before GPU stages',independent300_used=False,unix=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:first_error(ROOT/'supervisor_first_error.json',e);raise
