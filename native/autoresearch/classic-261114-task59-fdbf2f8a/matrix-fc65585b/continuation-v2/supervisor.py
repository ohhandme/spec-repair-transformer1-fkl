"""One durable successor: observe original queue, then own its released controller lock."""
import subprocess,sys,time,traceback
from common import *
def live_workers(argv):
 found=[]
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=p.joinpath('cmdline').read_bytes().split(b'\0');args=[x.decode() for x in args if x]
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if args==argv:
   x=identity(int(p.name))
   if x and x['state']!='Z':found.append(x)
 return found
def stages(priority):
 jobs=[{'name':'fkl_gate','argv':[sys.executable,'-B','-u',str(ROOT/'fkl_gpu.py'),'--phase','gate'],'out':str(ROOT/'fkl/gate'),'optional_gate':True}]
 for group in priority:
  for phase in ['smoke','train','eval']:
   jobs.append({'name':group+'_'+phase,'argv':[sys.executable,'-B','-u',str(ROOT/'priority_worker.py'),'--group',group,'--phase',phase],'out':str(ROOT/'runs'/group/phase)})
 jobs.append({'name':'fkl_backfill','argv':[sys.executable,'-B','-u',str(ROOT/'fkl_gpu.py'),'--phase','backfill'],'out':str(ROOT/'fkl/backfill'),'optional_gate':True})
 return jobs
def main():
 own=open(ROOT/'supervisor.lock','a+')
 try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:print('EXISTING_SUCCESSOR_NO_DISPATCH');return
 freeze=verify_freeze();assert read(ROOT/'runtime_CPU_receipt.json')['status']=='pass'
 write(ROOT/'supervisor_identity.json',dict(identity(),source_sha256=sha(ROOT/'supervisor.py'),freeze_sha256=sha(ROOT/'runtime_freeze.json'),role='single_waiting_successor',unix=time.time()))
 jobs=stages(freeze['priority']);immutable(ROOT/'persistent_queue.json',{'stages':jobs,'required_prior':str(WAVE),'no_runtime_cap':True,'independent300_used':False})
 event('successor_registered_waiting_prior')
 while not prior_ready():
  old=read(WAVE/'controller_identity.json');live=alive(old)
  if not live:
   # If completion publication raced process exit, read normal terminal again.
   if prior_ready():break
   raise RuntimeError('original controller exited without complete prior queue; diagnose, never redispatch its jobs')
  write(ROOT/'supervisor_status.json',{'phase':'waiting_original_queue','prior_pid':old['pid'],'prior_alive':True,'GPU_dispatches':0,'pid':os.getpid(),'unix':time.time()})
  time.sleep(15)
 control=open(CONTROLLER_LOCK,'a+')
 while True:
  try:fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:
   write(ROOT/'supervisor_status.json',{'phase':'waiting_released_controller_lock','pid':os.getpid(),'unix':time.time()});time.sleep(15)
 assert prior_ready()
 write(PARENT/'continuation_active.json',{'version':4,'root':str(ROOT),'pid':os.getpid(),'start_tick':identity()['start_tick'],'prior_completion_sha256':sha(WAVE/'completion.json'),'freeze_sha256':sha(ROOT/'runtime_freeze.json'),'unix':time.time()})
 state=read(ROOT/'queue_state.json') if (ROOT/'queue_state.json').exists() else {'stages':{}}
 for job in jobs:
  name=job['name'];out=pathlib.Path(job['out']);done=out/'completion.json';child=None
  if name=='fkl_backfill' and read(ROOT/'fkl/gate/completion.json')['status']!='complete':
   state['stages'][name]={'status':'blocked_by_exact_reconstruction_gate'};write(ROOT/'queue_state.json',state);continue
  if done.exists():
   d=read(done);assert d['status']=='complete' or job.get('optional_gate') and d['status']=='blocked'
   if (out/'gpu_lease.json').exists():assert read(out/'gpu_lease.json')['status']=='released'
   state['stages'][name]={'status':d['status'],'completion_sha256':sha(done)};write(ROOT/'queue_state.json',state);continue
  existing=live_workers(job['argv']);assert len(existing)<=1,'duplicate matching workers require diagnosis'
  if existing:owner=existing[0];event('adopt_existing_child',stage=name,child_pid=owner['pid'])
  else:
   old=state['stages'].get(name)
   assert not old or old.get('status')=='launching', 'previous worker ended without terminal; diagnosis required'
   assert not (out/'first_error.json').exists(), 'preserved stage failure needs diagnosis'
   # Every child also checks the same lease atomically. This wait does not
   # acquire or modify another process's GPU lock.
   while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip():
    write(ROOT/'supervisor_status.json',{'phase':'waiting_GPU_idle','next_stage':name,'pid':os.getpid(),'unix':time.time()});time.sleep(15)
   state['stages'][name]={'status':'launching','argv':job['argv']};write(ROOT/'queue_state.json',state)
   with (ROOT/(name+'.log')).open('ab') as log:
    child=subprocess.Popen(job['argv'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   owner=identity(child.pid);assert owner
  state['stages'][name]=dict(owner,status='running',argv=job['argv']);write(ROOT/'queue_state.json',state)
  event('child_observed',stage=name,child_pid=owner['pid'])
  while alive(owner):
   write(ROOT/'supervisor_status.json',{'phase':'child_running','stage':name,'child_pid':owner['pid'],'pid':os.getpid(),'unix':time.time()});time.sleep(15)
  rc=child.wait() if child else None;assert rc in (None,0) and done.exists(), f'{name} failed rc={rc}'
  terminal=read(done)['status'];assert terminal=='complete' or job.get('optional_gate') and terminal=='blocked'
  assert read(out/'gpu_lease.json')['status']=='released'
  state['stages'][name]={'status':terminal,'completion_sha256':sha(done)};write(ROOT/'queue_state.json',state);event('child_terminal',stage=name,status=terminal)
 write(ROOT/'completion.json',{'status':'complete','stages':state['stages'],'FKL_training_dispatched':False,'independent300_used':False,'unix':time.time()})
if __name__=='__main__':
 try:main()
 except Exception as e:write(ROOT/'supervisor_first_error.json',{'error':repr(e),'traceback':traceback.format_exc(),'pid':os.getpid(),'unix':time.time()});raise
