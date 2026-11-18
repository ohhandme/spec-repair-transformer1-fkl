"""Retire only proven zero-dispatch waiting supervisors; never touch a worker/lease."""
import fcntl,os,pathlib,signal,time
from common import read,sha,require,identity,alive

def process(pid):
 d=identity(pid)
 if d is None:return None
 p=pathlib.Path('/proc')/str(pid)
 try:d.update(argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x],ppid=int((p/'stat').read_text().rpartition(')')[2].split()[1]))
 except FileNotFoundError:return None
 return d

def snapshot(spec):
 root=pathlib.Path(spec['root']);owner=read(root/'supervisor_identity.json');status=read(root/'supervisor_status.json')
 require(alive(owner),'waiting supervisor is no longer live')
 live=process(owner['pid']);require(str(root/spec['entry']) in live['argv'],'waiting supervisor executable differs')
 require(status['pid']==owner['pid'] and status['phase']==spec['phase'] and status.get('GPU_dispatches')==0,'supervisor is not proven zero-dispatch waiting')
 require(time.time()-status['unix']<90,'waiting heartbeat is stale')
 state=read(root/'queue_state.json') if (root/'queue_state.json').exists() else None
 require(not state or not state.get('stages'),'stage already dispatched or registered; retain original execution')
 require(not (root/'completion.json').exists() and not (root/'supervisor_first_error.json').exists(),'terminal/error requires separate diagnosis')
 for folder in ['runs','fkl']:
  require(not any(p.is_file() for p in (root/folder).rglob('*')),'production artifact exists; no retirement')
 inventory=[d for p in pathlib.Path('/proc').glob('[0-9]*') if (d:=process(int(p.name))) and d['state']!='Z'];desc={owner['pid']}
 while True:
  new=desc|{d['pid'] for d in inventory if d['ppid'] in desc}
  if new==desc:break
  desc=new
 require(desc=={owner['pid']},'supervisor has a child; no retirement')
 matching=[d for d in inventory if any(a.startswith(str(root)+'/') and a.endswith('.py') for a in d['argv'])]
 require(len(matching)==1 and matching[0]['pid']==owner['pid'],'other live executable under prior revision')
 return dict(owner=owner,status=status,process=live,queue_state=state,source_sha256=sha(root/spec['entry']),queue_sha256=sha(root/'persistent_queue.json'),freeze_sha256=sha(root/'runtime_freeze.json'),root=str(root))

def retire_waiters(specs,original_guard=lambda:None):
 before=[snapshot(s) for s in specs];stopped=[];after=[]
 try:
  original_guard()
  for s,b in reversed(list(zip(specs,before))):
   owner=b['owner'];require(alive(owner),'identity changed before pause');os.kill(owner['pid'],signal.SIGSTOP);stopped.append(owner)
   deadline=time.time()+10
   while alive(owner) and process(owner['pid'])['state'] not in ('T','t') and time.time()<deadline:time.sleep(.02)
   require(alive(owner) and process(owner['pid'])['state'] in ('T','t'),'waiting supervisor did not enter stopped state')
  # Audit after pausing closes the observation-to-retirement dispatch race.
  for s in specs:after.append(snapshot(s))
  original_guard()
  for owner in stopped:
   require(alive(owner),'identity changed before termination');os.kill(owner['pid'],signal.SIGTERM);os.kill(owner['pid'],signal.SIGCONT)
   deadline=time.time()+10
   while alive(owner) and time.time()<deadline:time.sleep(.05)
   require(not alive(owner),'waiting supervisor did not exit; do not launch successor')
  for s in specs:
   with open(pathlib.Path(s['root'])/'supervisor.lock','a+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
  original_guard()
  return dict(status='safe_zero_dispatch_handoff',before=before,paused_recheck=after,retired=[dict(x,alive_after=False) for x in stopped],prior_files_modified=[],GPU_lease_modified=False,prior_supervisor_kernel_locks_released=True,unix=time.time())
 finally:
  for owner in stopped:
   if alive(owner) and process(owner['pid'])['state'] in ('T','t'):os.kill(owner['pid'],signal.SIGCONT)
