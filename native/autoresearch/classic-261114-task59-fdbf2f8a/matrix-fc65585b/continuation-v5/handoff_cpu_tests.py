import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import fcntl,json,pathlib,subprocess,sys,tempfile,time
from common import ROOT,write,sha,read,alive
from guards import SharedIntegrityError
from handoff_guard import snapshot,retire_waiters
SCRIPT='''import fcntl,json,os,pathlib,time
r=pathlib.Path(__file__).parent;lock=open(r/'supervisor.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
stat=pathlib.Path('/proc/self/stat').read_text().split();owner={'pid':os.getpid(),'start_tick':stat[21]}
(r/'supervisor_identity.json').write_text(json.dumps(owner))
(r/'supervisor_status.json').write_text(json.dumps({'pid':os.getpid(),'phase':'waiting_original_queue','GPU_dispatches':0,'unix':time.time()}))
while True:time.sleep(1)
'''
def main():
 tests=[];children=[]
 with tempfile.TemporaryDirectory() as td:
  base=pathlib.Path(td)
  def launch(name):
   root=base/name;root.mkdir();(root/'waiter.py').write_text(SCRIPT);write(root/'runtime_freeze.json',{});write(root/'persistent_queue.json',{})
   p=subprocess.Popen([sys.executable,str(root/'waiter.py')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);children.append(p)
   end=time.time()+10
   while not (root/'supervisor_status.json').exists() and time.time()<end:time.sleep(.02)
   return dict(root=str(root),entry='waiter.py',phase='waiting_original_queue')
  def denied(spec):
   try:retire_waiters([spec])
   except SharedIntegrityError:assert alive(read(pathlib.Path(spec['root'])/'supervisor_identity.json'));return
   raise AssertionError('unsafe handoff accepted')
  try:
   a=launch('a');b=launch('b');r=pathlib.Path(a['root']);d=read(r/'supervisor_status.json')
   write(r/'supervisor_status.json',dict(d,GPU_dispatches=1));denied(a);write(r/'supervisor_status.json',d);tests.append('nonzero_dispatch_preserves_live_supervisor')
   write(r/'queue_state.json',{'stages':{'gate':{'status':'launching'}}});denied(a);write(r/'queue_state.json',{'stages':{}});tests.append('launch_intent_blocks_retirement_before_Popen')
   (r/'fkl/gate').mkdir(parents=True);write(r/'fkl/gate/manifest.json',{'preserved':True});denied(a);(r/'fkl/gate/manifest.json').rename(r/'test_preserved_manifest.json');tests.append('real_stage_artifact_prevents_handoff')
   c=launch('c')
   def race():
    rr=pathlib.Path(c['root']);x=read(rr/'supervisor_status.json');write(rr/'supervisor_status.json',dict(x,GPU_dispatches=1))
   try:retire_waiters([c],race)
   except SharedIntegrityError:assert alive(read(pathlib.Path(c['root'])/'supervisor_identity.json'))
   else:raise AssertionError('dispatch race accepted')
   tests.append('post_SIGSTOP_recheck_detects_dispatch_race_and_resumes_owner')
   lease=base/'original_gpu_lease.json';write(lease,{'status':'held','owner':'original'});lease_sha=sha(lease);lock=open(base/'original_controller.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   def original_guard():assert sha(lease)==lease_sha
   before=[snapshot(a),snapshot(b)];receipt=retire_waiters([a,b],original_guard);assert receipt['status']=='safe_zero_dispatch_handoff';assert all(not alive(x['owner']) for x in before)
   assert all(sha(pathlib.Path(x['root'])/'waiter.py')==x['source_sha256'] for x in before);tests.append('two_waiting_owners_retired_with_files_and_original_lease_preserved')
   for x in [a,b]:
    with open(pathlib.Path(x['root'])/'supervisor.lock','a+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   tests.append('both_prior_supervisor_kernel_mutexes_released_before_new_dispatch')
   lock.close()
  finally:
   for p in children:
    if p.poll() is None:p.terminate()
    p.wait(timeout=10)
 write(ROOT/'handoff_CPU_receipt.json',dict(status='pass',count=len(tests),tests=tests,GPU_used=False,source_sha256=sha(ROOT/'handoff_guard.py'),unix=time.time()));print(json.dumps({'status':'pass','tests':tests}))
if __name__=='__main__':main()
