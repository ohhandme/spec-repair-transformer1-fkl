import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,pathlib,sys,tempfile,time
from common import *
from queue20 import Queue,stages,live_workers,record_supervisor
class InjectedControllerStop(Exception):pass
WORKER='''import fcntl,json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);mode=sys.argv[2];p.mkdir(parents=True,exist_ok=True)
f=open(p/'worker.lock','a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
count=p/'calls.json';n=json.loads(count.read_text())+1 if count.exists() else 1;count.write_text(json.dumps(n))
time.sleep(.3)
if mode=='crash_once' and n==1:
 (p/'checkpoint_fixture.json').write_text(json.dumps({'saved_update':1}));os._exit(7)
if mode in ('fkl','shared'):
 (p/'first_error.json').write_text(json.dumps({'category':'fkl_specific' if mode=='fkl' else 'shared_integrity','error':'CPU fixture'}))
 if mode=='fkl':(p/'completion.json').write_text(json.dumps({'status':'blocked'}))
 sys.exit(1)
(p/'completion.json').write_text(json.dumps({'status':'complete','calls':n,'CPU_fixture':True}))
'''
def main():
 tests=[]
 with tempfile.TemporaryDirectory() as td:
  root=pathlib.Path(td);script=root/'worker.py';script.write_text(WORKER)
  def job(name,mode='ok',fkl=False):return dict(name=name,kind='fixture',argv=[sys.executable,str(script),str(root/name),mode],out=str(root/name),optional_fkl=fkl)
  def validate(j):
   p=pathlib.Path(j['out'])/'completion.json'
   if not p.exists():return None
   d=read(p);return 'complete' if d['status']=='complete' else 'blocked_fkl'
  def cleanup(j):return not live_workers(j['argv'])
  def create(jobs,hook=None):return Queue(root,jobs,validate,cleanup,lambda:True,.05,hook)
  j=job('normal');q=create([j]);assert q.run_one(j)=='complete';before=sha(root/'normal/completion.json');q=create([j]);assert q.run_one(j)=='complete' and read(root/'normal/calls.json')==1 and sha(root/'normal/completion.json')==before;tests.append('completed_stage_revalidated_without_redispatch')
  j=job('spawn_crash')
  def hook(j,owner):raise InjectedControllerStop()
  try:create([j],hook).run_one(j)
  except InjectedControllerStop:pass
  else:raise AssertionError('controller stop not injected')
  assert read(root/'queue_state.json')['stages']['spawn_crash']['status']=='launching'
  assert create([j]).run_one(j)=='complete' and read(root/'spawn_crash/calls.json')==1;tests.append('Popen_state_crash_window_adopts_real_live_child_once')
  j=job('worker_crash','crash_once');assert create([j]).run_one(j)=='complete' and read(root/'worker_crash/calls.json')==2 and (root/'worker_crash/checkpoint_fixture.json').exists();tests.append('abrupt_worker_exit_auto_resumes_once_after_cleanup')
  j=job('fkl_only','fkl',True);assert create([j]).run_one(j)=='blocked_fkl';j2=job('ce_after_fkl');assert create([j2]).run_one(j2)=='complete';tests.append('scoped_FKL_failure_then_independent_CE')
  j=job('shared_error','shared',True)
  try:create([j]).run_one(j)
  except SharedIntegrityError:pass
  else:raise AssertionError('shared error was skipped')
  assert read(root/'shared_error/calls.json')==1;tests.append('shared_integrity_error_blocks_even_FKL_job')
  j=job('cleanup_required','fkl',True)
  try:Queue(root,[j],validate,lambda j:False,lambda:True,.05).run_one(j)
  except SharedIntegrityError:pass
  else:raise AssertionError('unreleased cleanup was skipped')
  tests.append('FKL_continuation_requires_dead_child_and_cleanup_proof')
  jobs=stages();assert len(jobs)==89
  assert len({j['name'] for j in jobs})==89 and not any(j.get('group') in H.INTERVENTIONS and j.get('phase')=='train' for j in jobs)
  gatepos=next(i for i,j in enumerate(jobs) if j['name']=='fkl_gate');backfill=next(i for i,j in enumerate(jobs) if j['name']=='fkl_backfill');assert gatepos==0 and backfill==73 and all(not j.get('requires_full_vocab') for j in jobs[1:backfill])
  assert all(next(i for i,j in enumerate(jobs) if j['name']==g+'_smoke')>next(i for i,j in enumerate(jobs) if j['name']=='H-VQ64-weighted_train') for g in H.INTERVENTIONS)
  tests.append('89_unique_stages_strict_gate_first_72_nonFKL_before_full_fill_five_FKL_gated')
  identroot=root/'identity';identroot.mkdir();write(identroot/'supervisor_identity.json',{'pid':99999999,'start_tick':'0'})
  record_supervisor(identroot,identity());assert (identroot/'supervisor_history/99999999-0.json').exists();tests.append('supervisor_restart_preserves_dead_owner_identity')
  child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']);write(identroot/'supervisor_identity.json',identity(child.pid))
  try:
   try:record_supervisor(identroot,identity())
   except SharedIntegrityError:tests.append('supervisor_restart_refuses_other_live_owner')
   else:raise AssertionError('live owner replaced')
  finally:child.terminate();child.wait(timeout=10)
  own=open(root/'supervisor.lock','a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
  cp=subprocess.run([sys.executable,'-c',"import fcntl,sys;f=open(sys.argv[1],'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)",str(root/'supervisor.lock')],capture_output=True);assert cp.returncode!=0;own.close();tests.append('real_process_supervisor_mutex_rejects_duplicate')
 write(ROOT/'queue_CPU_receipt.json',dict(status='pass',tests=tests,count=len(tests),GPU_used=False,source_sha256=sha(ROOT/'queue20.py'),unix=time.time()));print(json.dumps({'status':'pass','tests':tests}),flush=True)
if __name__=='__main__':
 import heads as H
 import subprocess
 try:main()
 except Exception as e:first_error(ROOT/'queue_CPU_first_error.json',e);raise
