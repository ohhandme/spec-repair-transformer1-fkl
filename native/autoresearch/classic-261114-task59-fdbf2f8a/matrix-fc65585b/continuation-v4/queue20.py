"""Single sequential remaining-20 queue, durable adoption and verified restart."""
import pathlib,subprocess,sys,time
from common import *
import checkpoints as CP
import heads as H
def live_workers(argv):
 found=[]
 for p in pathlib.Path('/proc').glob('[0-9]*'):
  try:args=[x.decode() for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x];owner=identity(int(p.name))
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
  if args==argv and owner and owner['state']!='Z':found.append(owner)
 return found
def record_supervisor(root,payload):
 root=pathlib.Path(root);path=root/'supervisor_identity.json'
 require(payload['pid']==os.getpid(),'supervisor identity must be the current process')
 if path.exists():
  old=read(path)
  require(not alive(old) or (old['pid']==payload['pid'] and old['start_tick']==payload['start_tick']),'different live supervisor owns this queue')
  immutable(root/'supervisor_history'/f'{old["pid"]}-{old["start_tick"]}.json',old)
 write(path,payload)
def stages(root=ROOT):
 jobs=[]
 def add(g,phase):jobs.append(dict(name=g+'_'+phase,kind='group',group=g,phase=phase,argv=[sys.executable,'-B','-u',str(root/'worker20.py'),'--group',g,'--phase',phase],out=str(root/'runs'/g/phase),requires_full_vocab=g in H.FKL))
 for g in H.GROUPS:
  if g in H.FKL or g in H.INTERVENTIONS:continue
  for phase in ('smoke','train','eval'):add(g,phase)
  if g=='H-VQ64-weighted':
   for intervention in sorted(H.INTERVENTIONS):
    for phase in ('smoke','eval'):add(intervention,phase)
 for phase in ('gate','backfill'):
  jobs.append(dict(name='fkl_'+phase,kind='fkl',phase=phase,optional_fkl=True,argv=[sys.executable,'-B','-u',str(root/'fkl_all_verified_gpu.py'),'--phase',phase],out=str(root/'fkl'/phase)))
 for g in H.GROUPS:
  if g in H.FKL:
   for phase in ('smoke','train','eval'):add(g,phase)
 require(len(jobs)==59 and len({j['name'] for j in jobs})==59,'remaining queue coverage differs');return jobs
def valid_terminal(job,root=ROOT):
 out=pathlib.Path(job['out']);p=out/'completion.json'
 if not p.exists():return None
 d=read(p)
 if job['kind']=='fkl':
  if d['status']=='blocked':return 'blocked_fkl'
  from fkl_integrity import coverage_inputs,validate_manifest,gate_jobs
  frozen,coverage=coverage_inputs(V3/'labels');expected=gate_jobs(frozen) if job['phase']=='gate' else frozen
  require(d['all_label_mapping_sha256']==coverage['mapping_receipt_sha256'],'FKL terminal frozen label binding differs');validate_manifest(out/'manifest.jsonl',expected,complete=True,terminal=d)
  return 'complete'
 require(d['status']=='complete' and d['group']==job['group'] and d['CPU_fixture'] is False,'invalid production worker terminal')
 require(d['config_sha256']==sha(root/'configs'/(job['group']+'.json')) and d['runtime_freeze_sha256']==sha(root/'runtime_freeze.json'),'terminal source/config changed')
 c=read(root/'configs'/(job['group']+'.json'))
 if job['phase']=='train':
  for s in read(out/'selected.json'):CP.binding(root,job['group'],s)
 else:
  from engine20 import validate_request_files
  initial=job['phase']=='smoke' and not c['evaluation_only'];selections=['initial'] if initial else list(read(root/'runs'/c['checkpoint_owner']/'train/selected.json'))
  roster=read(WAVE/'roster503.json');ids=set(read(WAVE/'smoke_ids.json'));roster=[r for r in roster if job['phase']=='eval' or r['request_id'] in ids];index={};seen=set()
  for s in selections:
   bound,_=CP.binding(root,job['group'],s,initial=initial)
   if bound['checkpoint_sha256'] in seen:continue
   seen.add(bound['checkpoint_sha256'])
   for req in roster:
    dest=out/s/req['request_id'];validate_request_files(dest,req,bound);index[s+'/'+req['request_id']]=dict(path=str(dest),receipt_sha256=sha(dest/'receipt.json'))
  require(sha(out/'sealed_index.json')==d['sealed_index_sha256'] and read(out/'sealed_index.json')==index,'terminal sealed index coverage differs')
 return 'complete'
def gpu_idle():
 return kernel_gpu_free() and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def cleanup_proof(job):
 out=pathlib.Path(job['out']);require(not live_workers(job['argv']),'live child cannot release/restart')
 if not gpu_idle():return False
 l=out/'gpu_lease.json'
 if l.exists() and read(l)['status']!='released':
  old=read(l);require(not alive(old),'lease owner still alive')
  # Preserve the old JSON; process exit releases the kernel lease, checked above.
  record=out/'recovery_leases'/f'{old["pid"]}-{old["start_tick"]}.json';immutable(record,old)
  write(out/'resume_cleanup_receipt.json',dict(previous_lease_sha256=sha(record),owner_dead=True,kernel_lease_free=True,GPU_idle=True,old_lease_JSON_preserved=True,unix=time.time()))
 return True
class Queue:
 def __init__(self,root,jobs,validate=valid_terminal,cleanup=cleanup_proof,idle=gpu_idle,poll=15,after_spawn=None):
  self.root=pathlib.Path(root);self.jobs=jobs;self.validate=validate;self.cleanup=cleanup;self.idle=idle;self.poll=poll;self.after_spawn=after_spawn
  self.path=self.root/'queue_state.json';self.state=read(self.path) if self.path.exists() else {'stages':{}}
 def save(self):write(self.path,self.state)
 def run_one(self,job):
  name=job['name'];out=pathlib.Path(job['out']);out.mkdir(parents=True,exist_ok=True);old=self.state['stages'].get(name,{});child=None;restarts=old.get('automatic_restarts',0)
  existing=live_workers(job['argv']);require(len(existing)<=1,'duplicate workers detected; no dispatch')
  if existing:owner=existing[0]
  else:
   # Completion can race process exit. Inspect it again after live inventory.
   if (out/'completion.json').exists():
    require(self.cleanup(job),'terminal child/GPU cleanup not yet verified');status=self.validate(job);require(status is not None,'terminal vanished')
    if status=='blocked_fkl':require(job.get('optional_fkl') and read(out/'first_error.json')['category']=='fkl_specific','blocking/shared failure cannot be skipped')
    self.state['stages'][name]=dict(status=status,completion_sha256=sha(out/'completion.json'),reused=True);self.save();return status
   error=out/'first_error.json'
   if error.exists():
    if job.get('optional_fkl') and read(error)['category']=='fkl_specific':
     require(self.cleanup(job),'FKL child cleanup incomplete');self.state['stages'][name]=dict(status='blocked_fkl',first_error_sha256=sha(error));self.save();return 'blocked_fkl'
    raise SharedIntegrityError('preserved worker first error blocks recovery: '+name)
   if old or (out/'identity.json').exists():
    require(self.cleanup(job),'abrupt exit cleanup unverified')
    self.state['stages'][name]=dict(status='resuming',previous=old,reason='child dead, GPU idle, kernel lease released; worker validates saved checkpoint/requests');self.save()
   while not self.idle():time.sleep(self.poll)
   # Intent + exact argv adoption cover death between Popen and state commit.
   self.state['stages'][name]=dict(status='launching',argv=job['argv'],automatic_restarts=restarts);self.save()
   with (self.root/(name+'.log')).open('ab') as log:child=subprocess.Popen(job['argv'],cwd=self.root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   owner=identity(child.pid);require(owner is not None,'child identity absent')
   if self.after_spawn:self.after_spawn(job,owner)
  self.state['stages'][name]=dict(owner,status='running',argv=job['argv'],automatic_restarts=restarts);self.save()
  while alive(owner):
   write(self.root/'supervisor_status.json',dict(phase='child_running',stage=name,child_pid=owner['pid'],pid=os.getpid(),unix=time.time()));time.sleep(self.poll)
  rc=child.wait() if child else None;require(self.cleanup(job),'child ended but cleanup not verified')
  error=out/'first_error.json'
  if not error.exists() and (rc not in (None,0) or not (out/'completion.json').exists()):
   immutable(out/'unexpected_exits'/f'{owner["pid"]}-{owner["start_tick"]}.json',dict(owner=owner,rc=rc,cleanup_verified=True,unix=time.time()))
   require(restarts<1,'repeated abrupt worker exit; preserved block')
   self.state['stages'][name]=dict(status='interrupted',automatic_restarts=restarts+1,previous_owner=owner);self.save();return self.run_one(job)
  if error.exists():
   require(job.get('optional_fkl') and read(error)['category']=='fkl_specific','shared/unclassified worker failure; preserve block')
   status='blocked_fkl'
  else:require(rc in (None,0),'child nonzero exit without scoped failure');status=self.validate(job);require(status is not None,'child ended without terminal')
  self.state['stages'][name]=dict(status=status,completion_sha256=sha(out/'completion.json') if (out/'completion.json').exists() else None,first_error_sha256=sha(error) if error.exists() else None,cleanup_verified=True);self.save();return status
def main():
 own=open(ROOT/'supervisor.lock','a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze=verify_freeze()
 require(read(ROOT/'CPU_E2E_receipt.json')['status']=='pass' and read(ROOT/'queue_CPU_receipt.json')['status']=='pass','CPU end-to-end/queue regression absent')
 record_supervisor(ROOT,dict(identity(),role='remaining20_successor_after_v3',source_sha256=sha(ROOT/'queue20.py'),freeze_sha256=sha(ROOT/'runtime_freeze.json')))
 while not prior_ready():
  owner=read(V3/'supervisor_identity.json')
  if not alive(owner):
   if prior_ready():break
   raise SharedIntegrityError('v3 exited without normal priority completion; keep existing queue for diagnosis')
  write(ROOT/'supervisor_status.json',dict(phase='waiting_v3_normal_terminal',prior_pid=owner['pid'],GPU_dispatches=0,pid=os.getpid(),unix=time.time()));time.sleep(15)
 control=open(CONTROLLER_LOCK,'a+')
 while True:
  try:fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(15)
 jobs=stages();require(jobs==read(ROOT/'persistent_queue.json')['stages'],'queue changed');q=Queue(ROOT,jobs)
 for job in jobs:
  if job['name']=='fkl_backfill' and q.state['stages'].get('fkl_gate',{}).get('status')!='complete':
   q.state['stages'][job['name']]={'status':'waiting_full_vocab_gate'};q.save();continue
  if job.get('requires_full_vocab') and q.state['stages'].get('fkl_backfill',{}).get('status')!='complete':
   q.state['stages'][job['name']]={'status':'waiting_full_vocab_acceptance'};q.save();continue
  q.run_one(job)
 blocked=[n for n,s in q.state['stages'].items() if s['status']!='complete']
 if blocked:
  write(ROOT/'nonFKL_completion.json',dict(status='nonFKL_complete_FKL_waiting',remaining=blocked,queue_state_sha256=sha(q.path),unix=time.time()))
  fcntl.flock(control,fcntl.LOCK_UN);control.close()
  # Observe a later independently accepted full-vocabulary receipt without
  # redispatching the known failed numeric gate or any completed CE job.
  while True:
   write(ROOT/'supervisor_status.json',dict(phase='nonFKL_complete_waiting_full_vocab_acceptance',remaining=blocked,pid=os.getpid(),unix=time.time()))
   full=ROOT/'fkl/backfill/completion.json'
   if full.exists() and read(full).get('FKL_training_admitted') is True:
    for phase in ('gate','backfill'):
     job=next(j for j in jobs if j['name']=='fkl_'+phase);require(valid_terminal(job)=='complete','new FKL acceptance invalid');q.state['stages'][job['name']]={'status':'complete','completion_sha256':sha(pathlib.Path(job['out'])/'completion.json'),'new_acceptance_verified':True}
    control=open(CONTROLLER_LOCK,'a+')
    try:fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:control.close();time.sleep(30);continue
    for job in jobs:
     if job.get('requires_full_vocab'):q.run_one(job)
    q.save();write(ROOT/'completion.json',dict(status='complete',stages=q.state['stages'],independent300_used=False,unix=time.time()));break
   time.sleep(30)
 else:write(ROOT/'completion.json',dict(status='complete',stages=q.state['stages'],independent300_used=False,unix=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:first_error(ROOT/'supervisor_first_error.json',e);raise
