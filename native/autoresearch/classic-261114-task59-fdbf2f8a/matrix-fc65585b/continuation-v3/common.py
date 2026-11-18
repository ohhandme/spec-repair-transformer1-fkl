from guards import SharedIntegrityError, FKLSpecificError, require, category
import contextlib,fcntl,hashlib,json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
MATRIX=ROOT.parent
V2=MATRIX/'continuation-v2'
PARENT=MATRIX.parent
WAVE=PARENT/'wave2-dev503-20261114'
TRAIN=pathlib.Path('/data/a100-repair-raw5000-20261113-v1')
DATA=pathlib.Path('/data/a100-ultrachat5000-vanilla-20261111-7858dc6a')
GPU_LOCK='/data/a100-first-rejection-20261111-9b21f68f/a1_gpu.lock'
CONTROLLER_LOCK=PARENT/'continuation-v1/controller.lock'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def lines(p):return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x]
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
 with tmp.open('w') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def immutable(p,x):
 if pathlib.Path(p).exists():assert read(p)==x, 'immutable artifact differs: '+str(p)
 else:write(p,x)
def event(kind,**fields):
 x=dict(fields,kind=kind,pid=os.getpid(),unix=time.time())
 with (ROOT/'events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
def identity(pid=None):
 pid=pid or os.getpid();p=pathlib.Path(f'/proc/{pid}/stat')
 if not p.exists():return None
 x=p.read_text().split();return {'pid':pid,'start_tick':x[21],'state':x[2]}
def alive(x):
 now=identity(x['pid']);return bool(now and now['state']!='Z' and now['start_tick']==x['start_tick'])
def prior_ready(base=WAVE):
 for stage in ['six_dev','ack_training','ack_dev']:
  d=base/stage/'completion.json';l=base/stage/'gpu_lease.json'
  if not d.exists() or read(d).get('status')!='complete':return False
  if not l.exists() or read(l).get('status')!='released':return False
 return (base/'completion.json').exists() and read(base/'completion.json').get('status')=='complete'
def verify_freeze():
 f=read(ROOT/'runtime_freeze.json')
 for p,h in f['files'].items():require(sha(p)==h, 'source or dependency changed: '+p)
 return f
@contextlib.contextmanager
def lease(out,require_prior=True):
 if require_prior:assert prior_ready(), 'original queue is not normally complete/released'
 lock=open(GPU_LOCK,'a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
 assert not active, 'GPU occupied; do not preempt: '+active
 payload=dict(status='held',**identity(),unix=time.time(),lock=GPU_LOCK,no_added_runtime_limit=True)
 write(out/'gpu_lease.json',payload)
 try:yield
 finally:
  write(out/'gpu_lease.json',dict(payload,status='released',released_unix=time.time()))
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def first_error(path,exc,**extra):
 import traceback
 path=pathlib.Path(path)
 if not path.exists():immutable(path,dict(error=repr(exc),category=category(exc),traceback=traceback.format_exc(),pid=os.getpid(),unix=time.time(),**extra))
def kernel_gpu_free():
 with open(GPU_LOCK,'a+') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return False
  fcntl.flock(f,fcntl.LOCK_UN);return True
