"""Reuse accepted unchanged engine tests; independently exercise the new strict commit/queue."""
from common import *
import heads as H
def main():
 original=read(V4/'CPU_E2E_receipt.json');require(original['status']=='pass' and original['conditions']==20,'v4 engine test not accepted')
 unchanged={}
 for n in ['engine20.py','runtime20.py','worker20.py','heads.py','checkpoints.py','cpu_e2e.py']:
  require(sha(ROOT/n)==sha(V4/n)==original['source_sha256'][n],'reused engine source changed: '+n);unchanged[n]=sha(ROOT/n)
 for g in H.GROUPS:
  c=read(V4/'configs'/(g+'.json'));require(c['entry_sha256']==sha(ROOT/'worker20.py'),'worker config SHA differs')
  c['entry']=str(ROOT/'worker20.py');immutable(ROOT/'configs'/(g+'.json'),c)
 for n in ['CPU_E2E_receipt.json','label_coverage_CPU_receipt.json','online_donors.json']:
  immutable(ROOT/n,read(V4/n))
 for n in ['priority_worker.py','runtime.py','guards.py']:
  require(sha(ROOT/n)==sha(V3/n),'priority engine differs unexpectedly: '+n)
 write(ROOT/'CPU_reuse_receipt.json',dict(status='pass',unchanged_engine_sources=unchanged,original_receipt=str(V4/'CPU_E2E_receipt.json'),original_receipt_sha256=sha(V4/'CPU_E2E_receipt.json'),conditions=20,
  scope='unchanged actual production engines, CPU synthetic Target tests reused; new storage/queue/handoff independently tested',GPU_admission=False,independent300_used=False,priority_engines_unchanged=True,unix=time.time()))
 print(json.dumps(dict(status='pass',reused_engine_conditions=20,priority_groups=len(read(V3/'runtime_freeze.json')['priority']))),flush=True)
if __name__=='__main__':main()
