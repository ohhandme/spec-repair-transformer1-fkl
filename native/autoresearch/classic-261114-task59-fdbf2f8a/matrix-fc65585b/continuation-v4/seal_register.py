"""Register one disjoint successor after v3; never alter a live v3 file/queue/lease."""
import subprocess,sys,time
from common import *
from queue20 import stages,live_workers
def main():
 lock=open(MATRIX/'remaining20_registration.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 require(sha(MATRIX/'program.md')=='fc65585bce083e6c95a3237c081abe91484273639db4d0527130389dce5ff71b','matrix program changed')
 e2e=read(ROOT/'CPU_E2E_receipt.json');q=read(ROOT/'queue_CPU_receipt.json');fkl=read(ROOT/'FKL_resume_CPU_receipt.json');coverage=read(ROOT/'label_coverage_CPU_receipt.json')
 require(e2e['status']=='pass' and e2e['conditions']==20 and len(e2e['groups'])==20,'remaining20 CPU end-to-end incomplete')
 for n in ['engine20.py','runtime20.py','worker20.py','heads.py','checkpoints.py','cpu_e2e.py']:
  require(e2e['source_sha256'][n]==sha(ROOT/n),'CPU E2E executable changed: '+n)
 require(q['status']=='pass' and q['source_sha256']==sha(ROOT/'queue20.py'),'queue regression/source differs')
 require(fkl['status']=='pass' and fkl['source_sha256']==sha(ROOT/'fkl_integrity.py'),'FKL integrity regression/source differs')
 require(coverage['status']=='pass' and coverage['source_sha256']==sha(ROOT/'fkl_integrity.py') and coverage['coverage']['occurrences']==1370920,'actual frozen-label coverage differs')
 require(sha(V3/'runtime_freeze.json')=='4227da7ed2a34d89d809d182b75801c6f02c6e6b06f627e4e09498b733cc21e4','parent v3 freeze changed')
 require(sha(V3/'persistent_queue.json')=='7a4d5b2807faa39166af645db55fa69dcad4bc7c67eecf7625064b9334bf75d8','parent v3 queue changed')
 require(sha(V3/'supervisor_final.py')=='60dc39b94d519378dd5a89ae7af8dcedad59af0525c20afb7ea701c90a77b31a','parent v3 live source changed')
 parent=read(V3/'supervisor_identity.json');require(alive(parent) or prior_ready(),'parent v3 is neither alive nor normally complete')
 files=dict(read(V3/'runtime_freeze.json')['files'])
 for p in ROOT.glob('*.py'):files[str(p)]=sha(p)
 for p in (ROOT/'configs').glob('*.json'):files[str(p)]=sha(p)
 for n in ['CPU_E2E_receipt.json','queue_CPU_receipt.json','FKL_resume_CPU_receipt.json','label_coverage_CPU_receipt.json','prepare_CPU_receipt.json','online_donors.json']:
  files[str(ROOT/n)]=sha(ROOT/n)
 for n in ['runtime_freeze.json','persistent_queue.json']:files[str(V3/n)]=sha(V3/n)
 for p,h in files.items():require(sha(p)==h,'source/model/data changed: '+p)
 admission=dict(status='CPU_pass_GPU_smoke_pending',E2E_sha256=sha(ROOT/'CPU_E2E_receipt.json'),queue_regression_sha256=sha(ROOT/'queue_CPU_receipt.json'),FKL_regression_sha256=sha(ROOT/'FKL_resume_CPU_receipt.json'),actual_label_mapping_sha256=sha(ROOT/'label_coverage_CPU_receipt.json'),
  E2E_scope='real production engines and real cached heads with small CPU budgets and synthetic Target; not real-model GPU acceptance',conditions=20,independent300_used=False)
 immutable(ROOT/'CPU_admission.json',admission);files[str(ROOT/'CPU_admission.json')]=sha(ROOT/'CPU_admission.json')
 immutable(ROOT/'runtime_freeze.json',dict(revision=4,program_sha256=sha(MATRIX/'program.md'),groups=H.GROUPS,files=files,parent_root=str(V3),parent_freeze_sha256=sha(V3/'runtime_freeze.json'),parent_queue_unchanged=True,independent300_used=False))
 jobs=stages();immutable(ROOT/'persistent_queue.json',dict(revision=4,stages=jobs,stage_count=59,required_parent=str(V3),parent_scope='all existing v3 stages retain their order; this queue contains only remaining20',
  nonFKL_before_full_vocab=True,interventions_use_verified_weighted_checkpoint=True,no_runtime_cap=True,independent300_used=False))
 if not (ROOT/'parent_preservation_receipt.json').exists():
  immutable(ROOT/'parent_preservation_receipt.json',dict(parent_identity=parent,live_source_sha256=sha(V3/'supervisor_final.py'),queue_sha256=sha(V3/'persistent_queue.json'),runtime_freeze_sha256=sha(V3/'runtime_freeze.json'),parent_files_written=[],parent_processes_interrupted=[],GPU_lease_changes=[],unix=time.time()))
 argv=[sys.executable,'-B','-u',str(ROOT/'queue20.py')];existing=live_workers(argv);require(len(existing)<=1,'duplicate remaining20 successor')
 if existing:owner=existing[0];launched=False
 else:
  require(not (ROOT/'supervisor_first_error.json').exists(),'previous successor failure needs diagnosis')
  if (ROOT/'supervisor_identity.json').exists():require(not alive(read(ROOT/'supervisor_identity.json')),'different live owner exists')
  with (ROOT/'supervisor.log').open('ab') as log:child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
  owner=identity(child.pid);require(owner is not None,'new successor identity absent');launched=True
 reg=PARENT/'followup_registry'/('matrix-'+sha(MATRIX/'program.md')+'-remaining20-v4.json')
 if reg.exists():require(read(reg)['root']==str(ROOT),'remaining20 registration belongs to another root')
 declaration=dict(scope='remaining20_only',root=str(ROOT),entry=str(ROOT/'queue20.py'),owner=owner,parent_root=str(V3),parent_pid=parent['pid'],runtime_freeze_sha256=sha(ROOT/'runtime_freeze.json'),queue_sha256=sha(ROOT/'persistent_queue.json'),groups=H.GROUPS,stage_count=59,initial_state='waiting_v3_normal_terminal',new_GPU_dispatches=0)
 write(reg,declaration);write(MATRIX/'remaining20_successor.json',dict(declaration,registration_path=str(reg),registration_sha256=sha(reg)))
 write(ROOT/'launch_receipt.json',dict(status='registered_new_disjoint_waiting_successor' if launched else 'adopted_existing_remaining20_successor',owner=owner,parent_owner=parent,argv=argv,registration_path=str(reg),registration_sha256=sha(reg),queue_sha256=sha(ROOT/'persistent_queue.json'),runtime_freeze_sha256=sha(ROOT/'runtime_freeze.json'),parent_unchanged=True,GPU_dispatches=0,unix=time.time()))
 print(json.dumps(read(ROOT/'launch_receipt.json')),flush=True)
if __name__=='__main__':
 import heads as H
 try:main()
 except Exception as e:first_error(ROOT/'seal_first_error.json',e);raise
