"""Real checkpoint bytes, scoped failures, child termination and mutex regressions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import fcntl,json,pathlib,subprocess,sys,tempfile,time
import torch
from guards import *
from common import write,first_error
def main():
 torch.set_num_threads(1);assert not torch.cuda.is_available();tests=[]
 def case(name,fn):fn();tests.append(name)
 def rejects(fn):
  try:fn()
  except SharedIntegrityError:return
  raise AssertionError('unsafe case was accepted')
 with tempfile.TemporaryDirectory() as td:
  root=pathlib.Path(td)/'root';matrix=pathlib.Path(td)/'matrix';train=root/'runs/G/train';train.mkdir(parents=True);(matrix/'configs').mkdir(parents=True)
  cfg=matrix/'configs/G.json';write(cfg,{'group':'G'});write(root/'runtime_freeze.json',{'files':{}})
  payload=dict(group='G',epoch=5,position=0,updates=783440,config_sha256=sha(cfg),runtime_freeze_sha256=sha(root/'runtime_freeze.json'),model={'weight':torch.tensor([1.])})
  checkpoint=train/'epoch5.pt';torch.save(payload,checkpoint);original=checkpoint.read_bytes()
  selected={'main':{'path':str(checkpoint),'sha256':sha(checkpoint),'rule':'original_epoch5'}}
  terminal=dict(status='complete',group='G',updates=783440,config_sha256=sha(cfg),runtime_freeze_sha256=sha(root/'runtime_freeze.json'),selected=selected)
  write(train/'selected.json',selected);write(train/'completion.json',terminal)
  binding,state=training_binding(root,matrix,'G','main');assert state['weight'].item()==1;tests.append('actual_checkpoint_initial_acceptance')
  case('intact_resume_acceptance',lambda:training_binding(root,matrix,'G','main',binding))
  torch.save(dict(payload,model={'weight':torch.tensor([2.])}),checkpoint)
  case('same_filename_replacement_rejected',lambda:rejects(lambda:training_binding(root,matrix,'G','main',binding)));checkpoint.write_bytes(original)
  wrong={'main':dict(selected['main'],sha256='0'*64)};write(train/'selected.json',wrong);write(train/'completion.json',dict(terminal,selected=wrong))
  case('wrong_expected_SHA_rejected',lambda:rejects(lambda:training_binding(root,matrix,'G','main')))
  write(train/'selected.json',selected);write(train/'completion.json',terminal)
  write(cfg,{'group':'OTHER'});case('changed_group_config_rejected',lambda:rejects(lambda:training_binding(root,matrix,'G','main')));write(cfg,{'group':'G'})
  for field,value in [('status','running'),('updates',123),('group','WRONG')]:
   write(train/'completion.json',dict(terminal,**{field:value}));case('wrong_training_'+field+'_rejected',lambda:rejects(lambda:training_binding(root,matrix,'G','main')))
  write(train/'completion.json',terminal)
  torch.save(dict(payload,group='WRONG'),checkpoint);modified={'main':dict(selected['main'],sha256=sha(checkpoint))};write(train/'selected.json',modified);write(train/'completion.json',dict(terminal,selected=modified))
  case('checkpoint_internal_group_rejected',lambda:rejects(lambda:training_binding(root,matrix,'G','main')))
  assert category(SharedIntegrityError())=='shared_integrity' and category(FKLSpecificError())=='fkl_specific' and category(ValueError())=='unclassified_blocking';tests.append('typed_exception_classification')
  j={'optional_fkl':True};err={'category':'fkl_specific'}
  assert terminal_decision(j,None,err,False,True,True,1)=='blocked_fkl_continue_ce';tests.append('FKL_nonzero_clean_exit_continues_CE')
  for live,released,free in [(True,True,True),(False,False,True),(False,True,False)]:assert terminal_decision(j,None,err,live,released,free,1)=='wait_cleanup'
  tests.append('live_child_and_unreleased_or_held_GPU_lease_deny_continuation')
  for error in [{'category':'shared_integrity'},{'category':'unclassified_blocking'},None]:assert terminal_decision(j,None,error,False,True,True,1)=='block_queue'
  assert terminal_decision({},None,err,False,True,True,1)=='block_queue';tests.append('shared_unknown_and_nonFKL_errors_block_queue')
  path=root/'first_error.json';first_error(path,FKLSpecificError('first'));before=sha(path);first_error(path,FKLSpecificError('second'));assert sha(path)==before;tests.append('first_error_preserved')
  assert dispatch_decision([{'pid':1}],False,False,{'status':'launching'})=='adopt'
  assert dispatch_decision([],True,False,{'status':'running'})=='terminal'
  assert dispatch_decision([],False,False,{'status':'running'})=='block'
  rejects(lambda:dispatch_decision([{},{}],False,False,None));tests.append('crash_window_adoption_terminal_race_and_duplicate_refusal')
  lock=root/'mutex';holder=open(lock,'a+');fcntl.flock(holder,fcntl.LOCK_EX|fcntl.LOCK_NB)
  script="import fcntl,sys;f=open(sys.argv[1],'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
  denied=subprocess.run([sys.executable,'-c',script,str(lock)],capture_output=True);assert denied.returncode!=0
  holder.close();allowed=subprocess.run([sys.executable,'-c',script,str(lock)],capture_output=True);assert allowed.returncode==0;tests.append('independent_process_mutex_prevents_double_dispatch')
  child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);assert child.poll() is None
  assert terminal_decision(j,None,err,child.poll() is None,True,True)=='wait_cleanup';child.terminate();child.wait(timeout=10)
  assert terminal_decision(j,None,err,child.poll() is None,True,True)=='blocked_fkl_continue_ce';tests.append('real_child_must_exit_before_CE_continue')
 result=dict(status='pass',tests=tests,test_count=len(tests),GPU_used=False,source_sha256={p.name:sha(p) for p in pathlib.Path(__file__).parent.glob('*.py')},unix=time.time())
 write(pathlib.Path(__file__).parent/'audit_final_CPU_receipt.json',result);print(json.dumps({'status':'pass','tests':tests}))
if __name__=='__main__':main()
