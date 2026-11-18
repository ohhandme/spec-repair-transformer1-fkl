"""Fail-closed integrity and queue decisions; independent of CUDA and controller state."""
import hashlib,io,json,pathlib
class SharedIntegrityError(RuntimeError): pass
class FKLSpecificError(RuntimeError): pass
def require(ok,message):
 if not ok:raise SharedIntegrityError(message)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def category(exc):
 if isinstance(exc,SharedIntegrityError):return 'shared_integrity'
 if isinstance(exc,FKLSpecificError):return 'fkl_specific'
 return 'unclassified_blocking'
def training_binding(root,matrix,group,selection,expected=None):
 """Validate all actual bytes before loading or binding any individual request."""
 import torch
 root=pathlib.Path(root);matrix=pathlib.Path(matrix);train=root/'runs'/group/'train'
 config=matrix/'configs'/(group+'.json');cfg=read(config);cfgsha=sha(config)
 freeze_sha=sha(root/'runtime_freeze.json');terminal=read(train/'completion.json');selected=read(train/'selected.json')
 require(terminal.get('status')=='complete','training is not normally complete')
 require(terminal.get('group')==group,'training group differs')
 require(terminal.get('updates')==783440,'training budget incomplete')
 require(terminal.get('config_sha256')==cfgsha,'training config changed')
 require(terminal.get('runtime_freeze_sha256')==freeze_sha,'training runtime changed')
 require(terminal.get('selected')==selected,'selected checkpoint not bound to training terminal')
 require(selection in selected,'unknown selection')
 chosen=selected[selection];path=pathlib.Path(chosen['path']).resolve()
 require(path.parent==train.resolve(),'checkpoint is outside its training run')
 raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 require(digest==chosen['sha256'],'actual checkpoint SHA differs from selected SHA')
 payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
 require(payload.get('group')==group,'checkpoint internal group differs')
 require(payload.get('config_sha256')==cfgsha,'checkpoint internal config differs')
 require(payload.get('runtime_freeze_sha256')==freeze_sha,'checkpoint internal runtime differs')
 epoch=5 if selection=='main' else chosen['epoch']
 require(payload.get('epoch')==epoch and payload.get('position')==0 and payload.get('updates')==epoch*156688,'checkpoint epoch/update boundary differs')
 binding={'group':group,'selection':selection,'checkpoint_sha256':digest,'checkpoint_path':str(path),
  'config_sha256':cfgsha,'runtime_freeze_sha256':freeze_sha,'training_completion_sha256':sha(train/'completion.json'),
  'selected_sha256':sha(train/'selected.json'),'epoch':epoch}
 if expected is not None:require(binding==expected,'checkpoint binding changed during evaluation/resume')
 return binding,payload['model']
def initial_binding(root,matrix,group,expected=None):
 from safetensors.torch import load
 root=pathlib.Path(root);matrix=pathlib.Path(matrix);cfg=matrix/'configs'/(group+'.json')
 path=matrix/'initial'/(group+'.safetensors');raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 require(digest==read(cfg)['initial_sha256'],'initial checkpoint actual SHA differs')
 b={'group':group,'selection':'initial','checkpoint_sha256':digest,'config_sha256':sha(cfg),'runtime_freeze_sha256':sha(root/'runtime_freeze.json')}
 if expected is not None:require(b==expected,'initial binding changed')
 return b,load(raw)
def terminal_decision(job,terminal,error,child_alive,lease_released,kernel_lease_free,rc=None):
 if child_alive or not lease_released or not kernel_lease_free:return 'wait_cleanup'
 if terminal and terminal.get('status')=='complete' and rc in (None,0) and not error:return 'complete'
 if job.get('optional_fkl') and error and error.get('category')=='fkl_specific':return 'blocked_fkl_continue_ce'
 return 'block_queue'
def dispatch_decision(live,terminal,error,old):
 require(len(live)<=1,'duplicate workers; refuse dispatch')
 if live:return 'adopt'
 if terminal or error:return 'terminal'
 if old and old.get('status') not in ('launching',):return 'block'
 return 'launch'
