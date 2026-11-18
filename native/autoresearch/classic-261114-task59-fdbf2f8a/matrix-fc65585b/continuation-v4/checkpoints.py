"""Immutable checkpoint blobs with atomic pointers and per-request training binding."""
import hashlib,io,json,os,pathlib
from common import read,write,sha,require
def save(out,payload,tag):
 import torch
 out=pathlib.Path(out);dest=out/'checkpoints';dest.mkdir(parents=True,exist_ok=True)
 tmp=dest/('pending-'+str(os.getpid())+'.pt');torch.save(payload,tmp);digest=sha(tmp)
 if tag=='latest':
  old=read(out/'latest.json') if (out/'latest.json').exists() else {}
  slot='resume-b.pt' if pathlib.Path(old.get('path','')).name=='resume-a.pt' else 'resume-a.pt'
  path=dest/slot;os.replace(tmp,path)
  pointer=dict(path=str(path),sha256=digest,epoch=payload['epoch'],position=payload['position'],updates=payload['updates']);write(out/'latest.json',pointer);return pointer
 path=dest/(str(payload['updates']).zfill(9)+'-'+digest+'.pt')
 if path.exists():require(sha(path)==digest,'checkpoint blob collision');tmp.unlink()
 else:os.replace(tmp,path)
 pointer=dict(path=str(path),sha256=digest,epoch=payload['epoch'],position=payload['position'],updates=payload['updates'])
 write(out/(tag+'.json'),pointer);return pointer
def load(pointer,group,config_sha,freeze_sha):
 import torch
 raw=pathlib.Path(pointer['path']).read_bytes();require(hashlib.sha256(raw).hexdigest()==pointer['sha256'],'actual checkpoint SHA differs')
 obj=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
 require(obj['group']==group and obj['config_sha256']==config_sha and obj['runtime_freeze_sha256']==freeze_sha,'checkpoint group/config/freeze differs')
 for k in ('epoch','position','updates'):
  if k in pointer:require(obj[k]==pointer[k],'checkpoint pointer position differs: '+k)
 return obj
def binding(root,group,selection,expected=None,initial=False):
 from safetensors.torch import load as load_safe
 root=pathlib.Path(root);cfgpath=root/'configs'/(group+'.json');cfg=read(cfgpath);owner=cfg['checkpoint_owner'];ownerpath=root/'configs'/(owner+'.json');owner_cfg=read(ownerpath)
 freeze_sha=sha(root/'runtime_freeze.json');train=root/'runs'/owner/'train'
 common=dict(group=group,checkpoint_owner=owner,selection=selection,group_config_sha256=sha(cfgpath),training_config_sha256=sha(ownerpath),runtime_freeze_sha256=freeze_sha)
 if initial:
  require(not cfg['evaluation_only'],'intervention cannot use an untrained initialization')
  raw=pathlib.Path(cfg['initial_path']).read_bytes();digest=hashlib.sha256(raw).hexdigest();require(digest==cfg['initial_sha256'],'actual initialization SHA differs')
  result=dict(common,checkpoint_sha256=digest,training_terminal='initial_only');state=load_safe(raw)
 else:
  terminal=read(train/'completion.json');selected=read(train/'selected.json')
  require(terminal.get('status')=='complete' and terminal['group']==owner and terminal['config_sha256']==sha(ownerpath) and terminal['runtime_freeze_sha256']==freeze_sha,'training terminal/config/runtime differs')
  require(terminal['updates']==owner_cfg['updates'] and terminal['selected']==selected,'training budget or selection terminal differs')
  pointer=selected[selection];require(pathlib.Path(pointer['path']).resolve().parent==(train/'checkpoints').resolve(),'checkpoint path outside owner training directory')
  obj=load(pointer,owner,sha(ownerpath),freeze_sha)
  require(obj['position']==0 and obj['updates']==obj['epoch']*owner_cfg['training_windows'],'selected checkpoint not at complete epoch boundary')
  require(1<=obj['epoch']<=owner_cfg['epochs'],'selected epoch outside frozen budget')
  if selection=='main':require(obj['epoch']==owner_cfg['epochs'],'main is not frozen final epoch')
  result=dict(common,checkpoint_sha256=pointer['sha256'],checkpoint_path=pointer['path'],epoch=obj['epoch'],training_completion_sha256=sha(train/'completion.json'),selected_sha256=sha(train/'selected.json'));state=obj['model']
 if expected is not None:require(result==expected,'checkpoint binding changed during resume/request')
 return result,state
