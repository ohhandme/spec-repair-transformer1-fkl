"""Durable strict FKL commit: file receipts precede rename; resume admits or quarantines."""
import hashlib,json,os,pathlib,time,uuid
from common import write,read,sha,require,immutable
from fkl_integrity import validate_entry,validate_manifest,canonical

def sync_dir(path):
 fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def sync_file(path):
 with open(path,'rb') as f:os.fsync(f.fileno())

def exact_bytes(path,data):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():require(path.read_bytes()==data,'preserved recovery bytes differ')
 else:
  with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  sync_dir(path.parent)

def repair_tail(out):
 path=out/'manifest.jsonl'
 if not path.exists():return None
 raw=path.read_bytes();cut=raw.rfind(b'\n')+1
 if cut==len(raw):return None
 digest=hashlib.sha256(raw).hexdigest();folder=out/'recovery'/'manifest_tails'/digest
 exact_bytes(folder/'original_manifest.bin',raw);exact_bytes(folder/'tail.bin',raw[cut:])
 immutable(folder/'receipt.json',dict(original_sha256=digest,retained_bytes=cut,tail_bytes=len(raw)-cut,tail_sha256=hashlib.sha256(raw[cut:]).hexdigest()))
 temp=path.with_name('manifest.repair.'+str(os.getpid()))
 with temp.open('wb') as f:f.write(raw[:cut]);f.flush();os.fsync(f.fileno())
 os.replace(temp,path);sync_dir(out)
 return dict(receipt_path=str(folder/'receipt.json'),receipt_sha256=sha(folder/'receipt.json'))

def strict_entry(dest,jobs,context,seen=None):
 dest=pathlib.Path(dest);require(not dest.is_symlink(),'symlink shard refused')
 receipt=dest/'seal.json';r=read(receipt)
 require(r['context']==context,'orphan/manifest frozen label or writer context differs')
 e=r['entry'];require(e['path']==str(dest/'hidden.safetensors') and e['metadata_path']==str(dest/'metadata.jsonl'),'shard path binding differs')
 for p in [dest/'hidden.safetensors',dest/'metadata.jsonl',receipt]:require(p.is_file() and not p.is_symlink(),'invalid shard member')
 ids=validate_entry(e,jobs,seen)
 require(bool(ids),'empty strict shard refused')
 require(r['prefix_ID_set_sha256']==hashlib.sha256(canonical(sorted(ids))).hexdigest(),'shard prefix ID receipt differs')
 return dict(e,seal_path=str(receipt),seal_sha256=sha(receipt)),ids

def append(out,entry):
 with (out/'manifest.jsonl').open('ab') as f:f.write(canonical(entry)+b'\n');f.flush();os.fsync(f.fileno())
 sync_dir(out)

def first_strict(out,entry,context,recovered):
 p=out/'first_strict_shard.json'
 if not p.exists():immutable(p,dict(entry=entry,context=context,recovered_orphan=recovered,unix=time.time()))

def quarantine(out,dest,reason):
 require(dest.parent.resolve()==(out/'shards').resolve(),'quarantine source outside shard root')
 inventory={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file() and not p.is_symlink()}
 tag=dest.name+'-'+hashlib.sha256(canonical(inventory)).hexdigest()[:16]
 target=out/'quarantine'/tag;target.parent.mkdir(parents=True,exist_ok=True)
 require(not target.exists(),'quarantine destination already exists')
 receipt=out/'recovery'/'quarantine_intents'/(tag+'.json')
 immutable(receipt,dict(source=str(dest),destination=str(target),reason=reason,files=inventory))
 sync_dir(receipt.parent);os.replace(dest,target);sync_dir(dest.parent);sync_dir(target.parent)
 return dict(path=str(target),receipt_path=str(receipt),receipt_sha256=sha(receipt),reason=reason)

def recover(out,jobs,context):
 from safetensors import SafetensorError
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);tail=repair_tail(out);manifest=out/'manifest.jsonl'
 accepted=validate_manifest(manifest,jobs);seen=set();referenced=set()
 for e in accepted['entries']:
  dest=pathlib.Path(e['path']).parent
  require(dest.parent.resolve()==(out/'shards').resolve(),'manifest shard is outside strict output root')
  actual,ids=strict_entry(dest,jobs,context,seen)
  require(actual==e,'manifest differs from strict seal receipt');seen.update(ids);referenced.add(dest)
 admitted=[];isolated=[]
 for dest in sorted((out/'shards').glob('*')):
  if dest in referenced:continue
  require(dest.is_dir() and not dest.is_symlink(),'unexpected object in strict shard directory')
  try:e,ids=strict_entry(dest,jobs,context,seen)
  except (KeyError,ValueError,OSError,RuntimeError,TypeError,SafetensorError) as exc:
   isolated.append(quarantine(out,dest,type(exc).__name__+': '+str(exc)));continue
  append(out,e);seen.update(ids);admitted.append(dict(path=str(dest),rows=len(ids),seal_sha256=e['seal_sha256']));first_strict(out,e,context,True)
 result=dict(seen=seen,admitted=admitted,quarantined=isolated,tail=tail)
 if tail or admitted or isolated:
  write(out/'recovery'/('pass-'+str(time.time_ns())+'.json'),dict(context=context,accepted_prefixes=len(seen),admitted=admitted,quarantined=isolated,tail=tail))
 return result

def publish(out,values,metadata,jobs,context,seen,after_rename=None):
 from safetensors.torch import save_file
 out=pathlib.Path(out);tag=str(time.time_ns())+'-'+uuid.uuid4().hex[:12]
 tmp=out/'incomplete'/tag;dest=out/'shards'/tag;tmp.mkdir(parents=True);dest.parent.mkdir(parents=True,exist_ok=True)
 save_file(values,str(tmp/'hidden.safetensors'));sync_file(tmp/'hidden.safetensors')
 exact_bytes(tmp/'metadata.jsonl',b''.join(canonical(r)+b'\n' for r in metadata))
 e=dict(path=str(dest/'hidden.safetensors'),sha256=sha(tmp/'hidden.safetensors'),metadata_path=str(dest/'metadata.jsonl'),metadata_sha256=sha(tmp/'metadata.jsonl'),rows=len(metadata))
 ids=validate_entry(dict(e,path=str(tmp/'hidden.safetensors'),metadata_path=str(tmp/'metadata.jsonl')),jobs,seen)
 write(tmp/'seal.json',dict(entry=e,context=context,prefix_ID_set_sha256=hashlib.sha256(canonical(sorted(ids))).hexdigest()))
 sync_dir(tmp);os.rename(tmp,dest);sync_dir(dest.parent);sync_dir(tmp.parent)
 if after_rename:after_rename(dest)
 entry,ids=strict_entry(dest,jobs,context,seen);append(out,entry);first_strict(out,entry,context,False)
 return ids
