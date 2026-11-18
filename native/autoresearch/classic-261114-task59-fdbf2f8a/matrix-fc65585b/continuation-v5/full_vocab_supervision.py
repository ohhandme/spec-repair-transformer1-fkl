import functools
from common import *
from fkl_integrity import coverage_inputs,validate_manifest,lines as iterlines
class FullVocabularySupervision:
 def __init__(self,head,head_binding):
  require(tuple(head.weight.shape)==(151936,4096) and not head.weight.requires_grad,'frozen Target output head dimensions differ')
  require(sha(head_binding['source_path'])==head_binding['source_sha256'],'Target output shard SHA differs')
  require(read(ROOT/'runtime_freeze.json')['files'].get(head_binding['source_path'])==head_binding['source_sha256'],'Target output shard not frozen')
  self.head=head;jobs,self.coverage=coverage_inputs(V3/'labels');done=ROOT/'fkl/backfill/completion.json'
  require(done.exists(),'full vocabulary backfill not accepted');terminal=read(done)
  require(terminal.get('FKL_training_admitted') is True and terminal['all_label_mapping_sha256']==self.coverage['mapping_receipt_sha256'],'full vocabulary admission differs')
  accepted=validate_manifest(ROOT/'fkl/backfill/manifest.jsonl',jobs,complete=True,terminal=terminal);self.index={}
  for e in accepted['entries']:
   for row in iterlines(e['metadata_path']):self.index[row['prefix_id']]=(e['path'],e['sha256'],row['expected_id'])
  self.windows={}
  for row in iterlines(V3/'labels/all_occurrences.jsonl'):
   self.windows.setdefault((row['request_id'],row['window_index']),[]).append((row['prefix_id'],row['expected_id']))
 @functools.lru_cache(maxsize=4)
 def shard(self,path,digest):
  from safetensors.torch import load_file
  require(sha(path)==digest,'accepted FKL data changed');return load_file(path,device='cpu')
 def logits(self,request,window,labels):
  import torch
  rows=self.windows[(request,window)];require([x[1] for x in rows]==labels.tolist(),'FKL teacher labels differ from sample')
  with torch.inference_mode():
   values=[]
   for key,label in rows:
    path,digest,_=self.index[key];h=self.shard(path,digest)[key].to(self.head.weight.device);z=self.head(h.reshape(1,-1))[0].float()
    require(int(z.argmax())==label,'canonical Target greedy reconstruction changed');values.append(z)
   return torch.stack(values).unsqueeze(0)
