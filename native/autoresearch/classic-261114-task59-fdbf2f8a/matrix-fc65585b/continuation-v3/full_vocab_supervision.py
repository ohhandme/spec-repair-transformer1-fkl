"""Full-vocabulary label loader, inaccessible until canonical all-prefix acceptance."""
import functools
from common import *
class FullVocabularySupervision:
 def __init__(self,target_output_head):
  import torch
  self.head=target_output_head
  require(all(not p.requires_grad for p in self.head.parameters()),'Target output head must be frozen')
  done=ROOT/'fkl/backfill/completion.json';require(done.exists(),'full-vocabulary canonical backfill not completed')
  d=read(done);require(d.get('status')=='complete' and d.get('FKL_training_admitted') is True and d.get('prefixes')==531692,'FKL full-prefix coverage not admitted')
  require(d.get('all_label_mapping_sha256')==sha(ROOT/'labels/completion.json'),'FKL label coverage receipt changed')
  manifest=ROOT/'fkl/backfill/manifest.jsonl';require(sha(manifest)==d['manifest_sha256'],'FKL manifest changed');self.prefixes={}
  for e in lines(manifest):
   require(sha(e['metadata_path'])==e['metadata_sha256'] and sha(e['path'])==e['sha256'],'FKL sealed shard changed')
   for row in lines(e['metadata_path']):
    key=row['prefix_id'];require(key not in self.prefixes,'duplicate canonical prefix')
    require(row['full_vocab_reconstruction_bitwise_equal'] and row['raw_top32_candidate_LSE_exact'],'unaccepted canonical prefix')
    self.prefixes[key]=(e['path'],e['sha256'],row['expected_id'])
  require(len(self.prefixes)==531692,'canonical prefix cardinality differs');self.windows={}
  coverage=read(ROOT/'labels/completion.json');e=coverage['files']['all_occurrences.jsonl'];require(sha(e['path'])==e['sha256'],'unified label map changed')
  count=0
  for row in lines(e['path']):
   key=row['prefix_id'];require(key in self.prefixes and self.prefixes[key][2]==row['expected_id'],'canonical prefix label mismatch')
   w=self.windows.setdefault((row['request_id'],row['window_index']),[]);require(row['label_position0']==len(w),'label occurrence order differs');w.append((key,row['expected_id']));count+=1
  require(count==1370920,'full vocabulary label count differs')
 @functools.lru_cache(maxsize=4)
 def shard(self,path,expected):
  from safetensors.torch import load_file
  require(sha(path)==expected,'FKL hidden cache changed');return load_file(path,device='cpu')
 def logits(self,request_id,window_index,labels):
  import torch
  rows=self.windows[(request_id,window_index)];require([x[1] for x in rows]==labels.tolist(),'teacher supervision labels differ')
  with torch.inference_mode():
   values=[]
   for key,label in rows:
    path,digest,_=self.prefixes[key];hidden=self.shard(path,digest)[key].to(self.head.weight.device)
    # Keep the proven one-row GPU lm_head operation; do not silently batch GEMM.
    z=self.head(hidden.reshape(1,-1))[0].float();require(int(z.argmax())==label,'Target output head binding or numeric path changed');values.append(z)
   return torch.stack(values).unsqueeze(0)
