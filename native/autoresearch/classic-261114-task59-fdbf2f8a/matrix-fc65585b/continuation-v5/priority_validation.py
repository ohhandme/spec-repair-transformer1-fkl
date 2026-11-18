"""Recheck original priority-ten actual checkpoint and every sealed request."""
from common import *
from guards import training_binding,initial_binding
def validate(job,root):
 out=pathlib.Path(job['out']);d=read(out/'completion.json');g=job['group'];phase=job['phase']
 require(d['status']=='complete' and d['group']==g,'priority terminal differs')
 if phase=='train':
  for selection in read(out/'selected.json'):training_binding(root,MATRIX,g,selection)
  return 'complete'
 roster=read(WAVE/'roster503.json');smoke=read(WAVE/'smoke_ids.json');roster=[r for r in roster if phase=='eval' or r['request_id'] in smoke]
 selections=['initial'] if phase=='smoke' else list(read(root/'runs'/g/'train/selected.json'));seen=set()
 for sel in selections:
  bound,_=initial_binding(root,MATRIX,g) if phase=='smoke' else training_binding(root,MATRIX,g,sel)
  if bound['checkpoint_sha256'] in seen:continue
  seen.add(bound['checkpoint_sha256'])
  for req in roster:
   dest=out/sel/req['request_id'];r=read(dest/'receipt.json')
   require(r['status']=='pass' and r['checkpoint_binding']==bound,'priority request checkpoint binding differs')
   require(r['rounds_sha256']==sha(dest/'rounds.jsonl'),'priority rounds SHA differs')
   require(r['output_ids']==req['frozen_target_ids'] and r['eos']==req['eos'] and r['truncated']==req['truncated'],'priority ID/EOS/truncation differs')
 return 'complete'
