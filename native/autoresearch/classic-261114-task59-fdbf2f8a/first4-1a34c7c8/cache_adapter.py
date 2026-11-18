"""Read-only RAW5000 cache adapter shared by all four first-wave heads."""
from functools import lru_cache
from pathlib import Path
import torch
from safetensors.torch import load_file

class FrozenWindows:
    def __init__(self, records, cache_root, split):
        if split not in ('train','calibration'):
            raise ValueError('first-four training adapter excludes heldout/confirmation')
        self.records=[r for r in records if r['split']==split]
        self.root=Path(cache_root)
        self.windows=[(ri,wi) for ri,r in enumerate(self.records) for wi in range(len(r['windows']))]

    def __len__(self):return len(self.windows)

    @lru_cache(maxsize=256)
    def cache(self,request_index):
        return load_file(self.root/(self.records[request_index]['request_id']+'.safetensors'),device='cpu')

    def batch(self, indices):
        n=len(indices)
        x={'hidden':torch.zeros(n,8,1024,dtype=torch.bfloat16),
           'local':torch.empty(n,1024,dtype=torch.bfloat16),
           'target':torch.empty(n,4096,dtype=torch.bfloat16),
           'labels':torch.full((n,8),-100,dtype=torch.long),
           'accepted':torch.empty(n,dtype=torch.long),'rejected':torch.empty(n,dtype=torch.long),
           'correction':torch.empty(n,dtype=torch.long),'request':torch.empty(n,dtype=torch.long)}
        for j,idx in enumerate(indices):
            ri,wi=self.windows[int(idx)];r=self.records[ri];w=r['windows'][wi];z=self.cache(ri)
            off=w['offset'];length=w['length']
            assert 1<=length<=8 and len(w['labels'])==length
            x['hidden'][j,:length]=z['draft_sequence'][off:off+length]
            x['local'][j]=z['draft_sequence'][off];x['target'][j]=z['target'][wi]
            x['labels'][j,:length]=torch.tensor(w['labels']);x['accepted'][j]=w['accepted']
            x['rejected'][j]=0 if w['rejected'] is None else w['rejected']
            x['correction'][j]=w['correction'];x['request'][j]=ri
        x['valid']=x['labels']!=-100
        return x

def attach_embeddings(x, embedding, device):
    out={k:v.to(device) for k,v in x.items() if k!='request'}
    with torch.no_grad():
        out['rejected_embedding']=embedding(out['rejected'])
        out['correction_embedding']=embedding(out['correction'])
    return out

def logits(head,x,*,generator=None):
    """No labels/masks/future target IDs enter the control network.

    Generator is private to each training group. Gumbel-ST uses the same
    exponential-noise formula as torch F.gumbel_softmax, temperature=1.
    """
    batch=x['local'].shape[0]
    args=(x['local'],x['rejected_embedding'],x['correction_embedding'],x['accepted'])
    if generator is None:
        codes=head.core.local_encode(*args)
    else:
        scores=head.core._local_scores(*args)
        noise=-torch.empty_like(scores).exponential_(generator=generator).log()
        soft=(scores+noise).softmax(-1)
        hard=torch.zeros_like(soft).scatter_(-1,soft.argmax(-1,keepdim=True),1.)
        codes=hard-soft.detach()+soft
    control=head.core.repair._lookup(codes[:,0],batch)+head.core.repair._lookup(codes[:,1],batch)
    local=x['local'].to(head.local_hidden_lift[0].weight.dtype)
    control=control+head.hidden_projection(head.local_hidden_lift(local)+x['target'].to(local.dtype))
    control=control+head.top32_projection(head.local_top32_surrogate(local))
    return head.core.repair.repair_from_control(x['hidden'],control,torch.arange(1,9,device=local.device))
