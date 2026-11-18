"""Passive Qwen3-eager K/V observation on the unchanged full-prefix verifier.

The wrapper receives post-RoPE K/V actually passed to eager attention. It
returns the original attention result and never supplies/updates a cache.
Install only in an isolated smoke/experiment process, never a live run.
"""
import dataclasses,hashlib,json,pathlib
import torch

QWEN3_SHA='4b95c371fd26d40c69083dab36ac1eafd8cf82b415a0bb827275097c5ad2305b'
def digest(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()

@dataclasses.dataclass(frozen=True)
class ObservedStep:
    token_id:int
    hidden:torch.Tensor
    kv:dict
    prefix_sha256:str
    prefix_length:int
    call_id:int

class SameForwardKV:
    def __init__(self,target,*,layer_ids=(34,35),check_source=True):
        from transformers.models.qwen3 import modeling_qwen3 as module
        if check_source:
            assert hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest()==QWEN3_SHA
        assert target.model.config.model_type=='qwen3' and target.model.config._attn_implementation=='eager'
        self.target=target;self.module=module;self.layers=tuple(layer_ids)
        self.attention={id(target.model.model.layers[n].self_attn):n for n in layer_ids}
        self.original=module.eager_attention_forward
        assert not getattr(self.original,'_task59_observer',False),'observer already installed'
        self.active=False;self.rows={};self.calls=0;self.expected_length=None
        def observe(attention,query,key,value,mask,*args,**kwargs):
            if self.active and id(attention) in self.attention:
                layer=self.attention[id(attention)];n=self.expected_length
                assert layer not in self.rows and key.shape==value.shape
                assert tuple(key.shape)==(1,8,n,128) and key.dtype==value.dtype==torch.bfloat16
                # Pure copies of the real attention inputs, before the answer token.
                self.rows[layer]=(key[0,:,max(0,n-4):n].permute(1,0,2).detach().cpu().contiguous(),value[0,:,max(0,n-4):n].permute(1,0,2).detach().cpu().contiguous())
            return self.original(attention,query,key,value,mask,*args,**kwargs)
        observe._task59_observer=True
        self.wrapper=observe;module.eager_attention_forward=observe

    def next(self,prefix):
        assert not self.active,'no nested or concurrent observation'
        self.rows={};self.expected_length=len(prefix);self.active=True;self.calls+=1
        try:step=self.target.next(prefix)
        finally:self.active=False
        assert set(self.rows)==set(self.layers)
        n=len(prefix);positions=list(range(max(0,n-4),n));count=len(positions)
        kv={'K':torch.zeros(2,4,8,128,dtype=torch.bfloat16),'V':torch.zeros(2,4,8,128,dtype=torch.bfloat16),'valid':torch.tensor([True]*count+[False]*(4-count))}
        for i,layer in enumerate(self.layers):kv['K'][i,:count],kv['V'][i,:count]=self.rows[layer]
        return ObservedStep(step.token_id,step.hidden,kv,digest(prefix),n,self.calls)

    def close(self):
        assert self.module.eager_attention_forward is self.wrapper
        self.module.eager_attention_forward=self.original

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def extract_reference(cache,n):
    """Only the isolated smoke oracle may request a separate full cache."""
    out={'K':torch.zeros(2,4,8,128,dtype=torch.bfloat16),'V':torch.zeros(2,4,8,128,dtype=torch.bfloat16),'valid':torch.tensor([True]*min(4,n)+[False]*max(0,4-n))}
    for i,layer_id in enumerate((34,35)):
        if hasattr(cache,'layers'):key,value=cache.layers[layer_id].keys,cache.layers[layer_id].values
        elif hasattr(cache,'key_cache'):key,value=cache.key_cache[layer_id],cache.value_cache[layer_id]
        else:key,value=cache[layer_id]
        assert tuple(key.shape)==tuple(value.shape)==(1,8,n,128)
        for name,t in [('K',key),('V',value)]:out[name][i,:min(4,n)]=t[0,:,max(0,n-4):n].permute(1,0,2).detach().cpu().contiguous()
    return out

@torch.inference_mode()
def compare_reference_cache(target,prefix,step):
    assert digest(prefix)==step.prefix_sha256 and len(prefix)==step.prefix_length
    output=target.model(input_ids=target._ids(prefix),use_cache=True,output_hidden_states=True,return_dict=True)
    assert int(output.logits[0,-1].argmax())==step.token_id,'cache-enabled debug pass changes greedy'
    assert torch.equal(output.hidden_states[-1][0,-1],step.hidden),'cache-enabled hidden differs'
    ref=extract_reference(output.past_key_values,len(prefix))
    diffs={}
    for name in ['K','V','valid']:
        assert torch.equal(ref[name],step.kv[name]),'same-forward observation differs from actual cache: '+name
        if name!='valid':diffs[name]=float((ref[name].float()-step.kv[name].float()).abs().max())
    return dict(status='pass',K_max_abs=diffs['K'],V_max_abs=diffs['V'],bitwise_equal=True,reference_forward_for_audit_only=True)
