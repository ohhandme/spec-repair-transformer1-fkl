"""CPU-adaptable original B/C/TOP32 and Task28 C-only residuals.

No future labels, Target calls or training partitions are read by this module.
The caller supplies the audited current observation and causal Draft cache.
"""
import torch
from feedback_codec.codec import CodecConfig
from feedback_codec.v10_four_arm import V10FourArmHead, V10PreparedState, semantic_current_top32
from feedback_codec.decoding import TargetLogitSummary
from repair_reference import COnlyHead
from arch_head import Architecture, CausalRepairStack

PRIORITY = ['B', 'C', 'C-weighted', 'MLP-C-CE-I0', 'TR1-C-CE-I0',
            'TR1-W128-C-CE', 'TR2-W64-C-CE', 'TR2-W128-C-CE', 'TR3-W64-C-CE', 'TOP32_v2']
STRUCTURES = {'TR1-W128-C-CE': (1,128), 'TR2-W64-C-CE': (2,64),
              'TR2-W128-C-CE': (2,128), 'TR3-W64-C-CE': (3,64)}

def build(group, output, embedding):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        mode = 'B' if group == 'B' else 'F-CURRENT-TOP32' if group == 'TOP32_v2' else 'C'
        base = V10FourArmHead(CodecConfig(4096,1024,151936), output,
                              mode=mode, draft_embedding=embedding)
        if group in ('B','C','C-weighted','TOP32_v2'):
            return base
        head = COnlyHead(base, 'mlp' if group.startswith('MLP') else 'transformer')
        if group in STRUCTURES:
            head.adapter = CausalRepairStack(Architecture(*STRUCTURES[group]))
        return head

def prepare_cached(group, record, window, trace, tensors, cache, embedding):
    """Exactly one sealed observation; Top32 uses its final *used* Target row."""
    rr = trace['rounds'][window['round']]
    assert trace['trajectory_policy'] == 'RAW' and trace['repair_enabled'] is False
    assert rr['observation_key'] == window['observation_key']
    assert rr['accepted_count'] == window['accepted']
    assert rr['correction_id'] == window['correction'] or rr['bonus_id'] == window['correction']
    assert rr['rejected_id'] == window['rejected']
    off = window['offset']
    prefix = tuple(record['prompt_ids'] + record['output_ids'][:off])
    assert len(prefix) == rr['prefix_after_commit_length'] and prefix[-1] == window['correction']
    local = cache['draft_sequence'][off].reshape(1,1024)
    mode = 'B' if group == 'B' else 'F-CURRENT-TOP32' if group == 'TOP32_v2' else 'C'
    rejected = correction = accepted = feature = accounting = None
    if mode != 'B':
        accepted = window['accepted']
        correction = embedding.weight[window['correction']].detach().reshape(1,-1)
        if window['rejected'] is not None:
            rejected = embedding.weight[window['rejected']].detach().reshape(1,-1)
        else:
            assert accepted == len(rr['candidate_ids'])
    if mode == 'F-CURRENT-TOP32':
        key = rr['tensor_prefix']; j = rr['top32']['used_rows'] - 1
        assert j == len(rr['target_decision_ids'])-1
        assert rr['top32']['absolute_source_positions'][j] == len(prefix)-2
        assert rr['top32']['greedy_ids'][j] == window['correction']
        ids = tensors[key+'.top32_ids'][j].tolist()
        values = tensors[key+'.top32_logits'][j].float()
        # A100 stores BF16. Preserve those saved numeric values explicitly;
        # conversion to FP16 is not claimed to restore historical FP16 rounding.
        assert torch.equal(values.to(torch.float16).float(), values)
        assert ids[0] == window['correction']
        summary = TargetLogitSummary.from_json(dict(greedy_token_id=ids[0],
            top_token_ids=ids, top_logits_fp16=values.tolist(),
            logsumexp_fp32=float(tensors[key+'.logsumexp'][j]),
            top2_margin_fp32=float(values[0]-values[1]), vocab_size=151936))
        accounting = semantic_current_top32(summary, embedding)
        feature = accounting.features
    return V10PreparedState(mode,prefix,local,rejected,correction,accepted,None,feature,accounting)

def forward(head, history, state, temperature=None):
    if isinstance(head,COnlyHead):
        return head.forward_logits(history,state,temperature=temperature)
    # Historical V10 lacks the task23 short-full fallback. Do not silently
    # invent a rejected ID. CPU admission records and excludes this case until
    # its explicit original fallback is bound for B/C/TOP32 evaluation.
    if state.mode != 'B' and state.rejected_embedding is None and state.accepted_count < 8:
        raise ValueError('original_V10_short_full_requires_explicit_adapter')
    return head.forward_logits(history,state,torch.arange(1,history.shape[1]+1),temperature=temperature)
