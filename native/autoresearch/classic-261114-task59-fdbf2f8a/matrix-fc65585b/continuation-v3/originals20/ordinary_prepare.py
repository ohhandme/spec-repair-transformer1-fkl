"""Exact audited short-full ordinary-C preparation; no legacy runtime imports."""
import torch
from feedback_codec.v10_four_arm import V10PreparedState

def _base_prepare(head, obs, draft):
    if obs.rejected_token is not None or obs.accepted_count == 8:
        return head.base.prepare_observation(obs, draft)
    if obs.rejected_token is None and obs.accepted_count == len(obs.proposals) and (0 <= obs.accepted_count < 8):
        if obs.target_hidden_prefix != obs.valid_prefix or obs.valid_prefix != obs.source_prefix + obs.proposals or obs.corrected_prefix != obs.valid_prefix + (obs.correction_token,) or (obs.remaining_budget <= 0):
            raise ValueError('invalid short-full causal boundary')
        table = draft.model.get_input_embeddings()
        if table is None or any((p.requires_grad for p in table.parameters())):
            raise ValueError('Draft table not frozen')
        if head.base._draft_embedding_ref is not None and head.base._draft_embedding_ref() is not table:
            raise ValueError('Draft table binding differs')
        local = draft.teacher_hidden(list(obs.corrected_prefix), [0]).detach().clone()
        if local.shape != (1, head.config.draft_hidden_size) or not bool(torch.isfinite(local).all()):
            raise ValueError('local hidden invalid')
        correction = head.base._protocol_embedding(table, obs.correction_token)
        return V10PreparedState('C', obs.corrected_prefix, local, None, correction, obs.accepted_count, None, None, None)
    raise ValueError('offered/accepted short-full mismatch')
