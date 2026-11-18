"""Restrict a new message to a verified current Target row and known protocol.

The future label object is never accepted by these functions. A missing first
round/current hidden, EOS, or no remaining budget returns no message (`None`).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from feedback_codec.fixed_probe import ProbeObservation
from h_vq64_codec import ConditionedVQ64, WIRE_BYTES


def validate_boundary(observation: ProbeObservation, *, eos: int) -> bool:
    if type(observation) is not ProbeObservation:
        raise TypeError("only frozen ProbeObservation may supply online operands")
    if (observation.window != WIRE_BYTES
            or type(observation.accepted_count) is not int
            or not 0 <= observation.accepted_count <= WIRE_BYTES
            or observation.accepted_count > len(observation.proposals)
            or observation.valid_prefix != observation.source_prefix
               + observation.proposals[:observation.accepted_count]
            or observation.target_hidden_prefix != observation.valid_prefix
            or observation.corrected_prefix != observation.valid_prefix
               + (observation.correction_token,)
            or (observation.rejected_token is None)
               != (observation.accepted_count == WIRE_BYTES)):
        raise ValueError("Target hidden/protocol boundary is not causal")
    if (observation.correction_token == eos
            or observation.remaining_budget <= 0
            or observation.target_hidden is None):
        return False
    if (not isinstance(observation.target_hidden, Tensor)
            or observation.target_hidden.ndim != 1
            or not observation.target_hidden.is_floating_point()
            or not torch.isfinite(observation.target_hidden).all()):
        raise ValueError("current Target hidden is invalid")
    return True


def _frozen_protocol_embedding(table: nn.Module, token: int | None,
                               *, hidden_size: int) -> Tensor | None:
    if token is None:
        return None
    weight = table.weight
    if (not isinstance(weight, Tensor) or weight.ndim != 2
            or weight.shape[1] != hidden_size or weight.requires_grad
            or type(token) is not int or not 0 <= token < weight.shape[0]):
        raise ValueError("known protocol token or frozen Draft table differs")
    return weight[token].detach().clone().reshape(1, -1)


def causal_inputs(observation: ProbeObservation, table: nn.Module, *,
                  target_hidden_size: int, draft_hidden_size: int,
                  eos: int) -> tuple[Tensor, int, Tensor | None, Tensor] | None:
    if not validate_boundary(observation, eos=eos):
        return None
    hidden = observation.target_hidden.detach().clone()
    if hidden.shape != (target_hidden_size,):
        raise ValueError("Target hidden width changed")
    rejected = _frozen_protocol_embedding(table, observation.rejected_token,
                                          hidden_size=draft_hidden_size)
    correction = _frozen_protocol_embedding(table, observation.correction_token,
                                            hidden_size=draft_hidden_size)
    assert correction is not None
    return hidden.reshape(1, -1), observation.accepted_count, rejected, correction


@torch.no_grad()
def encode_observation(codec: ConditionedVQ64,
                       observation: ProbeObservation | None,
                       table: nn.Module, *, eos: int) -> bytes | None:
    if observation is None:
        return None
    inputs = causal_inputs(observation, table,
                           target_hidden_size=codec.target_hidden_size,
                           draft_hidden_size=codec.draft_hidden_size, eos=eos)
    return None if inputs is None else codec.encode_payload(*inputs)
