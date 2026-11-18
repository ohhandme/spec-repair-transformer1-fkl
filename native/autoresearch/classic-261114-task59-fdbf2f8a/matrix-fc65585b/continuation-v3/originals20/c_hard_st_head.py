"""C head with deterministic hard local codes during training.

This candidate changes only the training-time local code selection. The
existing C head, parameters, initialization order, and argmax inference path
are inherited unchanged. The forward value is an argmax one-hot code; its
backward path is the ordinary softmax straight-through surrogate.
"""

from __future__ import annotations

import math
from types import MethodType

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from feedback_codec.v10_four_arm import V10FourArmHead


MODE = "C-HARD-ST"


def _local_hard_st(
    self,
    local_hidden: Tensor,
    rejected_embedding: Tensor | None = None,
    correction_embedding: Tensor | None = None,
    accepted_count: Tensor | int | None = None,
    *,
    temperature: float = 1.0,
) -> Tensor:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scores = self._local_scores(
        local_hidden, rejected_embedding, correction_embedding, accepted_count
    )
    soft = F.softmax(scores / temperature, dim=-1)
    hard = F.one_hot(scores.argmax(dim=-1), self.config.choices).to(soft.dtype)
    return hard + (soft - soft.detach())


class CHardSTHead(V10FourArmHead):
    """Matched C head, replacing only the local training sampler."""

    def __init__(self, config, output_head: nn.Module, *,
                 draft_embedding: nn.Module | None = None) -> None:
        super().__init__(config, output_head, mode="C",
                         draft_embedding=draft_embedding)
        if self.core.mode != "protocol_only":
            raise AssertionError("C local-code protocol mode changed")
        self.core.local_sample_st = MethodType(_local_hard_st, self.core)
