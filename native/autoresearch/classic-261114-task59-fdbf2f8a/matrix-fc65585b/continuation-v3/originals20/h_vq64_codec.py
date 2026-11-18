"""Causal 8-raw-byte VQ message for the conditioned hidden experiment.

The encoder accepts only current valid-prefix Target hidden and already known
protocol features. Future Target labels are absent from this API.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

WIRE_BYTES = 8
CHOICES = 256
CODE_DIM = 16
EDGE_CONTROL_DIM = WIRE_BYTES * CODE_DIM


class ConditionedVQ64(nn.Module):
    def __init__(self, *, target_hidden_size: int, draft_hidden_size: int,
                 conditioned: bool = True) -> None:
        super().__init__()
        if target_hidden_size <= 0 or draft_hidden_size <= 0:
            raise ValueError("hidden sizes must be positive")
        self.target_hidden_size = target_hidden_size
        self.draft_hidden_size = draft_hidden_size
        self.conditioned = conditioned
        input_dim = target_hidden_size + 2 * draft_hidden_size + 1
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, EDGE_CONTROL_DIM), nn.SiLU(),
            nn.Linear(EDGE_CONTROL_DIM, WIRE_BYTES * CHOICES),
        )
        self.codebook = nn.Parameter(
            torch.randn(WIRE_BYTES, CHOICES, CODE_DIM) * .02)

    def _inputs(self, target_hidden: Tensor, accepted_count: int,
                rejected_embedding: Tensor | None,
                correction_embedding: Tensor) -> Tensor:
        if (target_hidden.shape != (1, self.target_hidden_size)
                or not target_hidden.is_floating_point()
                or not torch.isfinite(target_hidden).all()):
            raise ValueError("current Target hidden has wrong shape or value")
        if (type(accepted_count) is not int
                or not 0 <= accepted_count <= WIRE_BYTES):
            raise ValueError("accepted_count is outside ordinary protocol")
        if (correction_embedding.shape != (1, self.draft_hidden_size)
                or not correction_embedding.is_floating_point()
                or not torch.isfinite(correction_embedding).all()):
            raise ValueError("correction embedding has wrong shape or value")
        if (rejected_embedding is not None
                and (rejected_embedding.shape != (1, self.draft_hidden_size)
                     or not rejected_embedding.is_floating_point()
                     or not torch.isfinite(rejected_embedding).all())):
            raise ValueError("rejected embedding has wrong shape or value")
        if (rejected_embedding is None) != (accepted_count == WIRE_BYTES):
            raise ValueError("full-accept bonus and rejected-token fields differ")
        device = self.codebook.device
        dtype = self.codebook.dtype
        target = target_hidden.detach().to(device=device, dtype=dtype)
        if self.conditioned:
            rejected = (torch.zeros(1, self.draft_hidden_size, device=device,
                                    dtype=dtype) if rejected_embedding is None
                        else rejected_embedding.detach().to(device=device,
                                                             dtype=dtype))
            correction = correction_embedding.detach().to(device=device,
                                                           dtype=dtype)
            accepted = torch.tensor([[accepted_count / WIRE_BYTES]],
                                    device=device, dtype=dtype)
        else:
            rejected = torch.zeros(1, self.draft_hidden_size, device=device,
                                   dtype=dtype)
            correction = torch.zeros_like(rejected)
            accepted = torch.zeros(1, 1, device=device, dtype=dtype)
        return torch.cat((target, rejected, correction, accepted), dim=-1)

    def logits(self, target_hidden: Tensor, accepted_count: int,
               rejected_embedding: Tensor | None,
               correction_embedding: Tensor) -> Tensor:
        x = self._inputs(target_hidden, accepted_count,
                         rejected_embedding, correction_embedding)
        logits = self.encoder(x).reshape(WIRE_BYTES, CHOICES)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite message logits")
        return logits

    @staticmethod
    def _wire(indices: Tensor) -> bytes:
        if indices.shape != (WIRE_BYTES,) or indices.dtype != torch.long:
            raise TypeError("VQ64 indices must be eight int64 categorical IDs")
        values = indices.detach().to(device="cpu").tolist()
        if any(type(value) is not int or not 0 <= value < CHOICES
               for value in values):
            raise ValueError("VQ64 index out of byte range")
        payload = bytes(values)
        if type(payload) is not bytes or len(payload) != WIRE_BYTES:
            raise AssertionError("VQ64 physical payload is not eight raw bytes")
        return payload

    @torch.no_grad()
    def encode_payload(self, target_hidden: Tensor, accepted_count: int,
                       rejected_embedding: Tensor | None,
                       correction_embedding: Tensor) -> bytes:
        indices = self.logits(target_hidden, accepted_count,
                              rejected_embedding, correction_embedding).argmax(-1)
        return self._wire(indices)

    def decode_payload(self, payload: bytes | None) -> Tensor:
        if payload is None:
            return torch.zeros(1, EDGE_CONTROL_DIM,
                               device=self.codebook.device,
                               dtype=self.codebook.dtype)
        if type(payload) is not bytes:
            raise TypeError("additional message must be raw bytes or None")
        if len(payload) != WIRE_BYTES:
            raise ValueError("additional message must be exactly eight bytes")
        indices = torch.tensor(list(payload), device=self.codebook.device,
                               dtype=torch.long)
        rows = torch.arange(WIRE_BYTES, device=self.codebook.device)
        return self.codebook[rows, indices].reshape(1, EDGE_CONTROL_DIM)

    def train_control(self, target_hidden: Tensor, accepted_count: int,
                      rejected_embedding: Tensor | None,
                      correction_embedding: Tensor,
                      *, temperature: float = 1.0) -> Tensor:
        if not 0 < temperature <= 10:
            raise ValueError("invalid straight-through temperature")
        logits = self.logits(target_hidden, accepted_count,
                             rejected_embedding, correction_embedding)
        soft = F.softmax(logits / temperature, dim=-1)
        hard = F.one_hot(logits.argmax(-1), CHOICES).to(dtype=soft.dtype)
        straight_through = hard - soft.detach() + soft
        control = torch.einsum("gc,gcd->gd", straight_through,
                               self.codebook).reshape(1, EDGE_CONTROL_DIM)
        if not torch.isfinite(control).all():
            raise FloatingPointError("nonfinite decoded message control")
        return control
