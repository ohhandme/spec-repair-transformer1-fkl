"""Matched C repair with one strict, fixed RANK64-MARGIN message.

The edge sees only eight raw bytes, its own Draft state and protocol b_t.
The fixed Target Top-K encoder and the precomputed causal cache are external.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from feedback_codec.v10_four_arm import V10FourArmHead, V10PreparedState
from feedback_codec.fixed_probe import ProbeObservation
from rank64_consumer import consume_rank64

MODE = "RANK64-MARGIN"
CODE_DIM = 16
CONTROL_DIM = 128
VOCAB = 151936
MARGIN_CLIP = 26.0


@dataclass(frozen=True)
class RankState:
    base: V10PreparedState
    payload: bytes | None
    corrected_prefix: tuple[int, ...]


class Rank64Head(nn.Module):
    def __init__(self, config, output_head: nn.Module, *,
                 draft_embedding: nn.Module, payloads: dict[tuple[str, int], bytes]):
        super().__init__()
        if (config.window, config.rank, config.groups,
                config.choices, config.code_dim, config.vocab_size) != (
                8, 32, 8, 256, 16, VOCAB):
            raise ValueError("frozen C rank/8-byte capacity or Target vocabulary differs")
        self.config = config
        self.mode = MODE
        self.base = V10FourArmHead(config, output_head, mode="C",
                                   draft_embedding=draft_embedding)
        self.edge_codebook = nn.Parameter(torch.randn(8, 256, CODE_DIM) * .02)
        self.payloads = payloads

    def prepare_observation(self, observation: ProbeObservation, draft) -> RankState:
        base = self.base.prepare_observation(observation, draft)
        key = (observation.prompt_id, observation.round_index)
        if key not in self.payloads:
            raise ValueError("frozen legal window has no current Target payload")
        payload = self.payloads[key]
        self._consume(payload, observation.correction_token)
        return RankState(base, payload, base.corrected_prefix)

    def with_payload(self, state: RankState, payload: bytes | None) -> RankState:
        self._consume(payload, state.base.corrected_prefix[-1])
        return replace(state, payload=payload)

    @staticmethod
    def _consume(payload: bytes | None, protocol_b: int):
        return consume_rank64(payload, protocol_b=protocol_b,
                              vocab_size=VOCAB, margin_clip=MARGIN_CLIP)

    def message_control(self, payload: bytes | None, protocol_b: int) -> Tensor:
        consumed = self._consume(payload, protocol_b)
        if consumed.kind in ("no_message", "literal_zero_payload_zero_control"):
            return torch.zeros(1, CONTROL_DIM, device=self.edge_codebook.device,
                               dtype=self.edge_codebook.dtype)
        indices = torch.tensor(list(payload), device=self.edge_codebook.device,
                               dtype=torch.long)
        rows = torch.arange(8, device=self.edge_codebook.device)
        return self.edge_codebook[rows, indices].reshape(1, CONTROL_DIM)

    def forward_logits(self, draft_hidden: Tensor, state: RankState,
                       position: int | Tensor,
                       *, temperature: float | None = None) -> Tensor:
        if type(state) is not RankState or state.base.mode != "C":
            raise TypeError("RANK64 head received wrong state")
        base = state.base
        if (base.local_hidden.shape != (1, self.config.draft_hidden_size)
                or draft_hidden.ndim not in (2, 3)
                or draft_hidden.shape[0] != 1
                or draft_hidden.shape[-1] != self.config.draft_hidden_size):
            raise ValueError("Draft hidden shape differs")
        local = base.local_hidden
        if temperature is None:
            codes = self.base.core.local_encode(
                local, base.rejected_embedding, base.correction_embedding,
                base.accepted_count)
        else:
            codes = self.base.core.local_sample_st(
                local, base.rejected_embedding, base.correction_embedding,
                base.accepted_count, temperature=temperature)
        control = (self.base.core.repair._lookup(codes[:, 0], 1)
                   + self.base.core.repair._lookup(codes[:, 1], 1))
        surrogate = self.base.local_hidden_lift(local.to(
            device=self.base.local_hidden_lift[0].weight.device,
            dtype=self.base.local_hidden_lift[0].weight.dtype))
        control = control + self.base.hidden_projection(surrogate)
        semantic = self.base.local_top32_surrogate(local.to(
            device=self.base.local_top32_surrogate.weight.device,
            dtype=self.base.local_top32_surrogate.weight.dtype))
        control = control + self.base.top32_projection(semantic)
        control = control + self.message_control(
            state.payload, base.corrected_prefix[-1]).to(
                device=control.device, dtype=control.dtype)
        logits = self.base.core.repair.repair_from_control(
            draft_hidden, control, position)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite RANK64 repaired logits")
        return logits

    @torch.inference_mode()
    def next_token(self, prefix: list[int], state: RankState,
                   position: int, draft) -> int:
        if tuple(prefix[:len(state.corrected_prefix)]) != state.corrected_prefix:
            raise ValueError("Draft generation left corrected prefix")
        hidden = draft.teacher_hidden(prefix, [0])
        logits = self.forward_logits(hidden, state, position)
        if logits.shape != (1, VOCAB):
            raise ValueError("RANK64 next-token vocabulary differs")
        return int(logits[0].argmax(-1).item())

    def trainable_state_dict(self) -> dict[str, Tensor]:
        return {name: parameter.detach().cpu().clone()
                for name, parameter in self.named_parameters()
                if parameter.requires_grad}

    def load_trainable_state_dict(self, state: dict[str, Tensor]) -> None:
        params = {name: parameter for name, parameter in self.named_parameters()
                  if parameter.requires_grad}
        if set(state) != set(params):
            raise ValueError("RANK64 trainable parameter membership differs")
        with torch.no_grad():
            for name, parameter in params.items():
                if state[name].shape != parameter.shape:
                    raise ValueError(f"RANK64 trainable parameter shape differs: {name}")
                parameter.copy_(state[name].to(device=parameter.device,
                                               dtype=parameter.dtype))

    def parameter_budget(self) -> dict[str, int]:
        return {
            "matched_C_edge_trainable": int(self.base.parameter_budget()["edge_trainable"]),
            "extra_edge_codebook_trainable": self.edge_codebook.numel(),
            "cloud_encoder_trainable": 0,
            "physical_wire_bytes": 8,
            "edge_codebook_control_dim": CONTROL_DIM,
        }
