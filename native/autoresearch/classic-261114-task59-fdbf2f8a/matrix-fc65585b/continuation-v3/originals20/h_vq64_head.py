"""Matched rank-32 C repair head with one conditioned 8-byte VQ control."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from feedback_codec.v10_four_arm import V10FourArmHead, V10PreparedState
from feedback_codec.fixed_probe import ProbeObservation
from causal_operands import causal_inputs
from h_vq64_codec import ConditionedVQ64, EDGE_CONTROL_DIM


EOS = 151645
MODE = "H-VQ64-TASK-conditioned"


@dataclass(frozen=True)
class MessageState:
    base: V10PreparedState
    online_inputs: tuple[Tensor, int, Tensor | None, Tensor]
    payload: bytes | None
    corrected_prefix: tuple[int, ...]


class HTaskHead(nn.Module):
    def __init__(self, config, output_head: nn.Module, *,
                 draft_embedding: nn.Module) -> None:
        super().__init__()
        if config.window != 8 or config.rank != 32 or config.groups != 8 \
                or config.choices != 256 or config.code_dim != 16:
            raise ValueError("frozen rank-32 repair/8x256x16 control changed")
        self.config = config
        self.mode = MODE
        self.base = V10FourArmHead(config, output_head, mode="C",
                                   draft_embedding=draft_embedding)
        self.message = ConditionedVQ64(
            target_hidden_size=config.target_hidden_size,
            draft_hidden_size=config.draft_hidden_size,
            conditioned=True)
        if self.message.codebook.shape != (8, 256, 16):
            raise AssertionError("edge codebook capacity differs")

    def prepare_observation(self, observation: ProbeObservation, draft) -> MessageState:
        base = self.base.prepare_observation(observation, draft)
        table = draft.model.get_input_embeddings()
        inputs = causal_inputs(
            observation, table,
            target_hidden_size=self.config.target_hidden_size,
            draft_hidden_size=self.config.draft_hidden_size, eos=EOS)
        if inputs is None:
            raise ValueError("legal H-VQ64 training/evaluation window lacks feedback")
        payload = self.message.encode_payload(*inputs)
        return MessageState(base=base, online_inputs=inputs,
                            payload=payload,
                            corrected_prefix=base.corrected_prefix)

    def with_payload(self, state: MessageState, payload: bytes | None) -> MessageState:
        if payload is not None and (type(payload) is not bytes or len(payload) != 8):
            raise ValueError("intervention must preserve 8 raw bytes or None")
        return replace(state, payload=payload)

    def forward_logits(self, draft_hidden: Tensor, state: MessageState,
                       position: int | Tensor,
                       *, temperature: float | None = None) -> Tensor:
        if type(state) is not MessageState or state.base.mode != "C":
            raise TypeError("H-VQ64 head received wrong state")
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
        message_control = (self.message.train_control(
            *state.online_inputs, temperature=temperature or 1.0)
            if self.training else self.message.decode_payload(state.payload))
        if message_control.shape != (1, EDGE_CONTROL_DIM):
            raise ValueError("decoded 8-byte control width changed")
        control = control + message_control.to(device=control.device,
                                               dtype=control.dtype)
        logits = self.base.core.repair.repair_from_control(
            draft_hidden, control, position)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite H-VQ64 repaired logits")
        return logits

    @torch.inference_mode()
    def next_token(self, prefix: list[int], state: MessageState,
                   position: int, draft) -> int:
        if tuple(prefix[:len(state.corrected_prefix)]) != state.corrected_prefix:
            raise ValueError("Draft generation left corrected prefix")
        hidden = draft.teacher_hidden(prefix, [0])
        logits = self.forward_logits(hidden, state, position)
        if logits.shape != (1, self.config.vocab_size):
            raise ValueError("H-VQ64 next-token vocabulary shape differs")
        return int(logits[0].argmax(-1).item())

    def trainable_state_dict(self) -> dict[str, Tensor]:
        return {name: parameter.detach().cpu().clone()
                for name, parameter in self.named_parameters()
                if parameter.requires_grad}

    def load_trainable_state_dict(self, state: dict[str, Tensor]) -> None:
        params = {name: parameter for name, parameter in self.named_parameters()
                  if parameter.requires_grad}
        if set(state) != set(params):
            raise ValueError("trainable parameter membership differs")
        with torch.no_grad():
            for name, parameter in params.items():
                if state[name].shape != parameter.shape:
                    raise ValueError(f"trainable parameter shape differs: {name}")
                parameter.copy_(state[name].to(device=parameter.device,
                                               dtype=parameter.dtype))

    def parameter_budget(self) -> dict[str, int]:
        base = self.base.parameter_budget()
        return {
            "matched_C_edge_trainable": int(base["edge_trainable"]),
            "extra_edge_codebook_trainable": self.message.codebook.numel(),
            "cloud_encoder_trainable": sum(p.numel() for p in self.message.encoder.parameters()),
            "physical_wire_bytes": 8,
            "edge_codebook_control_dim": EDGE_CONTROL_DIM,
        }
