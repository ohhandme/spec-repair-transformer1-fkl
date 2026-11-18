"""CPU-checkable task28 reference; no model loading, data access or GPU launch.

The caller supplies only legal Draft hidden history and ordinary C control.
History is reset for every corrected-prefix/round and contains positions 1..L
of the current window, never future labels or teacher features.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _check(h: Tensor, c: Tensor, d: int, window: int) -> None:
    if h.ndim != 3 or h.shape[-1] != d or not 1 <= h.shape[1] <= window:
        raise ValueError("history must be [B,L,D],1<=L<=window,starting at position1")
    if c.shape != (h.shape[0], 128):
        raise ValueError("ordinary control must be [B,128]")
    if not bool(torch.isfinite(h).all()) or not bool(torch.isfinite(c).all()):
        raise ValueError("nonfinite edge inputs")


class MLPResidual(nn.Module):
    def __init__(self, hidden_size: int = 1024, window: int = 8):
        super().__init__()
        self.hidden_size, self.window = hidden_size, window
        self.V = nn.Linear(hidden_size, 32, bias=False)
        self.gate_mlp = nn.Sequential(nn.Linear(161, 256), nn.SiLU(), nn.Linear(256, 32))
        self.U = nn.Linear(32, hidden_size, bias=False)
        nn.init.zeros_(self.U.weight)

    def forward(self, h: Tensor, c: Tensor, *, enabled: bool = True) -> Tensor:
        _check(h, c, self.hidden_size, self.window)
        if not enabled:
            return h
        u = self.V(h.to(self.V.weight.dtype))
        p = torch.arange(1, h.shape[1]+1, device=h.device, dtype=u.dtype)
        p = (p/self.window).view(1, -1, 1).expand(h.shape[0], -1, -1)
        control = c.to(u.dtype)[:, None, :].expand(-1, h.shape[1], -1)
        gate = self.gate_mlp(torch.cat((control, u, p), dim=-1))
        return h + self.U(gate*u).to(h.dtype)


class CausalTransformerResidual(nn.Module):
    """One pre-norm causal block; d64,4heads,FFN256,zero dropout."""
    def __init__(self, hidden_size: int = 1024, window: int = 8):
        super().__init__()
        self.hidden_size, self.window = hidden_size, window
        self.down = nn.Linear(hidden_size, 64, bias=False)
        self.control = nn.Linear(128, 64, bias=False)
        self.position = nn.Linear(1, 64, bias=False)
        self.norm1, self.norm2 = nn.LayerNorm(64), nn.LayerNorm(64)
        self.attn = nn.MultiheadAttention(64, 4, dropout=0.0, bias=True, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, 64))
        self.up = nn.Linear(64, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, h: Tensor, c: Tensor, *, enabled: bool = True) -> Tensor:
        _check(h, c, self.hidden_size, self.window)
        if not enabled:
            return h
        x = h.to(self.down.weight.dtype)
        length = x.shape[1]
        p = torch.arange(1, length+1, device=x.device, dtype=x.dtype).view(1, -1, 1)/self.window
        z = self.down(x) + self.control(c.to(x.dtype))[:, None, :] + self.position(p)
        mask = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        q = self.norm1(z)
        attention, _ = self.attn(q, q, q, attn_mask=mask, need_weights=False)
        z = z + attention
        z = z + self.ffn(self.norm2(z))
        return h + self.up(z).to(h.dtype)


def ordinary_c_control(base: nn.Module, state, temperature: float | None = None) -> Tensor:
    """Same C-only information as task23, including its short-full fallback.

    The separately audited ordinary-C prepared-state builder remains required;
    this function does not infer protocol legality or consume observations.
    """
    if base.mode != "C" or state.mode != "C":
        raise ValueError("ordinary C only")
    if state.target_hidden is not None or state.target_top32_features is not None:
        raise ValueError("online teacher features forbidden")
    if state.correction_embedding is None or state.accepted_count is None:
        raise ValueError("ordinary protocol missing")
    local = state.local_hidden
    rejected = state.rejected_embedding if state.rejected_embedding is not None else local
    args = (local, rejected, state.correction_embedding, state.accepted_count)
    if temperature is None:
        codes = base.core.local_encode(*args)
    else:
        codes = base.core.local_sample_st(*args, temperature=temperature)
    control = base.core.repair._lookup(codes[:, 0], 1) + base.core.repair._lookup(codes[:, 1], 1)
    lift = base.local_hidden_lift(local.to(base.local_hidden_lift[0].weight.dtype))
    semantic = base.local_top32_surrogate(local.to(base.local_top32_surrogate.weight.dtype))
    return control + base.hidden_projection(lift) + base.top32_projection(semantic)


class COnlyHead(nn.Module):
    """Integration skeleton: owns ordinary C base and an independent repair.

    Do not call base.forward_logits after constructing this wrapper. Its old
    output repair is inactive/frozen; codebooks and common C encoders stay active.
    Runner owns causal preparation, round reset and checkpoint/init locks.
    """
    def __init__(self, base: nn.Module, architecture: str):
        super().__init__()
        if base.mode != "C":
            raise ValueError("base must be C")
        self.base = base
        self.config, self.mode = base.config, "C"
        d = base.config.draft_hidden_size
        if architecture == "mlp":
            self.adapter = MLPResidual(d)
            self.adapter.V.load_state_dict(base.core.repair.V.state_dict())
            self.adapter.gate_mlp.load_state_dict(base.core.repair.gate_mlp.state_dict())
        elif architecture == "transformer":
            self.adapter = CausalTransformerResidual(d)
        else:
            raise ValueError("architecture must be mlp or transformer")
        for module in (base.core.repair.V, base.core.repair.gate_mlp, base.core.repair.U):
            module.requires_grad_(False)
        base.core.repair.output_head.requires_grad_(False)

    def forward_logits(self, history: Tensor, state, *, temperature=None, enabled=True) -> Tensor:
        if not enabled:
            return self.base.core.repair.output_head(history)
        control = ordinary_c_control(self.base, state, temperature)
        repaired = self.adapter(history, control)
        return self.base.core.repair.output_head(repaired)

    def parameter_budget(self) -> dict:
        # Legacy parameter_budget() names the inactive V/gate/U; never reuse it.
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        repair = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        return dict(edge_trainable=total, repair_trainable=repair,
                    common_trainable=total-repair, cloud_trainable=0,
                    additional_wire_bytes=0)

    def trainable_state_dict(self) -> dict[str, Tensor]:
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def load_trainable_state_dict(self, state: dict[str, Tensor]) -> None:
        expected={n:p for n,p in self.named_parameters() if p.requires_grad}
        if set(state)!=set(expected):
            raise ValueError("trainable checkpoint key set differs")
        for n,p in expected.items():
            if state[n].shape!=p.shape or not bool(torch.isfinite(state[n]).all()):
                raise ValueError("invalid trainable checkpoint tensor: "+n)
        with torch.no_grad():
            for n,p in expected.items():p.copy_(state[n].to(device=p.device,dtype=p.dtype))


def weighted_loss(student_logits: Tensor, valid: Tensor, *, labels: Tensor | None = None,
                  teacher_logits: Tensor | None = None, window: int = 8) -> Tensor:
    """Exactly CE or full-vocabulary forward KL, temperature1,weights8..1."""
    if (labels is None) == (teacher_logits is None):
        raise ValueError("provide exactly one of labels or teacher_logits")
    if student_logits.ndim != 3 or valid.dtype != torch.bool or valid.shape != student_logits.shape[:2]:
        raise ValueError("logits [B,L,V] and bool valid [B,L] required")
    length = student_logits.shape[1]
    if not 1 <= length <= window or not bool(valid.any()):
        raise ValueError("empty or oversized window")
    if not bool(torch.isfinite(student_logits).all()):
        raise FloatingPointError("student logits nonfinite")
    if labels is not None:
        if labels.shape != valid.shape:
            raise ValueError("labels shape differs")
        per = F.cross_entropy(student_logits.float().transpose(1, 2), labels, reduction="none")
    else:
        if teacher_logits.shape != student_logits.shape or not bool(torch.isfinite(teacher_logits).all()):
            raise ValueError("full finite teacher logits with identical shape required")
        logp = F.log_softmax(teacher_logits.detach().float(), dim=-1)
        logq = F.log_softmax(student_logits.float(), dim=-1)
        per = (logp.exp()*(logp-logq)).sum(dim=-1)
    weights = torch.arange(window, window-length, -1, device=per.device, dtype=per.dtype)[None, :]
    weights = weights * valid.to(per.dtype)
    return (weights*per).sum()/weights.sum()
