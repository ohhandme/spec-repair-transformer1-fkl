"""Causal output-side repair candidates for the independent 4090 search.

The caller supplies the current corrected-window Draft history and the same
ordinary C control used by Task28.  This module never reads Target labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Architecture:
    depth: int
    width: int
    heads: int = 4
    hidden_size: int = 1024
    control_size: int = 128
    window: int = 8

    def __post_init__(self) -> None:
        if self.depth not in (1, 2, 3):
            raise ValueError("depth must be 1, 2, or 3")
        if self.width not in (32, 64, 128, 256):
            raise ValueError("width is outside the registered search")
        if self.heads != 4 or self.width % self.heads:
            raise ValueError("four attention heads are required")
        if self.hidden_size != 1024 or self.control_size != 128 or self.window != 8:
            raise ValueError("Draft and protocol dimensions are locked")


class CausalBlock(nn.Module):
    def __init__(self, config: Architecture) -> None:
        super().__init__()
        width = config.width
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, config.heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 4), nn.SiLU(),
                                 nn.Linear(width * 4, width))

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        q = self.norm1(values)
        update, _ = self.attn(q, q, q, attn_mask=mask, need_weights=False)
        values = values + update
        return values + self.ffn(self.norm2(values))


class CausalRepairStack(nn.Module):
    """One to three causal blocks with a zero-initialized residual output."""

    def __init__(self, config: Architecture) -> None:
        super().__init__()
        self.config = config
        self.down = nn.Linear(config.hidden_size, config.width, bias=False)
        self.control = nn.Linear(config.control_size, config.width, bias=False)
        self.position = nn.Linear(1, config.width, bias=False)
        self.blocks = nn.ModuleList(CausalBlock(config) for _ in range(config.depth))
        self.up = nn.Linear(config.width, config.hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, history: Tensor, control: Tensor, *, enabled: bool = True) -> Tensor:
        c = self.config
        if (history.ndim != 3 or not 1 <= history.shape[1] <= c.window
                or history.shape[2] != c.hidden_size
                or control.shape != (history.shape[0], c.control_size)):
            raise ValueError("repair inputs must be [B,L,1024] and [B,128]")
        if not torch.isfinite(history).all() or not torch.isfinite(control).all():
            raise ValueError("nonfinite repair input")
        if not enabled:
            return history
        x = history.to(self.down.weight.dtype)
        t = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)
        t = t.view(1, -1, 1) / c.window
        values = (self.down(x)
                  + self.control(control.to(device=x.device, dtype=x.dtype))[:, None, :]
                  + self.position(t))
        causal_mask = torch.ones(
            x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            values = block(values, causal_mask)
        return history + self.up(values).to(history.dtype)
