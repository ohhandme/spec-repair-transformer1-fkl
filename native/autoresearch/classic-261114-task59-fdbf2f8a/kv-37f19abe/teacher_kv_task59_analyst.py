"""Read the sealed RAW5000 KV boundary and project valid positions into repair control."""

from hashlib import sha256
import json

import torch
from torch import nn


def _prefix_sha(ids):
    return sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def read_boundary(row, prefix_before, reader):
    """Return K, V, mask for one round eligible for next-window supervision."""
    proposal, accepted = row["candidate_ids"], row["accepted_count"]
    answer = row["correction_id"] if row["correction_id"] is not None else row["bonus_id"]
    prefix = prefix_before + proposal[:accepted]
    metadata = row["kv"]
    if (row["stop_reason"] != "continue" or answer is None or not 0 <= accepted <= len(proposal)
            or row["committed_ids"] != proposal[:accepted] + [answer]
            or row["prefix_before_sha256"] != _prefix_sha(prefix_before)
            or row["prefix_after_commit_sha256"] != _prefix_sha(prefix + [answer])
            or row["hidden"]["source_prefix_sha256"] != _prefix_sha(prefix)
            or row["hidden"]["predicted_token_id"] != answer):
        raise ValueError("KV supervision needs an aligned, nonterminal decision boundary")
    positions = list(range(max(0, len(prefix) - 4), len(prefix)))
    mask = [True] * len(positions) + [False] * (4 - len(positions))
    if (metadata["layer_ids"] != [34, 35] or metadata["absolute_positions"] != positions
            or metadata["cache_length"] != len(prefix) or metadata["valid_mask"] != mask
            or metadata["K_RoPE"] != "post_RoPE" or metadata["dtype"] != "BF16"
            or metadata["available_before_receipt"] is not True
            or metadata["excludes_unprocessed_answer"] is not True):
        raise ValueError("KV tensor metadata does not match the processed prefix")
    key = row["tensor_prefix"]
    k, v, valid = (reader.get_tensor(f"{key}.{field}") for field in ("K", "V", "valid"))
    if (valid.dtype != torch.bool or valid.shape != (4,) or valid.tolist() != mask
            or any(t.shape != (2, 4, 8, 128) or t.dtype != torch.bfloat16
                   or not torch.isfinite(t).all() or torch.count_nonzero(t[:, ~valid])
                   for t in (k, v))):
        raise ValueError("KV tensor content differs from the declared format")
    return k, v, valid


class KVControl(nn.Module):
    """Compress two layers of K/V to a correction-boundary control vector."""

    def __init__(self, control_dim=128, width=32):
        super().__init__()
        self.key = nn.Linear(1024, width, bias=False)
        self.value = nn.Linear(1024, width, bias=False)
        self.output = nn.Linear(2 * 4 * width, control_dim, bias=False)

    def forward(self, key, value, valid):
        batch = valid.shape[0]
        if (valid.shape != (batch, 4) or valid.dtype != torch.bool
                or key.shape != (batch, 2, 4, 8, 128) or value.shape != key.shape):
            raise ValueError("expected batched RAW5000 K, V and valid mask")
        mask = valid[:, None, :, None]
        k = self.key(key.flatten(-2).to(self.key.weight.dtype))
        v = self.value(value.flatten(-2).to(self.value.weight.dtype))
        projected = torch.where(mask, torch.nn.functional.silu(k + v), 0)
        return self.output(projected.flatten(1))
