"""Shared head construction and window-weighted CE for the A100 RAW5000 sweep."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from feedback_codec.v10_four_arm import V10FourArmHead
from feedback_codec.v10_transformer_bcfh import TransformerBCFHHead, TransformerRepairConfig


@dataclass(frozen=True)
class SweepTransformerConfig(TransformerRepairConfig):
    """Use the existing causal stack over the additional planned sizes."""

    def __post_init__(self):
        if self.depth not in (1, 2, 3, 4) or self.width not in (32, 64, 128, 256):
            raise ValueError("unsupported Transformer depth or width")
        if self.heads != 4:
            raise ValueError("use four attention heads for this sweep")


def build_head(config, output_head, draft_embedding, *, architecture, depth, width, seed,
               anchor_initialization=False):
    """Keep the RAW5000 F-H control and replace only the residual when needed."""
    if type(depth) is not int or type(width) is not int:
        raise TypeError("depth and width must be integers")
    # Existing modules initialize on CPU, so preserve the outer RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        options = dict(mode="F-H", draft_embedding=draft_embedding)
        if architecture == "transformer":
            if anchor_initialization:
                raise ValueError("historical initialization is specific to the MLP anchor")
            choice = SweepTransformerConfig(depth=depth, width=width)
            reference = V10FourArmHead(config, output_head, **options)
            torch.random.default_generator.manual_seed(seed)
            head = TransformerBCFHHead(config, output_head, choice=choice, **options)
            # The Transformer swaps its repair module after constructing the
            # common control. Match the original codebook initialization too.
            with torch.no_grad():
                head.core.repair.codebooks.copy_(reference.core.repair.codebooks)
            return head
        if architecture != "mlp" or depth not in (1, 2, 3, 4) or width not in (64, 128, 256, 512):
            raise ValueError("unsupported MLP architecture, depth or width")
        # CodecConfig.hidden_width also sizes the protocol encoder. Keep that
        # width fixed so this sweep changes only the output-side gate.
        head = V10FourArmHead(config, output_head, **options)
        repair = head.core.repair
        if anchor_initialization and (depth != 1 or width != config.hidden_width):
            raise ValueError("historical initialization requires the original MLP shape")
        if not anchor_initialization:
            nn.init.zeros_(repair.U.weight)
        if depth == 1 and width == config.hidden_width:
            return head
        input_size = repair.gate_mlp[0].in_features
        layers = [nn.Linear(input_size, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.SiLU()))
        layers.append(nn.Linear(width, config.rank))
        repair.gate_mlp = nn.Sequential(*layers)
        return head


def position_weights(policy, window):
    if policy == "equal":
        return (1,) * window
    if policy == "front":
        return tuple(range(window, 0, -1))
    raise ValueError("unsupported position weighting")


def window_ce(logits, labels, weights):
    """Mean of normalized per-window loss, with padding excluded."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or len(weights) != labels.shape[1]:
        raise ValueError("expected logits [batch, window, vocab] and aligned labels and weights")
    valid = labels.ne(-100)
    if any(weight <= 0 for weight in weights) or not valid.any(dim=1).all():
        raise ValueError("every window must have positive weight on a valid label")
    per_token = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(),
                                ignore_index=-100, reduction="none").reshape_as(labels)
    weights = valid * per_token.new_tensor(weights)
    return (per_token * weights).sum(dim=1) / weights.sum(dim=1)
