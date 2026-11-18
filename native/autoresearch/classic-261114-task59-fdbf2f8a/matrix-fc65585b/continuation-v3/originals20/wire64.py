"""One fixed RANK64-MARGIN wire layout; no target or future-label access.

Byte order is little endian. The protocol's already known top-1 token ID is
passed as a validation input and is not serialized into the extra message.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

TOP_ID_BITS = 18
TOP_ID_MAX = (1 << TOP_ID_BITS) - 1
WIRE_BYTES = 8
LAYOUT = (
    ("top2_id", 0, 18),
    ("top3_id", 18, 18),
    ("p2_q6", 36, 6),
    ("p3_q6", 42, 6),
    ("p1_q8", 48, 8),
    ("margin_q8", 56, 8),
)


@dataclass(frozen=True)
class Rank64Fields:
    top2_id: int
    top3_id: int
    p2_q6: int
    p3_q6: int
    p1_q8: int
    margin_q8: int


def _q_unit(value: float, levels: int) -> int:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probability must be finite and in [0, 1]")
    return min(levels, max(0, math.floor(value * levels + 0.5)))


def _q_margin(value: float, clip: float) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError("margin must be finite and nonnegative")
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("margin clip must be finite and positive")
    return min(255, max(0, math.floor(min(value, clip) * 255 / clip + 0.5)))


def encode_rank64_margin(*, top1_id: int, top2_id: int, top3_id: int,
                         p1: float, p2: float, p3: float, margin: float,
                         vocab_size: int, margin_clip: float) -> bytes:
    if not isinstance(vocab_size, int) or not 3 <= vocab_size <= TOP_ID_MAX:
        raise ValueError("vocabulary does not fit reserved 18-bit ID space")
    if any(not isinstance(x, int) or isinstance(x, bool)
           or not 0 <= x < vocab_size
           for x in (top1_id, top2_id, top3_id)):
        raise ValueError("top token ID outside vocabulary")
    if len({top1_id, top2_id, top3_id}) != 3:
        raise ValueError("top-1, top-2 and top-3 IDs must be distinct")
    if not (math.isfinite(p1) and math.isfinite(p2) and math.isfinite(p3)
            and 1 >= p1 >= p2 >= p3 >= 0):
        raise ValueError("top probabilities must be finite and ranked")
    fields = Rank64Fields(
        top2_id, top3_id, _q_unit(p2, 63), _q_unit(p3, 63),
        _q_unit(p1, 255), _q_margin(margin, margin_clip))
    word = 0
    for name, shift, width in LAYOUT:
        value = getattr(fields, name)
        if not 0 <= value < (1 << width):
            raise ValueError(f"{name} does not fit layout")
        word |= value << shift
    payload = word.to_bytes(WIRE_BYTES, "little", signed=False)
    if len(payload) != WIRE_BYTES:
        raise AssertionError("RANK64 physical payload length differs")
    return payload


def decode_rank64_margin(payload: bytes | bytearray, *,
                         margin_clip: float) -> dict:
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("RANK64 payload must be raw bytes")
    if len(payload) != WIRE_BYTES:
        raise ValueError("RANK64 payload must be exactly 8 bytes")
    if not math.isfinite(margin_clip) or margin_clip <= 0:
        raise ValueError("margin clip must be finite and positive")
    word = int.from_bytes(payload, "little", signed=False)
    fields = {
        name: (word >> shift) & ((1 << width) - 1)
        for name, shift, width in LAYOUT
    }
    return {
        **fields,
        "p1_hat": fields["p1_q8"] / 255,
        "p2_hat": fields["p2_q6"] / 63,
        "p3_hat": fields["p3_q6"] / 63,
        "margin_hat": fields["margin_q8"] * margin_clip / 255,
        "wire_bytes": WIRE_BYTES,
        "endian": "little",
    }


def encode_from_current_topk(target_topk: dict, *, protocol_b: int,
                             margin_clip: float) -> bytes:
    """Use b_t as canonical top-1 and the best two other current logits.

    The frozen BF16 packed Top-K can place a numerically tied alternative
    before the rechecked greedy b_t. The protocol supplies b_t; that ID is
    removed from the packed Top-K before selecting the two alternatives.
    """
    ids = target_topk["token_ids"]
    logits = target_topk["logits"]
    if (len(ids) < 3 or len(logits) < 3
            or target_topk["greedy_token_id"] != protocol_b
            or protocol_b not in ids):
        raise ValueError("current valid-prefix Top-K does not match protocol b")
    lse = float(target_topk["logsumexp"])
    if not math.isfinite(lse) or any(not math.isfinite(float(x))
                                     for x in logits[:3]):
        raise ValueError("nonfinite current Target logits")
    if not all(float(logits[i]) >= float(logits[i + 1])
               for i in range(len(logits) - 1)):
        raise ValueError("current Target top logits are not ranked")
    canonical_index = ids.index(protocol_b)
    alternatives = [(token_id, float(logit))
                    for token_id, logit in zip(ids, logits, strict=True)
                    if token_id != protocol_b]
    if len(alternatives) < 2:
        raise ValueError("current Target Top-K lacks two other IDs")
    b_logit = float(logits[canonical_index])
    if alternatives[0][1] > b_logit:
        raise ValueError("rechecked b_t ranks below a distinct packed logit")
    probabilities = [math.exp(value - lse)
                     for value in (b_logit, alternatives[0][1],
                                   alternatives[1][1])]
    if probabilities[0] > 1:
        # Top logits are stored in FP16, while logsumexp is FP32. At a narrow
        # numerical tie, clamp only this finite quantization discrepancy.
        if probabilities[0] > 1.02:
            raise ValueError("Top-K probability exceeds 1 beyond FP16 error")
        probabilities[0] = 1.0
    return encode_rank64_margin(
        top1_id=protocol_b, top2_id=alternatives[0][0],
        top3_id=alternatives[1][0],
        p1=probabilities[0], p2=min(probabilities[1], probabilities[0]),
        p3=min(probabilities[2], probabilities[1], probabilities[0]),
        margin=b_logit - alternatives[0][1],
        vocab_size=int(target_topk["vocab_size"]),
        margin_clip=margin_clip)
