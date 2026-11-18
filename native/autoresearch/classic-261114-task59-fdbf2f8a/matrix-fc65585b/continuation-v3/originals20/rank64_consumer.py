"""Strict edge consumer for the frozen little-endian RANK64-MARGIN wire."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

ROOT = Path("/mnt/files2/dream/feedback-coded-specdec-v2/runs/v13-single-run-20261102")
WIRE_SOURCE = ROOT / "source_8byte_preflight_v2/wire64.py"
WIRE_SHA256 = "ec8a8f0cef501869f570542e147f67ff66be84a3d1ec8a6befe878230bb569ec"
LOCKED_VOCAB_SIZE = 151936
RESERVED_ID = 262143
ZERO_CONTROL_PAYLOAD = bytes(8)


@dataclass(frozen=True)
class ConsumedRank64:
    kind: str
    control: dict | None
    physical_bytes: int


def _wire_decoder():
    with WIRE_SOURCE.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != WIRE_SHA256:
            raise ValueError("frozen little-endian wire source differs")
    sys.path.insert(0, str(WIRE_SOURCE.parent))
    from wire64 import decode_rank64_margin
    return decode_rank64_margin


def consume_rank64(payload: bytes | None, *, protocol_b: int,
                   vocab_size: int, margin_clip: float) -> ConsumedRank64:
    if type(vocab_size) is not int or vocab_size != LOCKED_VOCAB_SIZE:
        raise ValueError("actual Target vocabulary lock differs")
    if type(protocol_b) is not int or not 0 <= protocol_b < vocab_size:
        raise ValueError("already-known protocol b_t is invalid")
    if payload is None:
        return ConsumedRank64("no_message", None, 0)
    if type(payload) is not bytes or len(payload) != 8:
        raise TypeError("additional message must be exactly eight raw bytes")
    if payload == ZERO_CONTROL_PAYLOAD:
        return ConsumedRank64("literal_zero_payload_zero_control", None, 8)
    decoded = _wire_decoder()(payload, margin_clip=margin_clip)
    ids = (decoded["top2_id"], decoded["top3_id"])
    if any(identifier == RESERVED_ID or not 0 <= identifier < vocab_size
           for identifier in ids):
        raise ValueError("RANK64 decoded an out-of-vocabulary or reserved ID")
    if len({protocol_b, *ids}) != 3:
        raise ValueError("RANK64 alternatives conflict with protocol b_t")
    if decoded["p2_q6"] < decoded["p3_q6"]:
        raise ValueError("RANK64 alternative probabilities are not ranked")
    if decoded["p1_hat"] + (0.5 / 255 + 0.5 / 63) < decoded["p2_hat"]:
        raise ValueError("RANK64 top-1 probability conflicts with protocol b_t")
    return ConsumedRank64("rank64_message", decoded, 8)
