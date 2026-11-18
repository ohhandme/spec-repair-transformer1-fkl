"""Versioned teacher-conditioned repair and post-repair hazard interfaces.

No model loading, Target calls, dataset access, scheduler, or GPU launch here.
The runner supplies the frozen ordinary-C base, legal causal Draft history,
and a receipt-bound teacher message. Actual provenance still needs runner audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import sys

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from legacy_repair import COnlyHead, ordinary_c_control
from ordinary_prepare import _base_prepare


def digest_tokens(tokens) -> str:
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()


def _sha(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class RoundContext:
    prompt_id: str
    family_id: str
    split: str
    round_index: int
    valid_prefix: tuple[int, ...]
    corrected_prefix: tuple[int, ...]
    target_model_sha256: str

    def __post_init__(self):
        if (not self.prompt_id or not self.family_id or self.split not in {'train', 'dev', 'test'}
                or type(self.round_index) is not int or self.round_index < 0
                or not self.valid_prefix
                or self.corrected_prefix[:-1] != self.valid_prefix
                or not _sha(self.target_model_sha256)):
            raise ValueError('invalid receipt/round causal context')

    @property
    def round_key(self):
        return (self.prompt_id, self.round_index, digest_tokens(self.corrected_prefix))


@dataclass(frozen=True)
class TeacherMessage:
    """Extra application payload; required receipt/framing bytes are separate.

    A shuffle is an explicit locked negative control, never disguised as matched
    provenance. Source metadata describe the donor, destination the scored round.
    """
    codec: str
    payload: bytes
    feature_dim: int
    source_prompt_id: str
    source_family_id: str
    split: str
    source_round_index: int
    source_prefix_sha256: str
    destination_round_key: tuple[str, int, str]
    target_model_sha256: str
    pairing_lock_sha256: str | None = None
    codec_checkpoint_sha256: str | None = None

    @classmethod
    def full_hidden(cls, hidden: Tensor, source: RoundContext,
                    destination: RoundContext | None = None, *, pairing_lock_sha256=None):
        if (hidden.ndim != 2 or hidden.shape[0] != 1 or hidden.shape[1] <= 0
                or hidden.dtype not in {torch.bfloat16, torch.float32}
                or not bool(torch.isfinite(hidden).all()) or sys.byteorder != 'little'):
            raise ValueError('full teacher feature must be finite [1,T] bf16/fp32 on little-endian host')
        destination = source if destination is None else destination
        raw = hidden.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()
        return cls('bf16le-v1' if hidden.dtype == torch.bfloat16 else 'fp32le-v1', bytes(raw),
                   hidden.shape[1], source.prompt_id, source.family_id, source.split,
                   source.round_index, digest_tokens(source.valid_prefix), destination.round_key,
                   source.target_model_sha256, pairing_lock_sha256)

    def validate(self, context: RoundContext, *, shuffled: bool, feature_dim: int):
        if (type(self.payload) is not bytes or self.feature_dim != feature_dim
                or self.destination_round_key != context.round_key or self.split != context.split
                or self.target_model_sha256 != context.target_model_sha256
                or not _sha(self.source_prefix_sha256)):
            raise ValueError('teacher payload shape, split, model or destination binding differs')
        if shuffled:
            if (self.source_prompt_id == context.prompt_id or self.source_family_id == context.family_id
                    or not _sha(self.pairing_lock_sha256)):
                raise ValueError('shuffle needs a locked same-split different-family donor')
        elif (self.source_prompt_id != context.prompt_id or self.source_family_id != context.family_id
              or self.source_round_index != context.round_index
              or self.source_prefix_sha256 != digest_tokens(context.valid_prefix)
              or self.pairing_lock_sha256 is not None):
            raise ValueError('matched teacher source boundary differs')

    def decode_full_hidden(self, *, device, dtype):
        if self.codec not in {'bf16le-v1', 'fp32le-v1'}:
            raise ValueError('expected full-hidden feature codec')
        storage = torch.bfloat16 if self.codec == 'bf16le-v1' else torch.float32
        width = 2 if storage == torch.bfloat16 else 4
        if len(self.payload) != self.feature_dim * width:
            raise ValueError('full teacher payload byte length differs')
        x = torch.frombuffer(bytearray(self.payload), dtype=storage).clone().reshape(1, self.feature_dim)
        if not bool(torch.isfinite(x).all()):
            raise ValueError('nonfinite decoded teacher feature')
        return x.to(device=device, dtype=dtype)


class TeacherBridge(nn.Module):
    def __init__(self, target_size: int, control_size: int = 128, bottleneck: int = 64):
        super().__init__()
        self.target_size, self.control_size = target_size, control_size
        self.net = nn.Sequential(nn.Linear(target_size, bottleneck), nn.SiLU(),
                                 nn.Linear(bottleneck, control_size))

    def forward(self, feature):
        if feature.ndim != 2 or feature.shape[-1] != self.target_size or not torch.isfinite(feature).all():
            raise ValueError('teacher/local proxy feature shape or finiteness differs')
        return self.net(feature.to(self.net[0].weight))

    def checkpoint_digest(self):
        h = hashlib.sha256()
        for name, value in self.state_dict().items():
            h.update(name.encode())
            h.update(str((tuple(value.shape), value.dtype)).encode())
            h.update(bytes(value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
        return h.hexdigest()


@dataclass(frozen=True)
class StepEvidence:
    repaired_hidden: Tensor
    repaired_logits: Tensor
    candidates: Tensor
    teacher_control: Tensor
    round_key: tuple[str, int, str] | None
    prefix_sha256: str
    extra_payload_bytes: int
    signal_mode: str


class TeacherRepairHead(COnlyHead):
    """Same MLP/TR adapter, new matched-capacity conditional control branch.

    off: exact legacy C-only path, bridge frozen, old checkpoint names usable.
    local_proxy: 0 extra bytes, new bridge consumes local Target-hidden proxy.
    matched/shuffled: identical bridge/parameter graph, full teacher payload.
    encoded: cloud bridge output quantized by a version-bound progressive codec.
    Adapter output projection still starts at zero, hence all initial logits RAW.
    """
    def __init__(self, base, architecture: str, signal_mode: str, *, codec=None):
        if signal_mode not in {'off', 'local_proxy', 'matched', 'shuffled', 'encoded'}:
            raise ValueError('unknown teacher signal mode')
        super().__init__(base, architecture)
        self.signal_mode = signal_mode
        self._wire_digest = None
        self.teacher_bridge = TeacherBridge(self.config.target_hidden_size)
        self.codec = codec
        if signal_mode == 'off':
            self.teacher_bridge.requires_grad_(False)
        if signal_mode == 'encoded' and codec is None:
            raise ValueError('encoded mode requires an explicit checkpoint-bound codec')
        self._snapshot_names = frozenset(n for n,p in self.named_parameters() if p.requires_grad)

    def snapshot_state(self):
        """Full small-head snapshot remains complete AFTER freeze_for_serving.

        It excludes immutable model/output weights, includes frozen repair weights
        during compression fine-tuning, and avoids requires_grad-dependent saves.
        """
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if n in self._snapshot_names}

    def _restore_named(self, state, expected):
        parameters = dict(self.named_parameters())
        if set(state) != set(expected):
            raise ValueError('snapshot checkpoint key set differs')
        for n in expected:
            if state[n].shape != parameters[n].shape or not torch.isfinite(state[n]).all():
                raise ValueError('snapshot tensor shape or finiteness differs: '+n)
        with torch.no_grad():
            for n in expected:
                parameters[n].copy_(state[n].to(parameters[n]))
        self._wire_digest = None

    def restore_snapshot(self, state):
        self._restore_named(state, self._snapshot_names)

    def begin_codec_training(self, full_repair_snapshot):
        if self.signal_mode != 'encoded':
            raise ValueError('codec stage requires encoded head')
        expected = {n for n in self._snapshot_names if not n.startswith('codec.')}
        self._restore_named(full_repair_snapshot, expected)
        self.base.requires_grad_(False)
        self.adapter.requires_grad_(False)
        self.teacher_bridge.requires_grad_(True)
        self.codec.requires_grad_(True)
        self.train()

    def train(self, mode=True):
        if mode:
            self._wire_digest = None
        return super().train(mode)

    def freeze_for_serving(self):
        """Freeze the deployed generator; compute representation hashes ONCE.

        Never hash multi-megabyte codebooks in each decoding step. The immutable
        artifact/runner contract forbids subsequent hot mutation of these weights.
        """
        self.eval()
        self.requires_grad_(False)
        if self.codec is not None:
            self._wire_digest = hashlib.sha256((self.teacher_bridge.checkpoint_digest()
                                                + self.codec.checkpoint_digest()).encode()).hexdigest()
        return self

    @torch.no_grad()
    def encode_message(self, hidden, context, budget):
        """Cloud-side helper. Caller obtains only the legal valid-prefix feature."""
        if self.signal_mode != 'encoded' or self._wire_digest is None:
            raise ValueError('encoded serving head must be frozen with a checkpoint digest')
        latent = self.teacher_bridge(hidden.detach())
        payload = self.codec.encode(latent, budget)
        return TeacherMessage('residual-u8-v1', payload, 128, context.prompt_id, context.family_id,
                              context.split, context.round_index, digest_tokens(context.valid_prefix),
                              context.round_key, context.target_model_sha256,
                              codec_checkpoint_sha256=self._wire_digest)

    def train_encoded(self, history, state, *, context, full_message, budget):
        """Hard-byte forward with ST gradients, using causal full-feature cache.

        Full features are available to the training cloud encoder; the repair
        consumes ONLY the actual b-byte decoder output, including b=0 absence.
        """
        if not self.training or self.signal_mode != 'encoded':
            raise ValueError('encoded training interface requires training mode')
        if tuple(state.corrected_prefix) != context.corrected_prefix:
            raise ValueError('encoded training boundary differs')
        full_message.validate(context, shuffled=False, feature_dim=self.config.target_hidden_size)
        feature = full_message.decode_full_hidden(device=self.teacher_bridge.net[0].weight.device,
                                                 dtype=self.teacher_bridge.net[0].weight.dtype)
        payload, extra, commitment = self.codec.train_control(self.teacher_bridge(feature.detach()), budget)
        control = ordinary_c_control(self.base, state)
        repaired = self.adapter(history, control + extra.to(control))
        logits = self.base.core.repair.output_head(repaired)
        evidence = StepEvidence(repaired, logits, logits.argmax(-1), extra, context.round_key,
                                digest_tokens(state.corrected_prefix), len(payload), 'encoded')
        return evidence, commitment, payload

    def prepare_observation(self, observation, draft):
        return _base_prepare(self, observation, draft)

    def _extra_control(self, state, context, message):
        if self.signal_mode == 'off':
            if message is not None:
                raise ValueError('legacy off path must receive no teacher message')
            return state.local_hidden.new_zeros((1, 128)), 0
        if self.signal_mode == 'local_proxy':
            if message is not None:
                raise ValueError('0-byte matched-capacity arm must receive no teacher message')
            proxy = self.base.local_hidden_lift(state.local_hidden.to(self.base.local_hidden_lift[0].weight))
            return self.teacher_bridge(proxy), 0
        if context is None or message is None or tuple(state.corrected_prefix) != context.corrected_prefix:
            raise ValueError('online teacher arm needs receipt-bound context and message')
        if self.signal_mode == 'encoded':
            message.validate(context, shuffled=False, feature_dim=128)
            if (self._wire_digest is None or message.codec != 'residual-u8-v1'
                    or message.codec_checkpoint_sha256 != self._wire_digest):
                raise ValueError('wrong progressive codec or checkpoint digest')
            if len(message.payload) not in self.codec.budgets:
                raise ValueError('unsupported dynamic byte budget')
            control = self.codec.decode(message.payload)
            return control.to(self.teacher_bridge.net[0].weight), len(message.payload)
        message.validate(context, shuffled=self.signal_mode == 'shuffled', feature_dim=self.config.target_hidden_size)
        feature = message.decode_full_hidden(device=self.teacher_bridge.net[0].weight.device,
                                             dtype=self.teacher_bridge.net[0].weight.dtype)
        return self.teacher_bridge(feature.detach()), len(message.payload)

    def forward_evidence(self, history: Tensor, state, *, context=None, message=None, enabled=True):
        if (history.ndim != 3 or history.shape[0] != 1 or not 1 <= history.shape[1] <= 8
                or history.shape[2] != self.config.draft_hidden_size or not torch.isfinite(history).all()):
            raise ValueError('current entry requires finite one-window causal history [1,L<=8,D]')
        if context is not None and tuple(state.corrected_prefix) != context.corrected_prefix:
            raise ValueError('prepared round and context corrected-prefix differ')
        if enabled:
            control = ordinary_c_control(self.base, state)
            extra, payload_bytes = self._extra_control(state, context, message)
            repaired = self.adapter(history, control + extra.to(control))
        else:
            repaired = history
            extra, payload_bytes = history.new_zeros((1, 128)), 0
        logits = self.base.core.repair.output_head(repaired)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('nonfinite repaired logits')
        return StepEvidence(repaired, logits, logits.argmax(-1), extra,
                            None if context is None else context.round_key,
                            digest_tokens(state.corrected_prefix), payload_bytes, self.signal_mode)

    def forward_logits(self, history, state, *, context=None, message=None, enabled=True, temperature=None):
        if temperature is not None:
            raise ValueError('this version uses frozen deterministic ordinary-C encoding')
        return self.forward_evidence(history, state, context=context, message=message, enabled=enabled).repaired_logits

    @torch.inference_mode()
    def next_token_evidence(self, prefix, state, position, draft, *, context=None, message=None):
        """Return causal repaired evidence so online risk needs no second repair pass."""
        corrected = tuple(state.corrected_prefix)
        if tuple(prefix[:len(corrected)]) != corrected:
            raise ValueError('prefix left corrected branch')
        generated = list(prefix[len(corrected):])
        if len(generated) != position - 1 or not 1 <= position <= 8:
            raise ValueError('history position differs')
        history = draft.teacher_hidden(list(corrected), generated + [0])
        if history.shape != (position, self.config.draft_hidden_size):
            raise ValueError('causal Draft history shape differs')
        return self.forward_evidence(history.unsqueeze(0), state, context=context, message=message)

    @torch.inference_mode()
    def next_token(self, prefix, state, position, draft, *, context=None, message=None):
        evidence = self.next_token_evidence(prefix, state, position, draft, context=context, message=message)
        return int(evidence.candidates[0, -1])

    def parameter_budget(self):
        budget = super().parameter_budget()
        bridge = sum(p.numel() for p in self.teacher_bridge.parameters() if p.requires_grad)
        codec = 0 if self.codec is None else sum(p.numel() for p in self.codec.parameters() if p.requires_grad)
        budget['common_trainable'] -= bridge + codec
        budget.update(teacher_bridge_trainable=bridge, additional_wire_bytes='per-message actual payload; not fixed8',
                      codec_trainable=codec, signal_mode=self.signal_mode,
                      note='encoded deployment moves bridge to cloud; report physical placement separately')
        budget['unique_trainable_total'] = budget['edge_trainable']
        if self.signal_mode == 'encoded':
            budget['edge_trainable'] -= bridge
            budget['cloud_trainable'] = bridge
        return budget


def repaired_risk_statistics(evidence: StepEvidence, window=8):
    """No labels/Target access. Features come exclusively from repaired logits."""
    z = evidence.repaired_logits.detach().float()
    ids = evidence.candidates.detach()
    if z.ndim != 3 or ids.shape != z.shape[:2] or not torch.equal(ids, z.argmax(-1)):
        raise ValueError('risk evidence must bind to the actual repaired greedy candidates')
    if not 1 <= z.shape[1] <= window or z.shape[-1] < 2 or not torch.isfinite(z).all():
        raise ValueError('invalid repaired evidence')
    logp = F.log_softmax(z, -1)
    p = logp.exp()
    top2 = p.topk(2, -1).values
    selected = p.gather(-1, ids[..., None]).squeeze(-1)
    entropy = -(p * logp).sum(-1) / math.log(z.shape[-1])
    pos = torch.arange(1, z.shape[1]+1, device=z.device, dtype=z.dtype)[None, :].expand(z.shape[0], -1)/window
    byte_scale = min(1.0, math.log1p(evidence.extra_payload_bytes)/math.log1p(8192))
    budget = torch.full_like(pos, byte_scale)
    present = torch.full_like(pos, float(evidence.extra_payload_bytes > 0))
    return torch.stack((selected, entropy, top2[..., 0]-top2[..., 1], pos, budget, present), -1).detach()


class PostRepairHazardHead(nn.Module):
    """Train only after freezing the selected repair generator.

    local vs teacher risk arms use exactly the same StepEvidence/candidates.
    The optional teacher argument can be shuffled for the risk-only ablation
    without changing the generator's real message or its candidate sequence.
    """
    def __init__(self, hidden_size, *, use_teacher: bool):
        super().__init__()
        self.use_teacher = use_teacher
        self.hidden = nn.Sequential(nn.Linear(hidden_size, 64), nn.SiLU())
        self.net = nn.Sequential(nn.Linear(64+6+128, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, evidence: StepEvidence, *, teacher_control=None):
        stats = repaired_risk_statistics(evidence)
        if teacher_control is None:
            teacher_control = evidence.teacher_control
        return self.forward_cached(evidence.repaired_hidden, stats, teacher_control)

    def forward_cached(self, repaired_hidden, post_repair_statistics, teacher_control):
        """CPU/head-only training from byte-locked post-repair feature caches.

        Cache producer must bind its generator checkpoint, round, candidate IDs,
        and this statistics schema. No full-vocabulary logits need be cached.
        """
        h = repaired_hidden.detach()
        stats = post_repair_statistics.detach()
        if stats.shape != h.shape[:2]+(6,) or not torch.isfinite(h).all() or not torch.isfinite(stats).all():
            raise ValueError('invalid post-repair feature cache row')
        t = teacher_control.detach()
        if t.shape != (h.shape[0], 128) or not torch.isfinite(t).all():
            raise ValueError('risk teacher control shape or finiteness differs')
        t = t if self.use_teacher else torch.zeros_like(t)
        hidden = self.hidden(h.to(self.hidden[0].weight))
        t = t.to(hidden)[:, None, :].expand(-1, hidden.shape[1], -1)
        return self.net(torch.cat((hidden, stats.to(hidden), t), -1)).squeeze(-1)


def first_rejection_labels(candidates, target, *, eos=151645):
    """0=survived,1=first rejection,mask suffix; short/EOS ends are censored."""
    if not candidates or not target or len(candidates) > 8:
        raise ValueError('nonempty actual candidate and Target reference needed')
    if any(type(x) is not int or x < 0 for x in list(candidates)+list(target)) or eos in candidates[:-1]:
        raise ValueError('invalid token IDs or candidates continue after EOS')
    labels = [0.0] * len(candidates)
    valid = [False] * len(candidates)
    rejection = None
    observed = 0
    for j, (candidate, expected) in enumerate(zip(candidates, target)):
        valid[j] = True
        observed = j+1
        if candidate != expected:
            labels[j] = 1.0
            rejection = j+1
            break
        if candidate == eos:
            break
    if rejection is None and observed < len(candidates) and candidates[observed-1] != eos:
        raise ValueError('incomplete Target reference, not an observed complete rejection trace')
    return dict(labels=labels, valid=valid, first_rejection_one_based=rejection,
                observed_length=observed, right_censored=rejection is None)


def hazard_loss(logits, labels, valid):
    if logits.shape != labels.shape or labels.shape != valid.shape or valid.dtype != torch.bool or not valid.any():
        raise ValueError('hazard target/mask shape differs or empty')
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all() or not ((labels==0)|(labels==1)).all():
        raise ValueError('invalid hazard logits/labels')
    return F.binary_cross_entropy_with_logits(logits[valid].float(), labels[valid].float())


def choose_upload_length(hazard_probabilities, costs_seconds, *, maximum=8):
    """Cost-aware synchronous baseline, not a claim of globally optimal RL.

    Each candidate L has a measured/predicted entire round cost. In a no-EOS
    round with one Target correction/bonus, E[committed]=1+sum survival(j).
    No future acceptance label or future queue truth is consumed.
    """
    if (not 1 <= len(hazard_probabilities) <= maximum
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in hazard_probabilities)):
        raise ValueError('invalid conditional hazard curve')
    survival, expected, table = 1.0, 1.0, {}
    for j, p in enumerate(hazard_probabilities, 1):
        survival *= 1-p
        expected += survival
        if j in costs_seconds:
            cost = costs_seconds[j]
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError('round cost must be finite positive seconds')
            table[j] = dict(expected_committed=expected, cost_seconds=cost, tokens_per_second=expected/cost)
    if not table:
        raise ValueError('no supported upload length')
    best = max(table, key=lambda k: (table[k]['tokens_per_second'], -k))
    return best, table


def continue_one_more(observed_hazards, predicted_next_hazard, cost_send_now, cost_after_one_more, *, maximum=8):
    """Online one-step baseline: no features of ungenerated future tokens.

    predicted_next_hazard is a forecast available NOW (e.g. a calibrated
    persistence forecast), not a label or a hazard scored after drafting a full
    future block. Its estimation error must be evaluated separately.
    """
    if len(observed_hazards) >= maximum:
        return False
    length = len(observed_hazards)
    if length < 1:
        raise ValueError('generate at least one candidate before this decision')
    curve = list(observed_hazards)+[predicted_next_hazard]
    best, _ = choose_upload_length(curve, {length:cost_send_now, length+1:cost_after_one_more}, maximum=maximum)
    return best == length+1
