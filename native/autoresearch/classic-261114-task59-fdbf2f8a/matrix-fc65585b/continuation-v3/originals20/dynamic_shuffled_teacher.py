"""Select and bind the frozen Task34 donor at each realized repair round."""
from dataclasses import dataclass, replace
import json
from types import SimpleNamespace

from verify_donor_roster import PAIR_SHA, SCOPE, assignments, canonical, digest

REFERENCE_MAP_SHA = "ad648be5602ab0d10a8af11f041ac7e8bccb3443a840da18c1f2b870bf861956"


@dataclass(frozen=True)
class DonorPool:
    selected: dict
    prototypes: dict
    metadata: dict
    lock: dict

    @property
    def sha256(self):
        return digest(canonical(self.lock))


def freeze_pool(requests, dev_windows, protocol, pairing_bytes):
    """Build the assignment and payload lock from the actual frozen dev split."""
    if digest(pairing_bytes) != PAIR_SHA:
        raise ValueError("frozen Task34 pairing manifest differs")
    fixed = json.loads(pairing_bytes)["splits"]["val"]
    expected = {row["destination_window_id"]: row["destination_family_id"] for row in fixed}
    if len(requests) != 200 or len(dev_windows) != 1598 or not len(expected) == len(fixed) == 1598:
        raise ValueError("dynamic Task34 roster has wrong coverage")
    prototypes, metadata = {}, {}
    for window in dev_windows:
        donor_id = protocol.b.previous.causal_window_id(window)
        source, message = protocol.message(window, "val", "matched")
        if (donor_id in prototypes or expected.get(donor_id) != source.family_id
                or requests[source.prompt_id]["family_id"] != source.family_id
                or source.split != "dev" or message.codec != "bf16le-v1"
                or len(message.payload) != 8192):
            raise ValueError(f"invalid frozen donor: {donor_id}")
        prototypes[donor_id] = message
        metadata[donor_id] = {
            "donor_id": donor_id, "prompt_id": source.prompt_id,
            "family_id": source.family_id, "split": source.split,
            "source_round_index": source.round_index,
            "source_prefix_sha256": message.source_prefix_sha256,
            "source_corrected_prefix_sha256": source.round_key[2],
            "target_model_sha256": source.target_model_sha256,
            "payload_sha256": digest(message.payload),
        }
    if metadata.keys() != expected.keys():
        raise ValueError("frozen Task34 donor IDs differ")
    recipients = {prompt_id: {"prompt_id": prompt_id, "family_id": request["family_id"], "split": "dev"}
                  for prompt_id, request in requests.items()}
    selected = {prompt_id: assignments(recipient, metadata.values())
                for prompt_id, recipient in recipients.items()}
    selected_sha = digest(canonical(selected))
    if selected_sha != REFERENCE_MAP_SHA:
        raise ValueError("actual dynamic Task34 assignment differs from the frozen reference")
    lock = {"scope": SCOPE, "historical_pairing_sha256": PAIR_SHA,
            "assignment_sha256": selected_sha, "donor_metadata_sha256": digest(canonical(metadata)),
            "recipient_roster_sha256": digest(canonical(recipients))}
    return DonorPool(selected, prototypes, metadata, lock)


class DynamicShuffledTeacher:
    def __init__(self, head, protocol, pool):
        self.head = head
        self.protocol = protocol
        self.pool = pool
        self.context = None
        self.message = None
        self.round_receipts = []
        self.pairing_sha = pool.sha256

    def prepare_observation(self, observation, draft):
        context = self.protocol.context(SimpleNamespace(observation=observation), "val")
        ordinal = context.round_index + 1
        selected = self.pool.selected[context.prompt_id]
        if ordinal != len(self.round_receipts) + 1 or not 1 <= ordinal <= len(selected):
            raise ValueError("nonsequential or out-of-range dynamic Task34 repair round")
        donor_id = selected[ordinal - 1]
        donor = self.pool.metadata[donor_id]
        message = replace(self.pool.prototypes[donor_id], destination_round_key=context.round_key,
                          pairing_lock_sha256=self.pairing_sha)
        message.validate(context, shuffled=True, feature_dim=4096)
        payload_sha = digest(message.payload)
        if payload_sha != donor["payload_sha256"] or len(message.payload) != 8192:
            raise ValueError("dynamic Task34 donor payload differs")
        state = self.head.prepare_observation(observation, draft)
        self.context, self.message = context, message
        self.round_receipts.append({
            "round_key": context.round_key, "next_round_index": ordinal,
            "valid_prefix_sha256": digest(canonical(list(context.valid_prefix))),
            "corrected_prefix_sha256": context.round_key[2],
            "target_model_sha256": context.target_model_sha256,
            "teacher_source": "frozen_shuffled_dev_donor", "teacher_payload_bytes": len(message.payload),
            "teacher_payload_sha256": payload_sha, "pairing_lock_sha256": self.pairing_sha,
            "donor": donor,
        })
        return state

    def next_token(self, prefix, state, position, draft):
        return self.head.next_token(prefix, state, position, draft,
                                    context=self.context, message=self.message)
