"""Install the native-logit revision beside the historical run, with a separate freeze and receipts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def packed(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("existing native-logit installation differs: " + str(path))
        return
    descriptor, name = tempfile.mkstemp(dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("concurrent native-logit installation differs: " + str(path))
    finally:
        temporary.unlink(missing_ok=True)


def sources(repo):
    lock = json.loads((repo / "exact-runtime.json").read_bytes())
    result = {}
    for name, expected in lock.items():
        data = (repo / "exact_runtime" / name).read_bytes()
        if sha(data) != expected:
            raise ValueError("native-logit source differs: " + name)
        result[name] = data
    return result


def prepare(repo, legacy, freeze_bytes):
    overlays = sources(repo)
    new_root = legacy.with_name(legacy.name + "-native-logits-v1")
    freeze = json.loads(freeze_bytes)
    files = dict(freeze["files"])
    # Preserve the old file bindings. Copy only files from the original runtime
    # directory, including the externally mounted configs and CPU receipts.
    for original, expected in freeze["files"].items():
        path = Path(original)
        if not path.is_relative_to(legacy):
            continue
        name = path.relative_to(legacy).as_posix()
        if name not in overlays:
            data = path.read_bytes()
            if sha(data) != expected:
                raise ValueError("legacy runtime dependency changed: " + original)
            overlays[name] = data
    overlays["secondary_eval_adapter_config.json"] = packed({"source_sha256": sha(overlays["secondary_eval_adapter.py"])})
    for name, data in overlays.items():
        dest = new_root / name
        publish(dest, data)
        files[str(dest)] = sha(data)
    freeze.update(files=files, native_logit_format="native-logits-v1", parent_freeze_sha256=sha(freeze_bytes))
    publish(new_root / "runtime_freeze.json", packed(freeze))
    return new_root
