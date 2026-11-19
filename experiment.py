"""Run one frozen legacy FKL experiment in the original /data layout."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
DATA = Path("/data")
V5 = DATA / "autoresearch/classic-261114-task59-fdbf2f8a/matrix-fc65585b/continuation-v5"
WAVE = V5.parents[1] / "wave2-dev503-20261114"


def read_json(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(name):
    path = PurePosixPath(name).relative_to("/data")
    if ".." in path.parts:
        raise ValueError(f"invalid input path {name}")
    return Path(*path.parts)


def install(files):
    for name, checksum in files.items():
        relative = relative_path(name)
        src = HERE / "native" / relative
        if not src.is_file() or digest(src) != checksum:
            raise ValueError(f"invalid repository member {relative}")
        dst = DATA / relative
        if dst.exists():
            if digest(dst) != checksum:
                raise ValueError(f"existing file has a different digest {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Publish complete bytes without replacing a concurrent install.
            fd, temporary = tempfile.mkstemp(dir=dst.parent, prefix=dst.name + ".")
            try:
                with os.fdopen(fd, "wb") as output, src.open("rb") as source:
                    shutil.copyfileobj(source, output)
                if digest(Path(temporary)) != checksum:
                    raise ValueError(f"copied file differs {relative}")
                os.chmod(temporary, src.stat().st_mode & 0o777)
                try:
                    os.link(temporary, dst)
                except FileExistsError:
                    if digest(dst) != checksum:
                        raise ValueError(f"existing file has a different digest {dst}")
            finally:
                Path(temporary).unlink(missing_ok=True)
    # The frozen worker opens this lock with "a+"; its parent is not a frozen input.
    (DATA / "a100-first-rejection-20261111-9b21f68f").mkdir(parents=True, exist_ok=True)


def preflight(files, verify_hashes):
    required = {DATA / relative_path(path): checksum for path, checksum in files.items()}
    required.update({WAVE / "completion.json": None})
    for stage in ("six_dev", "ack_training", "ack_dev"):
        required[WAVE / stage / "completion.json"] = None
        required[WAVE / stage / "gpu_lease.json"] = None
    for path in (V5 / "runtime_freeze.json", V5 / "CPU_E2E_receipt.json"):
        required.setdefault(path, None)
    missing = [path for path in required if not path.is_file()]
    if missing:
        for path in missing[:15]:
            print(f"missing {path}")
        counts = Counter(str(path.relative_to(DATA).parts[0]) for path in missing)
        raise ValueError(f"{len(missing)} missing inputs by source directory: {dict(counts)}")
    if verify_hashes:
        for path, checksum in required.items():
            if checksum and digest(path) != checksum:
                raise ValueError(f"input digest differs {path}")
    print(f"{len(required)} input paths passed {'existence and digest' if verify_hashes else 'existence'} checks")


def main():
    cfg = read_json(HERE / "experiment.json")
    parser = argparse.ArgumentParser(description=cfg["group"])
    parser.add_argument("phase", choices=("check-repo", "install", "preflight", "gate", "backfill", "smoke", "train", "eval"))
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    native = read_json(HERE / "bundled-files.json")
    if args.phase in ("check-repo", "install"):
        for name, hash_value in native.items():
            if digest(HERE / "native" / relative_path(name)) != hash_value:
                raise ValueError(f"repository file differs {name}")
        print(f"{len(native)} bundled files passed byte verification")
        if args.phase == "install":
            install(native)
        return
    freeze = read_json(HERE / "native" / V5.relative_to(DATA) / "runtime_freeze.json")
    preflight({**freeze["files"], **native}, args.verify_hashes)
    for name, expected in native.items():
        if (name not in freeze["files"] or name.endswith("/CPU_E2E_receipt.json")) and digest(DATA / relative_path(name)) != expected:
            raise ValueError(f"installed execution receipt differs {name}")
    if args.phase == "preflight":
        return
    environment = dict(os.environ, PYTHONPATH=str(DATA / "Spec_v2-main"))
    if args.phase in ("gate", "backfill"):
        command = [str(V5 / "fkl_all_verified_gpu.py"), "--phase", args.phase]
    elif args.phase == "eval":
        adapter = V5.parents[1] / "standalone-single-eval" / "secondary_eval_adapter.py"
        command = [str(adapter), "--group", cfg["group"]]
    else:
        command = [str(V5 / "worker20.py"), "--group", cfg["group"], "--phase", args.phase]
    subprocess.run([sys.executable, "-B", "-u", *command], cwd=V5, env=environment, check=True)


if __name__ == "__main__":
    main()
