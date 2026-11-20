"""Offline regression for the measured BF16 mismatch and the exact gate-to-training path."""
import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent
LEGACY = next((REPO / "native").rglob("continuation-v5"))
sys.path[:0] = [str(REPO / "exact_runtime"), str(LEGACY)]
if os.name == "nt":
    sys.modules.setdefault("fcntl", SimpleNamespace())
import fkl_all_verified_gpu as worker
import fkl_shards
import full_vocab_supervision as supervision
import native_logits as codec
import exact_layout
from guards import SharedIntegrityError

torch.set_num_threads(1)


def measured_pair():
    # Only id 1505 and the row maximum are from the A100 incident. The
    # second changed index is synthetic; the incident did not report its ID.
    native = torch.zeros(codec.VOCAB, dtype=torch.bfloat16)
    native[:40] = torch.arange(40, dtype=torch.bfloat16)
    native[1505] = -0.0189208984375
    native[2000] = 8
    rebuilt = native.clone()
    rebuilt[1505] = -0.018798828125
    rebuilt[2000] += 0.0625
    return native, rebuilt


class ExactTeacher(unittest.TestCase):
    def test_revision_layout_preserves_blocked_gate_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            legacy=base/"continuation-v5";legacy.mkdir()
            code=legacy/"common.py";code.write_bytes(b"original")
            config=legacy/"configs/g.json";worker.write(config,{"group":"fixture"})
            blocked=legacy/"fkl/gate/completion.json";worker.write(blocked,{"status":"blocked"})
            freeze=exact_layout.packed({"files":{str(p):worker.sha(p) for p in (code,config)}})
            revision=exact_layout.prepare(REPO,legacy,freeze)
            self.assertNotEqual(revision,legacy)
            self.assertEqual(worker.read(blocked),{"status":"blocked"})
            self.assertFalse((revision/"fkl/gate/completion.json").exists())
            self.assertEqual(worker.sha(revision/"configs/g.json"),worker.sha(config))
            self.assertEqual(exact_layout.prepare(REPO,legacy,freeze),revision)
            for path,checksum in worker.read(revision/"runtime_freeze.json")["files"].items():
                self.assertEqual(worker.sha(path),checksum)
            (revision/"native_logits.py").write_bytes(b"different")
            with self.assertRaises(ValueError):exact_layout.prepare(REPO,legacy,freeze)

    def test_modes_and_failure_paths(self):
        native, rebuilt = measured_pair()
        self.assertEqual(float((native-rebuilt).abs().max()),0.0625)
        self.assertEqual(native.argmax(),rebuilt.argmax())
        for reference, projection, mode in [(native,native,"hidden"),(native,rebuilt,"sparse"),
                                             (native+1,rebuilt,"dense")]:
            info,data=codec.encode("p",reference,projection)
            self.assertEqual(info["mode"],mode)
            self.assertTrue(torch.equal(codec.decode("p",info,data,projection),reference.float()))
            with self.assertRaises(SharedIntegrityError):
                codec.decode("p",dict(info,native_sha256="0"*64),data,projection)
        info,data=codec.encode("p",native,rebuilt)
        for key in ("p.ids","p.values"):
            changed=copy.deepcopy(data)
            changed[key][0]=changed[key][1]
            with self.assertRaises(SharedIntegrityError):
                codec.decode("p",info,changed,rebuilt)
        changed=rebuilt.clone()
        changed[70]=1
        with self.assertRaises(SharedIntegrityError):
            codec.decode("p",info,data,changed)
        native[0]=float("nan")
        with self.assertRaises(SharedIntegrityError):codec.encode("p",native,rebuilt)

    def test_writer_disk_gate_backfill_training_and_fkl(self):
        native,rebuilt=measured_pair()
        saved=torch.zeros(4096,dtype=torch.bfloat16)
        class FixtureHead:
            weight=torch.zeros(1,dtype=torch.bfloat16).expand(codec.VOCAB,4096)
            def __call__(self,h):
                assert torch.equal(h,saved.reshape(1,-1))
                return rebuilt.reshape(1,-1)
        head=FixtureHead()
        device_model=SimpleNamespace(get_output_embeddings=lambda:head)
        target=SimpleNamespace(model=device_model,_ids=lambda x:x,
            _forward_with_final_hidden=lambda _: (SimpleNamespace(logits=native.reshape(1,1,-1)),saved.reshape(1,1,-1)),
            next=lambda _:SimpleNamespace(token_id=int(native.argmax()),hidden=saved))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frozen=root/"weights";frozen.write_bytes(b"test-head")
            worker.write(root/"runtime_freeze.json",{"files":{str(frozen):worker.sha(frozen)}})
            labels=root/"labels";worker.write(labels/"completion.json",{"fixture":True})
            trace=root/"trace.json";worker.write(trace,{"rounds":[{"tensor_prefix":"r","candidate_ids":[1505]}]})
            ids=native.float().topk(32).indices;features=root/"shards/r/features.safetensors"
            features.parent.mkdir(parents=True)
            raw={"r.top32_ids":ids[None],"r.top32_logits":native[ids][None],
                 "r.logsumexp":native.float().logsumexp(0).reshape(1),"r.candidate_valid":torch.tensor([True]),
                 "r.candidate_logits":native[1505].reshape(1)}
            save_file(raw,str(features))
            key=sha256(b"fixture-prefix").hexdigest()
            job=dict(prefix_id=key,request_id="r",split="train",confirmed_output_tokens=0,
                     prefix_sha256=sha256(b"[1]").hexdigest(),expected_id=int(native.argmax()),raw_round=0,raw_decision_row=0)
            jobs={key:job}
            requests={"r":dict(request_id="r",prompt_ids=[1],output_ids=[],trace_path=str(trace),trace_sha256=worker.sha(trace))}
            (labels/"all_occurrences.jsonl").write_text(json.dumps(dict(request_id="r",window_index=0,prefix_id=key,expected_id=job["expected_id"]))+"\n")
            ctx={"logit_format":codec.FORMAT,"codec_sha256":worker.sha(Path(codec.__file__))}
            import contextlib
            with contextlib.ExitStack() as stack:
                for mod,keyname,value in [(worker,"INPUT",labels),(worker,"DATA",root),(supervision,"ROOT",root),
                                          (supervision,"V3",root),(worker,"strict_context",lambda:ctx),
                                          (supervision,"coverage_inputs",lambda _:(jobs,{"mapping_receipt_sha256":worker.sha(labels/"completion.json")}))]:
                    stack.enter_context(patch.object(mod,keyname,value))
                if os.name=="nt":
                    stack.enter_context(patch.object(fkl_shards,"sync_dir",lambda _:None))
                    def sync_windows(path):
                        with open(path,"r+b") as stream:os.fsync(stream.fileno())
                    stack.enter_context(patch.object(fkl_shards,"sync_file",sync_windows))
                for phase in ("gate","backfill"):
                    out=root/"fkl"/phase;out.mkdir(parents=True)
                    worker.run_target(phase,out,requests,[job],ctx,target)
                report=worker.read(root/"fkl/gate/native_roundtrip.json")
                self.assertEqual(report["rows"][0]["max_abs_error"],0.0625)
                self.assertEqual(report["rows"][0]["replaced"],2)
                # Production requires all 531692 prefixes. Scope this one-prefix
                # reader fixture explicitly; never treat its terminal as a GPU receipt.
                finish=root/"fkl/backfill/completion.json"
                rec=worker.read(finish)
                self.assertFalse(rec["FKL_training_admitted"])
                worker.write(finish,dict(rec,FKL_training_admitted=True))
                reader=supervision.FullVocabularySupervision(head,dict(source_path=str(frozen),source_sha256=worker.sha(frozen)))
                got=reader.logits("r",0,torch.tensor([job["expected_id"]]))[0,0]
                self.assertTrue(torch.equal(got,native.float()))
                logits=torch.zeros_like(got,requires_grad=True)
                loss=torch.nn.functional.kl_div(logits.log_softmax(-1),got.log_softmax(-1),log_target=True,reduction="sum")
                loss.backward()
                torch.testing.assert_close(logits.grad,logits.softmax(-1)-native.float().softmax(-1),rtol=2e-5,atol=2e-7)
                raw["r.candidate_logits"]=torch.tensor([12.])
                save_file(raw,str(features))
                wrong=root/"bad_raw";wrong.mkdir()
                with self.assertRaises(worker.Mismatch):
                    worker.run_target("gate",wrong,requests,[job],ctx,target)
                self.assertFalse((wrong/"completion.json").exists())

if __name__=="__main__":
    unittest.main(verbosity=2)
