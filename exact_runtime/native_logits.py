"""Lossless replacements for the native full-prefix Target logits."""
import hashlib

import torch
from guards import require


FORMAT = "native-logits-v1"
VOCAB = 151936


def fingerprint(logits):
    data = logits.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def encode(key, native, reconstructed):
    native = native.detach().float().contiguous()
    reconstructed = reconstructed.detach().float().contiguous()
    require(native.ndim == 1 and native.shape == reconstructed.shape, "teacher vocabulary differs")
    require(bool(native.isfinite().all()) and bool(reconstructed.isfinite().all()), "nonfinite teacher logits")
    indices = (native.view(torch.int32) != reconstructed.view(torch.int32)).nonzero().flatten()
    count = indices.numel()
    info = dict(format=FORMAT, mode="hidden", vocabulary=native.numel(), replaced=count,
                max_abs_error=float((native - reconstructed).abs().max()), native_sha256=fingerprint(native))
    tensors = {}
    if count:
        values = native[indices]
        compact = values.bfloat16()
        dtype = torch.bfloat16 if torch.equal(compact.float(), values) else torch.float32
        if count * (4 + torch.empty((),dtype=dtype).element_size()) < native.numel() * 2:
            info["mode"] = "sparse"
            tensors[key + ".ids"] = indices.cpu().int()
            tensors[key + ".values"] = values.to(device="cpu", dtype=dtype)
        else:
            info["mode"] = "dense"
            compact = native.bfloat16()
            tensors[key + ".values"] = (compact if torch.equal(compact.float(), native) else native).cpu()
    return info, tensors


def tensor_names(key, mode):
    require(mode in ("hidden", "sparse", "dense"), "unknown native-logit representation")
    if mode == "sparse":
        return [key + ".ids", key + ".values"]
    return [key + ".values"] if mode == "dense" else []


def validate(key, info, get, vocab):
    require(info["format"] == FORMAT and info["vocabulary"] == vocab, "wrong native-logit format or vocabulary")
    count = info["replaced"]
    require(type(count) is int and 0 <= count <= vocab, "invalid replacement count")
    mode = info["mode"]
    tensor_names(key, mode)
    require((count == 0) == (mode == "hidden"), "invalid native-logit payload mode")
    require(isinstance(info["native_sha256"], str) and len(info["native_sha256"]) == 64, "missing native-logit digest")
    if mode != "hidden":
        values = get(key + ".values")
        require(values.dtype in (torch.bfloat16, torch.float32) and values.shape == (count if mode == "sparse" else vocab,),
                "bad native-logit values")
        require(bool(torch.isfinite(values).all()), "nonfinite native-logit values")
    if mode == "sparse":
        ids = get(key + ".ids")
        require(ids.dtype == torch.int32 and ids.shape == (count,) and 0 <= int(ids[0]) and int(ids[-1]) < vocab
                and bool((ids[1:] > ids[:-1]).all()), "invalid native-logit indices")


def decode(key, info, tensors, reconstructed):
    vocab = reconstructed.numel()
    validate(key, info, tensors.__getitem__, vocab)
    result = reconstructed.detach().float().reshape(-1)
    if info["mode"] == "dense":
        result = tensors[key + ".values"].to(result.device, torch.float32)
    elif info["mode"] == "sparse":
        ids = tensors[key + ".ids"].to(result.device, torch.long)
        result = result.clone()
        result[ids] = tensors[key + ".values"].to(result.device, torch.float32)
    require(fingerprint(result) == info["native_sha256"], "native teacher logit digest differs after reconstruction")
    return result
