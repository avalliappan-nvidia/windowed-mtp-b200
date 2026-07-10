#!/usr/bin/env python3
"""Build a lightweight YaRN-prepared model directory.

The two Qwen checkpoints ship a native 262,144-token context. To reach ~1.05M we
set a YaRN rope_scaling factor of 4. These are *multimodal* configs (model_type
qwen3_5_moe) whose text model is nested under `text_config`. SGLang's runtime
`json_model_override_args` shallow-replaces `text_config`, which drops
`num_attention_heads` and crashes with:
    assert hasattr(config.text_config, "num_attention_heads")  -> AssertionError
So we bake the edit on disk instead (exactly what the paper's runs used):

  * symlink every file from the HF snapshot into <dst> (no weight duplication),
  * write a REAL config.json whose text-model rope is switched to YaRN factor N,
    preserving all other rope fields (mrope_*, partial_rotary_factor, rope_theta)
    and raising max_position_embeddings to orig*factor.

Usage:
  python src/prep_yarn_model.py --src <HF repo id | snapshot dir> --dst <out dir> \
      [--factor 4.0] [--orig 262144]
"""
import argparse
import json
import os
from pathlib import Path


def resolve_snapshot(src: str):
    """Return (real_snapshot_dir, link_snapshot_dir).

    real_ is always a listable path (used to read files). link_ is the path we
    point symlinks at. For a downloaded repo we prefer the snapshot *under
    HF_HOME* over snapshot_download()'s realpath, because on many clusters the
    HF cache lives on a filesystem exposed at two paths (e.g. /lustre and
    /scratch) and realpath() canonicalizes to the one that is NOT bind-mounted
    into the container -> the symlinks dangle inside the container and the model
    fails to load. Keeping them under HF_HOME (which run_point.sh mounts 1:1)
    plus emitting *relative* symlinks makes the prepared dir mount-agnostic.
    """
    if os.path.isdir(src):
        real = os.path.realpath(src)
        return real, real
    from huggingface_hub import snapshot_download
    real = snapshot_download(src, local_files_only=True)  # may canonicalize to /scratch
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        h = os.path.basename(real.rstrip("/"))
        org, _, name = src.partition("/")
        repo_dir = f"models--{org}--{name}" if name else f"models--{org}"
        cand = os.path.join(hf_home, "hub", repo_dir, "snapshots", h)
        if os.path.isdir(cand):
            return real, cand
    return real, real


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF repo id or local snapshot dir")
    ap.add_argument("--dst", required=True, help="output prepared model dir")
    ap.add_argument("--factor", type=float, default=4.0)
    ap.add_argument("--orig", type=int, default=262144)
    args = ap.parse_args()

    real_snap, link_snap = resolve_snapshot(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Symlink every entry except config.json (which we materialize as a real
    # file). Use RELATIVE symlinks into link_snap so the dir is mount-agnostic
    # (resolves under whatever path the repo is bind-mounted at).
    for name in os.listdir(real_snap):
        if name == "config.json":
            continue
        link = dst / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(os.path.join(link_snap, name), dst), link)

    with open(os.path.join(real_snap, "config.json")) as f:
        cfg = json.load(f)

    ext = int(args.orig * args.factor)
    # Target the text model: nested text_config (multimodal) or the top level.
    tgt = cfg["text_config"] if isinstance(cfg.get("text_config"), dict) else cfg
    # Start from whatever rope block exists so we keep mrope_*, partial_rotary_factor,
    # rope_theta, etc., and only switch the type to yarn.
    rope = dict(tgt.get("rope_parameters") or tgt.get("rope_scaling") or {})
    rope.update({
        "rope_type": "yarn",
        "factor": args.factor,
        "original_max_position_embeddings": args.orig,
    })
    tgt["rope_parameters"] = rope
    tgt["max_position_embeddings"] = ext

    with open(dst / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[prep-yarn] {args.src}")
    print(f"[prep-yarn]   -> {dst}")
    print(f"[prep-yarn]   max_position_embeddings={ext}  rope={rope}")


if __name__ == "__main__":
    main()
