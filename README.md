# Windowed-MTP — B200 reproduction package

[![arXiv](https://img.shields.io/badge/arXiv-2607.21535-b31b1b.svg)](https://arxiv.org/abs/2607.21535)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21522901.svg)](https://doi.org/10.5281/zenodo.21522901)

This artifact reproduces the **core mechanism** of the paper: at 1M-token
context, replacing the multi-token-prediction (MTP) draft's full-KV attention
with a small **sliding window + attention sink** (a compact draft-KV **ring
pool**, plus a numerics-neutral **num_stages** kernel tweak) cuts the
**per-decode-step cost** by **+28% to +44%** across three long-context
architectures, while **preserving acceptance length (AL)**.

It runs this on two **seeded, synthetic RULER** workloads that are built entirely
in-repo (no dataset license, no network, byte-reproducible on any Python 3.8+):

- **`niah_multiquery_enum`** — retrieve-and-list 48 magic numbers from a 1M-token
  haystack (structured long answer). *The paper's headline input.*
- **`fwe_enum`** — list the 30 most frequently occurring words (aggregation).

> The paper's *end-to-end* tok/s figures are the per-step cost win compounded over
> a full generation; end-to-end additionally moves with the per-input acceptance
> length. The cleanest, input-invariant claim — and the one this artifact
> targets — is the **per-decode-step cost reduction**. The end-to-end marquee task
> (**RULER CWE**) is one command away (`INPUTS=cwe`, see *Extending*).

The mechanism is a ~180-line diff plus a few drop-in files on top of stock
`sglang:26.06`. No draft model is trained: the MTP/NEXTN head ships inside each
target checkpoint (self-speculation). Windowing is toggled entirely by
environment variables (`RK_MTP_WINDOW`, `RK_MTP_SINK`, `RK_DRAFT_RING`,
`RK_EXTEND_STAGES`, `RK_DECODE_STAGES`).

## What you need

- **1× NVIDIA B200** (the paper's headline is single-GPU, TP1). H100 also works
  at smaller context; 1M context needs the B200's HBM.
- A container runtime: **Docker** (`--gpus all`) *or* **enroot/pyxis + SLURM**.
  Set `CONTAINER_ENGINE` in `config.env` accordingly (auto-detected by default).
- **Host Python ≥ 3.8** (for the setup venv, input generation, and result
  parsing only; the benchmark itself runs inside the container). `python3 -m venv`
  must be available (Debian/Ubuntu: `apt install python3-venv`).
- ~**300–400 GB** free disk for the three NVFP4 checkpoints + HF cache.
- Network access to `nvcr.io` and HuggingFace (first run only). The **inputs need
  no network** — they are synthesized locally.

### Where to run

Split the work in two: **setup** needs only network + disk; the **timed runs**
need the GPU.

- **`setup/*.sh`** (venv, image, models, inputs, patch): run wherever is
  convenient — a login node is fine.
- **`run/*.sh`** (the actual benchmark): where you launch these depends on the
  container engine.

**Docker** (`CONTAINER_ENGINE=docker`): run `run/*.sh` **on the GPU box itself**;
`docker run --gpus all` needs the GPUs local.

**SLURM + enroot/pyxis** (`CONTAINER_ENGINE=enroot`): every cluster has its own
**account / partition / GRES** flags that this package cannot guess. So **get an
allocation yourself first with `salloc`, then run the script inside it** — an
`srun` step launched within an allocation inherits those settings automatically,
so the scripts stay cluster-agnostic:

```bash
# 1. Grab a GPU using YOUR cluster's flags (account/partition/gres vary):
salloc -A <account> -p <b200-partition> --gres=gpu:1 -N1 -t 04:00:00

# 2. salloc holds the allocation and gives you a shell that is STILL ON THE
#    LOGIN/SUBMIT NODE (not the GPU node). Run the script from that shell: the
#    srun inside run/*.sh dispatches the actual work onto the allocated GPU node
#    (inheriting account/partition/GRES). You never need to ssh to the GPU node.
bash run/run_headline.sh
#   or a single point:
bash run/run_point.sh q35 nextn 1 1040000 4032 1 mqe
```

`run/*.sh` refuse to start under `enroot` with **no allocation** (they print the
`salloc` hint above and exit) rather than emitting a cryptic
`You must specify an account` error or silently queueing. You only need the
allocation for step 4 (the runs); steps 0–3 can be done on the login node.

## Quick start

```bash
# 0. Configure paths / engine. Edit config.env if defaults don't suit you.
#    CONTAINER_ENGINE auto-detects docker, else enroot.
cat config.env

# 0b. Create a host-side Python venv (HF CLI). Avoids the PEP 668
#     "externally-managed-environment" pip error on modern distros. Run once;
#     all later setup/*.sh auto-activate it via config.env.
bash setup/00_venv.sh

# 1. Get the SGLang 26.06 container image.
bash setup/01_get_image.sh

# 2. Download the 3 checkpoints + synthesize the RULER corpora (~300 GB models;
#    inputs are generated locally in seconds and verified against
#    setup/input_checksums.sha256).
bash setup/02_get_models.sh

# 3. Build the patched SGLang tree (extract from image, apply windowing patch +
#    num_stages kernels).
bash setup/03_apply_patch.sh

# 4. Reproduce the headline: native MTP vs Windowed-MTP, 1M ctx, all 3 models,
#    on each workload in $INPUTS (default: niah_multiquery_enum fwe_enum).
#    3 models × 2 inputs × 2 points = 12 runs; each = one 1M-token prefill +
#    timed decode. Budget ~2–4 h.
#    SLURM/enroot users: run this INSIDE an salloc allocation (see "Where to run").
bash run/run_headline.sh
#    (One workload only? e.g.  INPUTS=fwe_enum bash run/run_headline.sh)
```

Step 4 prints, **per input**, a summary table of TPOT, acceptance length (AL),
per-decode-step cost, and the per-step **win%** for each model.

## Expected results

`run/run_headline.sh` runs native MTP vs Windowed-MTP (d=7, γ=6, 1M context,
B=1, bf16 KV, single B200) for every model on every input in `$INPUTS`, and
prints, **per input**:

- `step` = one speculative-iteration cost `= TPOT_gt × mean_AL` (ms)
- `win%` = `step_native / step_win − 1` (the paper's convention: how much faster
  one decode step is under windowing)

What this artifact reproduces — AL is **preserved (within ≈±10%)** by windowing,
and the per-decode-step cost drops by **+28% to +44%**, largest for the MoE-FFN
Qwen-35B draft and smallest for Nemotron's Mamba2 draft:

**`niah_multiquery_enum` (d=7):**

| Model                       | AL (native → win) | step ms (native → win) | win%     |
|-----------------------------|-------------------|------------------------|----------|
| Qwen3.6-35B-A3B (GDN-MoE)   | 4.49 → 4.74       | 26.7 → 18.4            | **+44%** |
| Qwen3.5-122B-A10B (GDN-MoE) | 5.57 → 4.74       | 33.9 → 25.9            | **+31%** |
| Nemotron-3-120B (Mamba2)    | 3.75 → 3.61       | 32.8 → 25.8            | **+27%** |

**`fwe_enum` (d=7):** the same band (q35 ≈ +40%, q122 ≈ +28%, nem ≈ +26%).

(Exact numbers are printed by the script and vary a little run-to-run; the values
above are the paper's reference points. The single-point `step = TPOT×AL` tracks
the paper's multi-depth latency fit to within ~1 pt.)

**End-to-end** decode speedup (native-TPOT / win-TPOT) at the same point is
**1.5×** (q35), **1.1×** (q122), **1.2×** (nem) on `niah_multiquery_enum`;
end-to-end also moves with the per-input AL (windowing trades a little AL on
q122 retrieval, which the cost-side win more than covers).

**Not reproduced by the default sweep:** the Qwen CWE points (`tab:bestdepth`/`tab:hero`;
Qwen35 **1.53×**, Qwen122 **1.48×**). Add them with `INPUTS=cwe bash run/run_headline.sh`
(corpus built by step 2) — see *Extending*.

### Inputs & determinism

The workloads are **synthesized from a seeded, in-repo generator**
(`src/gen_ruler.py`), not shipped as files. `setup/02_get_models.sh` builds:

- `data/ruler_qa_1m/niah_multiquery_enum_1150000.jsonl` — 48-key NIAH retrieval.
- `data/ruler_qa_1m/fwe_enum_1150000.jsonl` — top-30 frequent-word extraction.
- `data/ruler_qa_1m/cwe_1150000.jsonl` — common-word extraction (marquee task).

`gen_ruler.py` is a dependency-free re-implementation of the RULER task families
(Hsieh et al., 2024). Its RNG is seeded per (task, length), so output is
**byte-identical on any Python 3.8+** — each file is verified against
`setup/input_checksums.sha256` (the paper's exact hashes).

All tasks run in **QA mode** (`ignore_eos=False`, `--qa-max-new 512`), with a
**repetition penalty of 1.15** applied *identically* to the draft and target
logits — so speculation stays lossless w.r.t. the penalized distribution while
the greedy repetition loops that otherwise inflate AL on long decodes are broken.

## Reproducing one point manually

```bash
# Windowed-MTP, Qwen3.6-35B, 1M ctx, batch 1, window 4032 + ring, mqe input:
run/run_point.sh q35 nextn 1 1040000 4032 1 mqe
# Native MTP baseline (window off):
run/run_point.sh q35 nextn 1 1040000 0 0 mqe
# Dense (non-speculative) reference:
run/run_point.sh q35 none  1 1040000 0 0 mqe
# fwe_enum aggregation task:
run/run_point.sh q35 nextn 1 1040000 4032 1 fwe
# Any other corpus: pass a jsonl path straight to --real-input:
run/run_point.sh q35 nextn 1 1040000 4032 1 /path/to/some.jsonl
```

Model keys: `q35`, `q122`, `nemo` (or pass a full HF id). Args:
`MODEL ALGO B ISL W RING [INPUT]` where
`INPUT ∈ random | mqe | fwe | cwe | <path.jsonl>`.
See the header of `run/run_point.sh` for all env overrides.

## Directory layout

```
b200/
├── README.md              # this file
├── LICENSE                # Apache-2.0 (this package's own code)
├── NOTICE                 # third-party software / dataset / model attributions
├── config.env             # paths, image tag, model ids, engine choice
├── setup/
│   ├── 00_venv.sh          # create host venv (.venv) for HF CLI (PEP 668)
│   ├── 01_get_image.sh     # docker pull / enroot import sglang:26.06
│   ├── 02_get_models.sh    # HF download 3 checkpoints + synth RULER corpora
│   ├── 03_apply_patch.sh   # extract sglang, apply patch, drop in new files
│   └── input_checksums.sha256  # reference SHA256 of the synthesized corpora
├── patch/
│   ├── windowed_mtp.patch  # diff vs stock sglang 26.06 (windowing + ring)
│   ├── ring_draft.py       # NEW: compact draft-KV ring pool
│   ├── attn_mass.py        # NEW: draft attention-mass probe (analysis, optional)
│   ├── extend_attention.py # num_stages-tunable triton extend kernel (drop-in)
│   └── decode_attention.py # num_stages-tunable triton decode kernel (drop-in)
├── src/
│   ├── bench_sglang_mtp.py # the benchmark harness (offline sgl.Engine)
│   ├── parse_gt.py         # extracts steady-state TPOT/AL from the scheduler log
│   ├── prep_yarn_model.py  # bakes YaRN factor 4 into a local Qwen model dir
│   └── gen_ruler.py        # seeded RULER generator (builds the headline inputs)
└── run/
    ├── run_point.sh        # one point in the container (docker or enroot)
    └── run_headline.sh     # native-vs-windowed sweep over models × inputs + summary
```

## How the measurement works

Each run does a tiny warmup, a cache flush, then **one** timed run = a full 1M
prefill plus a timed decode. The SGLang scheduler logs decode throughput every
step (`decode-log-interval 1`). Over the steady-state lines (`#running-req == B_eff`,
ramp dropped, `#full token: 0` prefill-flush artifacts removed) `parse_gt.py` computes the
per-iteration wall `T_iter = accept_len / throughput` per line, takes the **median** of
those `T_iter`s, and reports `TPOT = median(T_iter) / mean(AL)` (per-seq; ×`B_eff` for the
batch). Dividing by **mean** AL — rather than inverting the median throughput — avoids a
15–40% bias from `median(AL)` snapping to integer accept lengths. The per-decode-step cost
(one speculative iteration = γ draft forwards + verify) is `step = TPOT × mean_AL`.

## The contribution (what the patch does)

- `models/qwen3_5.py` — gate the MTP/NEXTN draft's RadixAttention on
  `RK_MTP_WINDOW` / `RK_MTP_SINK` (window + sink), leaving the target untouched.
- `layers/attention/{flashinfer,triton}_backend.py` — build a windowed
  draft-decode KV index (graph-safe); triton backend does the ring index remap.
- `speculative/eagle_info_v2.py` + `ring_draft.py` — the compact draft-KV **ring
  pool** so the draft's KV footprint is O(window) instead of O(context).
- `layers/attention/triton_ops/{extend,decode}_attention.py` — the **num_stages**
  optimization: the Triton draft-attention pipeline depth is read from
  `RK_EXTEND_STAGES` (2) / `RK_DECODE_STAGES` (3), overlapping the windowed KV
  loads with the MMA. Numerics-neutral (AL unchanged); a large part of the per-step
  `t_draft` win. Shipped as drop-in replacements of the stock 26.06 kernels.

Windowed-MTP requires the **triton** draft backend (ring remap + num_stages);
native MTP uses flashinfer. `t_draft` is verified backend-invariant in the paper,
so this backend choice shifts no per-step cost on its own.

## Extending

- **CWE (Qwen-only; `tab:bestdepth`/`tab:hero`):** hard common-word extraction —
  Qwen-35B best-depth **1.53×**, Qwen-122B **1.48×**. Degenerate for Nemotron (its
  short-answer decodes are too brief to compare across arms). The corpus is built by
  step 2, so just:

  ```bash
  INPUTS=cwe bash run/run_headline.sh
  # or a single point:
  run/run_point.sh q35 nextn 1 1040000 4032 1 cwe   # windowed
  run/run_point.sh q35 nextn 1 1040000 0    0 cwe   # native
  ```

- **Other RULER tasks / lengths:** `src/gen_ruler.py` reimplements the RULER task
  families. Build any of `{niah_single,niah_multikey,niah_multivalue,
  niah_multiquery,vt,cwe,fwe}` at any length, then point a run at the jsonl:

  ```bash
  python src/gen_ruler.py --lengths 1150000 --num-samples 8 \
      --tasks niah_multivalue --niah-vals 64 --label enum \
      --out-dir data/ruler_qa_1m
  run/run_point.sh q35 nextn 1 1040000 4032 1 data/ruler_qa_1m/niah_multivalue_enum_1150000.jsonl
  ```

- **Draft-depth / batch sweeps:** set `DRAFT=3|5|7` (num-steps follows as d−1),
  or rerun with `B>1` (+ `MEMFRAC` / `MAXRUN`).

## Citation

If you use this artifact or its results, please cite the paper (GitHub also renders a
"Cite this repository" button from `CITATION.cff`):

```bibtex
@misc{valliappan2026windowedmtp,
  title  = {Windowed-MTP: Removing the Full-Context Draft-KV Tax at Million-Token Context},
  author = {Alagappan Valliappan},
  year   = {2026},
  note   = {NVIDIA. Code: https://github.com/avalliappan-nvidia/windowed-mtp-b200},
  url    = {https://arxiv.org/abs/2607.21535},
  doi    = {10.5281/zenodo.21522901},
}
```

## License and attribution

This package's own code (`src/*.py`, `run/*.sh`, `setup/*.sh`) is released under
the **Apache License 2.0** (see `LICENSE`). Full third-party attributions are in
`NOTICE`; in brief:

- **SGLang** (Apache-2.0) — `patch/windowed_mtp.patch` and the drop-in files in
  `patch/` are **derivative works** of SGLang 26.06; changed regions carry inline
  `[Windowed-MTP]` notices. **FlashInfer** (Apache-2.0) and **Triton** (MIT) are
  used via the container. The `nvcr.io/nvidia/sglang:26.06-py3` image is pulled at
  setup time (not redistributed) under NVIDIA's NGC container terms.
- **Inputs** are synthesized locally by `src/gen_ruler.py`, an independent
  re-implementation of the **RULER** (Hsieh et al., 2024) task templates — no
  RULER data is embedded and no dataset license applies. (**StreamingLLM** and
  **YaRN** are cited techniques, not vendored code.)
- **Models** are downloaded, not redistributed: `RedHatAI/Qwen3.6-35B-A3B-NVFP4`
  and `nvidia/Qwen3.5-122B-A10B-NVFP4` are **Apache-2.0**;
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` is under the **NVIDIA Nemotron
  Open Model License** (commercial use + derivatives permitted; accept the terms
  on its Hugging Face card before download).

All datasets, models, and tools used here are licensed for use and publication.

## Notes

- Qwen checkpoints reach 1M via a **YaRN factor 4**. `setup/02_get_models.sh`
  bakes it into a local model dir `models/<name>-yarn4/` (weights symlinked from
  the HF cache, plus a real `config.json` with `text_config.rope_parameters`
  switched to `{rope_type: yarn, factor: 4.0, original_max_position_embeddings:
  262144}` and `max_position_embeddings=1048576`). We do this on disk rather than
  a runtime override because these are *multimodal* configs: a partial
  `text_config` override drops `num_attention_heads` and crashes engine init.
  Nemotron is native NoPE (no rope edit). `run/run_point.sh` mounts the repo 1:1
  so the model dir + its symlinks resolve identically inside the container.
- **Checkpoints** (full HF repo IDs, set in `config.env`): q35 =
  `RedHatAI/Qwen3.6-35B-A3B-NVFP4`, q122 = `nvidia/Qwen3.5-122B-A10B-NVFP4`, nem =
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`.
- **Nemotron** needs `--mamba-scheduler-strategy no_buffer --disable-radix-cache`
  with `--mamba-full-memory-ratio 0` under speculation (set automatically in
  `run/run_point.sh`); Qwen uses `--mamba-full-memory-ratio 0.1`.
- All KV is standardized to **bf16** (`--kv-cache-dtype bfloat16`); the 122B ships
  FP8 KV scales that are overridden.
