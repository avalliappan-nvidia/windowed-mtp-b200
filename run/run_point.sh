#!/usr/bin/env bash
# Run ONE benchmark point (one model, one config) on a single B200 and print
# TPOT / acceptance-length. Thin, self-describing wrapper around the container.
# Flags mirror the paper's Reproducibility appendix exactly.
#
# Usage:
#   run/run_point.sh MODEL ALGO B ISL W RING [INPUT]
#     MODEL  full HF id (or a key: q35|q122|nemo)
#     ALGO   nextn (self-MTP speculative) | none (dense baseline)
#     B      batch size (concurrent requests)
#     ISL    input (prompt) length in tokens (headline: 1040000)
#     W      draft attention window (0 = native full-KV MTP; >0 = Windowed-MTP; headline 4032)
#     RING   1 = compact draft-KV ring pool (needs W>0); 0 = compute-only window
#     INPUT  one of:
#              random  engine's built-in random tokens (no real corpus)
#              mqe     niah_multiquery_enum  (RULER, 48 keys -> long list; HERO input, QA mode)
#              fwe     fwe_enum              (RULER, top-30 frequent words; HERO input, QA mode)
#              cwe     common-word extraction (RULER; the end-to-end marquee task, QA mode)
#              <path>  an explicit *.jsonl (passed straight to --real-input)
#            mqe/fwe/cwe are seeded synthetic RULER corpora built by setup/02_get_models.sh
#            (byte-reproducible, no license required).
#
# Env overrides: MEMFRAC(0.85) SINK(64) STEPS(=d-1) TOPK(1) DRAFT/d(7) NDEC(512)
#   KVDTYPE(bfloat16) PAGESIZE(1) MAXRUN(0=auto) QAMAXNEW(512) REPPEN(1.15)
#   RK_EXTEND_STAGES(2) RK_DECODE_STAGES(3)  # num_stages draft-attn opt (windowed runs).
set -euo pipefail
cd "$(dirname "$0")/.."
source ./config.env

echo "[run_point] host=$(hostname) engine=$CONTAINER_ENGINE slurm_job=${SLURM_JOB_ID:-<none>} $(date '+%F %H:%M:%S')"

MODEL_ARG=${1:?MODEL}; ALGO=${2:?ALGO}; B=${3:?B}; ISL=${4:?ISL}; W=${5:?W}; RING=${6:?RING}
INPUT=${7:-random}

# Resolve INPUT -> (real-input path, QA flags, short filename label). Known keys
# map to the seeded corpora built by setup/02_get_models.sh; an explicit *.jsonl
# path is passed straight through (fixed-decode unless QAMODE=1 is set).
# The RULER enum corpora carry the canonical 1150000-target-token suffix; the
# harness re-tokenizes and slices to the exact ISL, so the on-disk length is loose.
RULER_DIR="$DATA_DIR/ruler_qa_1m"
REALIN=""; QA_FLAGS=""
case "$INPUT" in
  random)  INPUT_LABEL=random ;;
  mqe|niah_multiquery_enum)
           INPUT_LABEL=niah_multiquery_enum; REALIN="$RULER_DIR/niah_multiquery_enum_1150000.jsonl"
           QA_FLAGS="--qa-mode --qa-max-new ${QAMAXNEW:-512}" ;;
  fwe|fwe_enum)
           INPUT_LABEL=fwe_enum; REALIN="$RULER_DIR/fwe_enum_1150000.jsonl"
           QA_FLAGS="--qa-mode --qa-max-new ${QAMAXNEW:-512}" ;;
  cwe)     INPUT_LABEL=cwe; REALIN="$RULER_DIR/cwe_1150000.jsonl"
           QA_FLAGS="--qa-mode --qa-max-new ${QAMAXNEW:-512}" ;;
  *.jsonl) INPUT_LABEL="$(basename "$INPUT" .jsonl)"; REALIN="$INPUT"
           [ "${QAMODE:-0}" != 0 ] && QA_FLAGS="--qa-mode --qa-max-new ${QAMAXNEW:-512}" ;;
  *) echo "[ERR] unknown INPUT '$INPUT' (use: random | mqe | fwe | cwe | <path-to.jsonl>)" >&2; exit 6 ;;
esac
if [ -n "$REALIN" ] && [ ! -f "$REALIN" ]; then
  echo "[ERR] input file missing: $REALIN" >&2
  echo "      Run setup/02_get_models.sh first (it builds the RULER enum corpora)." >&2
  exit 6
fi

# q35/q122 use the local YaRN-prepared dirs (see setup/02_get_models.sh); nemo is
# native NoPE straight from the HF cache; anything else is passed through as-is.
case "$MODEL_ARG" in
  q35)  MODEL="$Q35_LOCAL" ;;
  q122) MODEL="$Q122_LOCAL" ;;
  nemo) MODEL="$MODEL_NEMO" ;;
  *)    MODEL="$MODEL_ARG" ;;
esac

# For the Qwen keys the prepared dir must exist (built by setup/02_get_models.sh).
case "$MODEL_ARG" in
  q35|q122)
    if [ ! -f "$MODEL/config.json" ]; then
      echo "[ERR] YaRN-prepared model dir missing: $MODEL" >&2
      echo "      Run setup/02_get_models.sh first (it downloads + prepares it)." >&2
      exit 5
    fi ;;
esac

# Headline speculation geometry: d = num-draft-tokens, num-steps = d-1, topk 1.
DRAFT=${DRAFT:-7}; STEPS=${STEPS:-$((DRAFT-1))}; TOPK=${TOPK:-1}
NDEC=${NDEC:-512}; MEMFRAC=${MEMFRAC:-0.85}; SINK=${SINK:-64}
PAGESIZE=${PAGESIZE:-1}; MAXRUN=${MAXRUN:-0}
KVDTYPE=${KVDTYPE:-bfloat16}
# Sampling: a repetition penalty (default 1.15) is applied IDENTICALLY to the
# draft and target logits, so speculation stays lossless w.r.t. the penalized
# distribution while breaking the greedy repetition loops that otherwise inflate
# AL on long decodes. Set REPPEN=1.0 to disable. QAMAXNEW caps decoded answer len.
REPPEN=${REPPEN:-1.15}; QAMAXNEW=${QAMAXNEW:-512}

# Draft attention backend: Windowed-MTP requires triton (ring index remap);
# native MTP uses flashinfer. Verify/target backend is always flashinfer.
if [ "$W" != 0 ]; then DBK=triton; else DBK=flashinfer; fi

mkdir -p "$RESULTS_DIR" "$LOGS_DIR" "$CACHE_DIR"
# NOTE: keep this set -e safe. A `TAG="...$([ cond ] && echo x)..."` form makes
# the whole assignment inherit the command substitution's exit status, so when
# the test is false (e.g. RING=0) the assignment "fails" and set -e silently
# kills the script before it prints anything useful. Compute the suffix first.
RING_SFX=""
if [ "$RING" != 0 ]; then RING_SFX="_ring"; fi
TAG="$(basename "$MODEL")_${ALGO}_B${B}_isl${ISL}_w${W}${RING_SFX}_${INPUT_LABEL}"
OUT="$RESULTS_DIR/${TAG}.jsonl"; rm -f "$OUT"
LOG="$LOGS_DIR/${TAG}.log"

# --- per-model + windowing flags -------------------------------------------
EXTRA="--kv-cache-dtype $KVDTYPE --page-size $PAGESIZE"
case "$MODEL" in
  *Qwen3*)   EXTRA="$EXTRA --mamba-full-memory-ratio 0.1" ;;   # YaRN is baked into the local config.json
  *Nemotron*) EXTRA="$EXTRA --mamba-full-memory-ratio 0 --mamba-scheduler-strategy no_buffer --disable-radix-cache" ;;
esac
[ "$MAXRUN" != 0 ] && EXTRA="$EXTRA --max-running-requests $MAXRUN"
[ -n "$REALIN" ] && EXTRA="$EXTRA --real-input $REALIN"
[ -n "$QA_FLAGS" ] && EXTRA="$EXTRA $QA_FLAGS"
# Repetition penalty (identical draft+target -> lossless). Only emit when != 1.0.
[ "$REPPEN" != "1.0" ] && EXTRA="$EXTRA --repetition-penalty $REPPEN"

# --- container env ----------------------------------------------------------
# Caches (flashinfer/triton/inductor) default to $HOME/.cache = /root/.cache,
# which we bind-mount to $CACHE_DIR below (flashinfer ignores XDG_CACHE_HOME).
# HF_HOME points at the host cache path (mounted 1:1, see MOUNTS) so the model
# dirs' symlinks into the snapshot resolve identically inside the container.
CENV="export HF_HOME=$HF_CACHE SGLANG_ENABLE_SPEC_V2=1 \
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1;"
if [ "$W" != 0 ]; then
  CENV="$CENV export RK_MTP_WINDOW=$W RK_MTP_SINK=$SINK;"
  [ "$RING" != 0 ] && CENV="$CENV export RK_DRAFT_RING=1;"
  # num_stages draft-attn optimization: deeper Triton pipeline overlaps the
  # windowed KV loads with the MMA. ON for every windowed run (W>0 => triton
  # draft). Numerics-neutral (AL unchanged); it is a large part of the per-step
  # t_draft win. Set RK_EXTEND_STAGES=1 RK_DECODE_STAGES=2 (stock depths) to disable.
  CENV="$CENV export RK_EXTEND_STAGES=${RK_EXTEND_STAGES:-2} RK_DECODE_STAGES=${RK_DECODE_STAGES:-3};"
fi

# NOTE: this command runs INSIDE the container, so it uses the *container's*
# python3 (SGLang's interpreter) — NOT the host venv. Host-side scripts below
# use the host python3 (the venv activated by config.env). They are deliberately
# separate interpreters; do not "unify" them. Paths are host-absolute and the
# repo is mounted 1:1 (see MOUNTS), so $MODEL / --real-input / --out resolve to
# the same path inside the container (this is also what lets the prepared model
# dir's symlinks into the HF snapshot resolve).
CMD="python3 $REPRO_ROOT/src/bench_sglang_mtp.py --model $MODEL --algo $ALGO \
  --batch $B --isl $ISL --n-decode $NDEC --attention-backend flashinfer \
  --draft-attention-backend $DBK --mem-frac $MEMFRAC --tp 1 \
  --steps $STEPS --topk $TOPK --draft-tokens $DRAFT $EXTRA \
  --out $OUT --tag $TAG"

echo "########## POINT $TAG ##########"
echo "[model] $MODEL  [algo] $ALGO  B=$B ISL=$ISL W=$W sink=$SINK ring=$RING draft_bk=$DBK d=$DRAFT input=$INPUT"
echo "[cmd] $CMD"

# --- preflight: make sure the chosen container engine is actually usable ------
# (A misconfigured engine is the #1 cause of a "nothing happens / stuck" run.)
case "$CONTAINER_ENGINE" in
  docker)
    command -v docker >/dev/null 2>&1 || {
      echo "[ERR] CONTAINER_ENGINE=docker but 'docker' is not in PATH." >&2
      echo "      On a SLURM + pyxis/enroot cluster set CONTAINER_ENGINE=enroot" >&2
      echo "      (export it or edit config.env). See README 'Where to run'." >&2
      exit 3; }
    ;;
  enroot)
    command -v srun >/dev/null 2>&1 || {
      echo "[ERR] CONTAINER_ENGINE=enroot but 'srun' is not in PATH." >&2; exit 3; }
    # Require an existing allocation. Without one, `srun` would try to CREATE a
    # job and fail with cluster-specific errors ("You must specify an account")
    # since we cannot guess your account/partition/GRES. Inside an allocation,
    # the srun step below inherits all of those automatically.
    if [ -z "${SLURM_JOB_ID:-}" ]; then
      echo "[ERR] No active SLURM allocation (SLURM_JOB_ID is unset)." >&2
      echo "      This package can't guess your cluster's account / partition / GRES," >&2
      echo "      so grab an allocation yourself, then re-run the script inside it:" >&2
      echo "" >&2
      echo "        salloc -A <account> -p <b200-partition> --gres=gpu:1 -N1 -t 04:00:00" >&2
      echo "        # then, in the shell salloc gives you:" >&2
      echo "        bash run/run_headline.sh          # or: bash run/run_point.sh ..." >&2
      echo "" >&2
      echo "      (srun launched inside the allocation inherits account/partition/GRES.)" >&2
      exit 4
    fi
    ;;
  *) echo "[ERR] unknown CONTAINER_ENGINE=$CONTAINER_ENGINE (use docker or enroot)" >&2; exit 2 ;;
esac

# --- launch -----------------------------------------------------------------
# Bind-mounts. We mount the repo (and, if separate, the HF cache) 1:1 at their
# host paths so every absolute path we pass — the model dir, its symlinks into
# the HF snapshot, --real-input, --out — resolves identically inside the
# container. Also: patched sglang over site-packages, and a writable JIT cache
# at /root/.cache (flashinfer hard-codes $HOME/.cache).
MOUNTS="$SGLANG_SRC:$SGLANG_IN_CONTAINER,$REPRO_ROOT:$REPRO_ROOT,$CACHE_DIR:/root/.cache"
case "$HF_CACHE" in
  "$REPRO_ROOT"/*) : ;;                          # nested in the repo, already mounted
  *) MOUNTS="$MOUNTS,$HF_CACHE:$HF_CACHE" ;;      # external cache -> mount 1:1 too
esac
echo "[launch] starting container via '$CONTAINER_ENGINE' at $(date '+%H:%M:%S') ..."
echo "         first run downloads the image (~10s of GB) and JIT-compiles kernels;"
echo "         this can take many minutes with NO output. Live progress -> $LOG"
echo "[mounts] $MOUNTS"
case "$CONTAINER_ENGINE" in
  docker)
    # translate the comma-mount list into -v flags
    DV=""; IFS=',' read -ra _M <<< "$MOUNTS"; for m in "${_M[@]}"; do DV="$DV -v $m"; done
    echo "[docker] docker run --rm --gpus all --ipc=host$DV $IMAGE_TAG bash -lc \"<CENV> <CMD>\""
    docker run --rm --gpus all --ipc=host $DV \
      "$IMAGE_TAG" bash -lc "$CENV $CMD" 2>&1 | tee "$LOG"
    ;;
  enroot)
    IMG="$IMAGE_TAG"; [ -f "$IMAGE_SQSH" ] && IMG="$IMAGE_SQSH"
    echo "[srun] srun --container-image $IMG \\"
    echo "         --container-mounts $MOUNTS \\"
    echo "         bash -lc \"$CENV $CMD\""
    srun --container-image "$IMG" \
      --container-mounts "$MOUNTS" \
      bash -lc "$CENV $CMD" 2>&1 | tee "$LOG"
    ;;
esac
echo "[launch] container exited at $(date '+%H:%M:%S') (rc from pipe: ${PIPESTATUS[0]:-?})"

echo "########## TPOT / AL (steady-state, from scheduler log) ##########"
# Host-side parse (host python3 / the venv activated by config.env).
python3 src/parse_gt.py "$LOG" || true
echo "result -> $OUT"
