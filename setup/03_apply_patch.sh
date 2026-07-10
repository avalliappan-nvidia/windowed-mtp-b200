#!/usr/bin/env bash
# Produce a patched SGLang source tree ("$SGLANG_SRC") that we bind-mount over
# the container's site-packages at run time. Steps:
#   1. extract the stock sglang package out of the container image,
#   2. apply patch/windowed_mtp.patch (windowing + O(S) extend-pass edits),
#   3. drop in the new files: ring_draft.py, attn_mass.py, and the two
#      num_stages-tunable triton attention kernels (extend/decode_attention.py).
set -euo pipefail
cd "$(dirname "$0")/.."
source ./config.env

if [ -d "$SGLANG_SRC" ]; then
  echo "[03] $SGLANG_SRC already exists; remove it to re-extract. Skipping extraction."
else
  echo "[03] extracting stock sglang from container ($CONTAINER_ENGINE) ..."
  case "$CONTAINER_ENGINE" in
    docker)
      cid=$(docker create "$IMAGE_TAG")
      docker cp "$cid:$SGLANG_IN_CONTAINER" "$SGLANG_SRC"
      docker rm "$cid" >/dev/null
      ;;
    enroot)
      name="wmtp_extract_$$"
      enroot create -n "$name" "$IMAGE_SQSH"
      enroot start --rw --mount "$REPRO_ROOT:/out" "$name" \
        bash -c "cp -a $SGLANG_IN_CONTAINER /out/$(basename "$SGLANG_SRC")"
      enroot remove -f "$name"
      ;;
    *) echo "unknown CONTAINER_ENGINE=$CONTAINER_ENGINE" >&2; exit 2 ;;
  esac
fi

echo "[03] applying windowed_mtp.patch ..."
patch -p1 -d "$SGLANG_SRC" < patch/windowed_mtp.patch

echo "[03] installing new files (ring_draft.py, attn_mass.py) ..."
cp patch/ring_draft.py "$SGLANG_SRC/srt/speculative/ring_draft.py"
cp patch/attn_mass.py  "$SGLANG_SRC/srt/speculative/attn_mass.py"

# num_stages draft-attn optimization (the paper's per-step cost win). These two
# files are stock sglang 26.06 triton kernels with ONE change: the Triton
# pipeline depth (num_stages) is read from RK_EXTEND_STAGES / RK_DECODE_STAGES
# (see run/run_point.sh, which exports 2 / 3 for every windowed run). It is a
# numerics-neutral scheduling tweak -- AL is unchanged; only t_draft drops.
# Shipped as drop-in replacements (not a diff) so they stay robust to whitespace.
echo "[03] installing num_stages triton kernels (extend/decode_attention.py) ..."
cp patch/extend_attention.py "$SGLANG_SRC/srt/layers/attention/triton_ops/extend_attention.py"
cp patch/decode_attention.py "$SGLANG_SRC/srt/layers/attention/triton_ops/decode_attention.py"

echo "[03] verifying markers present ..."
grep -q "RK_MTP_WINDOW" "$SGLANG_SRC/srt/models/qwen3_5.py" && echo "  qwen3_5.py OK"
grep -q "RK_DRAFT_RING" "$SGLANG_SRC/srt/speculative/eagle_info_v2.py" && echo "  eagle_info_v2.py OK"
test -f "$SGLANG_SRC/srt/speculative/ring_draft.py" && echo "  ring_draft.py OK"
grep -q "RK_EXTEND_STAGES" "$SGLANG_SRC/srt/layers/attention/triton_ops/extend_attention.py" && echo "  extend_attention.py (num_stages) OK"
grep -q "RK_DECODE_STAGES" "$SGLANG_SRC/srt/layers/attention/triton_ops/decode_attention.py" && echo "  decode_attention.py (num_stages) OK"
echo "[03] done. Patched tree at: $SGLANG_SRC"
