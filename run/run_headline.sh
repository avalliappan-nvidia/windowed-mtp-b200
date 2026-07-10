#!/usr/bin/env bash
# Headline reproduction: native MTP vs Windowed-MTP at 1M context, B=1, d=7, for
# all three checkpoints, on each workload in $INPUTS. Prints a per-input summary
# of TPOT, acceptance length (AL), and the per-decode-step cost reduction.
#
#   per-decode-step cost  step = TPOT_gt * mean_AL   (one speculative iteration)
#   win%                  = step_native / step_win - 1   (the paper's convention:
#                           how much faster ONE decode step is under windowing)
#
# This is the artifact's core, self-contained claim: windowing cuts the
# per-decode-step cost by +28% to +44% while preserving acceptance length, across
# models and inputs (Table 1 of the paper). The two headline inputs are seeded,
# byte-reproducible synthetic RULER tasks built by setup/02_get_models.sh (no HF
# license): niah_multiquery_enum (48-key retrieval) and fwe_enum (top-30
# aggregation). (The paper also reports end-to-end tok/s on RULER CWE; run that
# with INPUTS=cwe or via run_point.sh -- see README "Extending".)
set -euo pipefail
cd "$(dirname "$0")/.."
source ./config.env

ISL=${ISL:-1040000}
MODELS=${MODELS:-"q35 q122 nemo"}
INPUTS=${INPUTS:-"niah_multiquery_enum fwe_enum"}
WIN=${WIN:-4032}

NMODELS=$(echo $MODELS | wc -w); NINPUTS=$(echo $INPUTS | wc -w)
NRUN=$((NMODELS*NINPUTS*2))
echo "############################ HEADLINE START ############################"
echo "[headline] host=$(hostname) engine=$CONTAINER_ENGINE $(date '+%F %H:%M:%S')"
echo "[headline] models='$MODELS'  inputs='$INPUTS'  ISL=$ISL  WIN=$WIN"
echo "[headline] $NMODELS models x $NINPUTS inputs x 2 points = $NRUN container runs; budget ~$((NRUN/6+1))-$((NRUN/3+1)) h."
echo "[headline] per-point logs -> $LOGS_DIR   (tail them if a point looks stuck)"
echo "[headline] NOTE: with CONTAINER_ENGINE=docker run this ON the GPU node;"
echo "           with CONTAINER_ENGINE=enroot run it from a SLURM allocation."
echo "           The first container run pulls a large image and may sit silent for minutes."

for INPUT in $INPUTS; do
  i=0
  for M in $MODELS; do
    i=$((i+1))
    echo "============ [$INPUT $i/$NMODELS] $M : native MTP (w=0) [$(date '+%H:%M:%S')] ============"
    run/run_point.sh "$M" nextn 1 "$ISL" 0    0 "$INPUT"
    echo "============ [$INPUT $i/$NMODELS] $M : Windowed-MTP (w=$WIN, ring) [$(date '+%H:%M:%S')] ============"
    run/run_point.sh "$M" nextn 1 "$ISL" "$WIN" 1 "$INPUT"
  done
done
echo "[headline] all points finished at $(date '+%F %H:%M:%S')"

echo
echo "############################ HEADLINE SUMMARY ############################"
python3 - "$LOGS_DIR" "$ISL" "$WIN" "$INPUTS" <<'PY'
import sys, glob, subprocess, json, os
logs_dir, isl, win, inputs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()
def gt(logpath):
    if not logpath:
        return None
    try:
        out = subprocess.check_output([sys.executable, "src/parse_gt.py", logpath, "--json"], text=True)
    except Exception:
        return None
    rows = [json.loads(l) for l in out.splitlines() if l.strip()]
    return rows[-1] if rows else None
def base(p):
    return os.path.basename(p).split("_nextn_")[0]
for inp in inputs:
    print(f"\n=== input={inp} (d=7, gamma=6, 1M, B=1) ===")
    print(f"{'model':22s} {'config':8s} {'TPOT_ms':>9s} {'AL':>6s} {'step_ms':>9s} {'win%':>6s}")
    # Pair every native log with its windowed sibling by model basename.
    nats = {base(p): p for p in glob.glob(f"{logs_dir}/*_nextn_B1_isl{isl}_w0_{inp}.log")}
    wins = {base(p): p for p in glob.glob(f"{logs_dir}/*_nextn_B1_isl{isl}_w{win}_ring_{inp}.log")}
    for name in sorted(nats):
        rn, rw = gt(nats[name]), gt(wins.get(name, ""))
        if not rn or not rw:
            print(f"{name:20s} (missing native or windowed log)"); continue
        tn = rn["tpot_gt_ms"] * rn["mean_al"]; tw = rw["tpot_gt_ms"] * rw["mean_al"]
        # paper convention: win% = step_native/step_win - 1 (per-step speedup).
        win_pct = (tn / tw - 1) * 100 if tw else float("nan")
        print(f"{name:20s} native   {rn['tpot_gt_ms']:9.2f} {rn['mean_al']:6.2f} {tn:9.2f}")
        print(f"{name:20s} windowed {rw['tpot_gt_ms']:9.2f} {rw['mean_al']:6.2f} {tw:9.2f}   {win_pct:6.1f}")
print("\n#########################################################################")
print("Expected (d=7): acceptance length preserved (within ~+-10%) native->win;")
print("per-decode-step cost win% = step_nat/step_win-1 in the +28% to +44% band")
print("(largest for the MoE-FFN Qwen-35B draft, smallest for Nemotron's Mamba2).")
PY
