"""Extract SGLang-native steady-state decode TPOT from a bench job log.

The bench harness brackets its timed decode with:
    RK_DEC_BEGIN tag=<tag> B=<B>
    ... (scheduler emits: "Decode batch, #running-req: <n>, accept len: <AL>, ... gen throughput (token/s): <X>")
    RK_DEC_END tag=<tag>

  *** TPOT recovery (corrected 2026-06-30) ***
  The TRUE per-output-token latency a user experiences = total_decode_wall / total_tokens
  = T_iter / mean(AL), where T_iter is the per-iteration wall. The OLD estimator
  TPOT = B/median(throughput) equals T_iter/MEDIAN(AL) — and median(AL) SNAPS to integer
  values (each decode line reports accept len ~3.00/4.00/...), so when mean(AL) sits near a
  half-integer the median jumps and biases TPOT by 15-40% (stably, both directions). This
  systematically inflated windowed win/native (windowed AL is skewed toward the ceiling).
  Fix: per-line T_iter = accept_len_line / throughput_line (AL cancels exactly per interval),
  median over steady lines, then TPOT = T_iter / mean(AL). We emit the corrected `tpot_gt_ms`
  and keep the old one as `tpot_med_ms` for reference.

Usage: python parse_gt.py <logfile> [--drop 2]
"""
import argparse
import json
import re
import statistics
import sys

BEGIN = re.compile(r"RK_DEC_BEGIN tag=(\S+) B=(\d+)")
END = re.compile(r"RK_DEC_END tag=(\S+)")
# NOTE: dense (non-spec) decode lines have NO "accept len:" field (that field only
# appears under speculative decoding). So we match running-req + throughput here and
# capture accept-len SEPARATELY, defaulting to 1.0 when absent (dense => 1 tok/step).
DEC = re.compile(r"Decode batch.*?#running-req:\s*(\d+).*?gen throughput \(token/s\):\s*([\d.]+)")
AL_RE = re.compile(r"accept len:\s*([\d.]+)")
# "#full token: 0" lines are prefill-flush / empty-batch artifacts: no KV tokens are
# actually resident, so the reported gen throughput spikes to nonsense (e.g. 46431 tok/s)
# and would bias T_iter/TPOT. Drop them at parse time (only when the field is present).
FULLTOK = re.compile(r"#full token:\s*(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--drop", type=int, default=2, help="ramp steps to drop at segment start")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    lines = open(args.log, errors="ignore").read().splitlines()
    cur = None  # (tag, B, [throughputs])
    out = []
    for ln in lines:
        m = BEGIN.search(ln)
        if m:
            cur = [m.group(1), int(m.group(2)), []]
            continue
        if cur is not None:
            me = END.search(ln)
            if me and me.group(1) == cur[0]:
                tag, B, xs = cur
                # Pick the steady-state batch = the LARGEST concurrency actually
                # sustained (>=5 steps). Prefer the requested B; else the highest
                # running-req held for >=5 steps. NOT the global mode: with spec
                # decode, per-seq AL variance staggers completion, so a low-AL seq
                # can decode ALONE for a long ramp-down tail (#running-req=1) that
                # out-counts the concurrent phase and would wrongly win the mode
                # (e.g. B8/255k windowed: 127 tail steps @1 vs 81 @7). The real
                # steady-state TPOT is the full-batch phase; the tail is ramp-down.
                from collections import Counter
                counts = Counter(n for (n, _, _) in xs if n > 0)
                b_eff = B
                if counts.get(B, 0) < 5 and counts:
                    sustained = [n for n, c in counts.items() if c >= 5]
                    b_eff = max(sustained) if sustained else counts.most_common(1)[0][0]
                steady = [(al, x) for (n, al, x) in xs if n == b_eff and x > 0]
                steady = steady[args.drop:] if len(steady) > args.drop + 1 else steady
                tputs = [x for (_, x) in steady]
                als = [al for (al, _) in steady]
                med = statistics.median(tputs) if tputs else float("nan")
                # OLD (biased) estimator, kept for reference: B / median(throughput)
                tpot_med = (b_eff / med * 1000.0) if med and med == med and med > 0 else float("nan")
                # CORRECTED: per-line T_iter = accept_len/throughput (AL cancels per interval),
                # robust-median, then TPOT = T_iter / mean(AL). For B>1, T_iter is per-batch-step
                # so divide by b_eff to get per-seq per-token.
                tpot = float("nan")
                mean_al = float("nan")
                if steady:
                    # accept_len/throughput = T_iter/b_eff (throughput is batch-total tok/s);
                    # so per-seq per-token TPOT = (T_iter)/mean_al = (titer*b_eff)/mean_al.
                    titer = [al / x * 1000.0 for (al, x) in steady]
                    tmd = statistics.median(titer)
                    # keep the SAME band-clipped lines for BOTH T_iter and mean(AL) so
                    # prefill-flush spikes (full token:0 -> huge tok/s, low AL) can't leak
                    # into either; <=2% effect but keeps the two estimators consistent.
                    kept = [(al, t) for (al, _), t in zip(steady, titer) if 0.4 * tmd <= t <= 2.5 * tmd] or list(zip(als, titer))
                    t_iter = statistics.median([t for _, t in kept])
                    mean_al = statistics.mean([al for al, _ in kept])
                    if mean_al > 0:
                        tpot = t_iter * b_eff / mean_al
                # derived throughput (from the corrected per-user TPOT):
                #   tok_s_user = per-user decode rate; tok_s_gpu = AGGREGATE system
                #   throughput over the B_eff resident requests (per TP group, NOT
                #   divided by TP). This is TP-agnostic on purpose: the consumer
                #   (collect / plot) divides by the TP size for per-physical-GPU.
                ok = tpot == tpot and tpot > 0
                tok_s_user = (1000.0 / tpot) if ok else float("nan")
                tok_s_gpu = (b_eff * 1000.0 / tpot) if ok else float("nan")
                out.append({
                    "tag": tag, "B": B, "B_eff": b_eff, "n_steady": len(steady),
                    "gen_tps_median": med, "tpot_gt_ms": tpot, "tpot_med_ms": tpot_med,
                    "mean_al": mean_al,
                    "tok_s_user": tok_s_user, "tok_s_gpu": tok_s_gpu,
                })
                cur = None
                continue
            md = DEC.search(ln)
            if md:
                mft = FULLTOK.search(ln)
                if mft and int(mft.group(1)) == 0:
                    continue  # prefill-flush / empty-batch artifact: spurious huge tok/s
                mal = AL_RE.search(ln)
                al = float(mal.group(1)) if mal else 1.0
                cur[2].append((int(md.group(1)), al, float(md.group(2))))

    for r in out:
        if args.json:
            print(json.dumps(r))
        else:
            beff = f"(B_eff={r['B_eff']})" if r.get("B_eff") != r["B"] else ""
            print(f"  {r['tag']:30s} TPOT_gt={r['tpot_gt_ms']:7.2f}ms "
                  f"(old_med={r.get('tpot_med_ms', float('nan')):7.2f}) meanAL={r.get('mean_al', float('nan')):.3f} "
                  f"tok/s/user={r.get('tok_s_user', float('nan')):6.1f} tok/s={r.get('tok_s_gpu', float('nan')):7.1f} "
                  f"steps={r['n_steady']} {beff}")
    if not out:
        print("(no RK_DEC segments found)", file=sys.stderr)


if __name__ == "__main__":
    main()
