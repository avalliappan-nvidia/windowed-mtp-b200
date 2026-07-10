"""[Windowed-MTP] DRAFT attention-MASS probe  (enable with RK_ATTN_MASS=1).

WHAT / WHY
----------
This is the DIRECT evidence for the windowing mechanism claimed in the paper.
The paper's headline result is that truncating the *draft* model's KV to a
trailing window of W (+ S sink) tokens barely hurts acceptance length (AL) at
1M context. That is currently argued *indirectly* (window the draft -> AL stays
high). This probe measures the *cause*: on a NATIVE (full-context) draft, what
fraction of the draft's softmax attention mass already lands inside the last W
tokens (plus the first S sink tokens)?

    if in-window mass ~= 1.0  ->  removing the far KV cannot change the draft's
                                  output  ->  windowing is (nearly) lossless.
    if in-window mass << 1.0  ->  the draft genuinely needs far context  ->
                                  windowing will cost AL (the failure cases:
                                  code / retrieval).

So a single number -- in-window attention mass vs W, per model, per task --
turns "windowing happens to work" into "windowing works BECAUSE draft attention
is local", and simultaneously explains the code/retrieval regressions.

HOW
---
We recompute the attention distribution in fp32 from the *already-cached* draft
K, so the measurement is independent of the FlashInfer/Triton kernel (which
never exposes per-position scores). For a decode query q_h (head h) over the
request's chronological key positions [0, seq):

    scores_h[j] = scaling * <q_h, K_{kvhead(h), j}>        (GQA: kvhead=h//group)
    probs_h     = softmax(scores_h)
    mass(W,S)   = sum_{j in [0,S)} probs_h[j] + sum_{j in [seq-W, seq)} probs_h[j]

averaged over heads, requests, layers and the first RK_ATTN_MASS_PASSES decode
passes. Results are dumped to JSON at process exit.

COST / SAFETY
-------------
- EAGER ONLY. The hook is skipped while a CUDA graph is capturing, and under a
  captured graph forward_decode is *replayed* (Python never runs) -> it would
  never fire. => the measurement run MUST pass --disable-cuda-graph.
- Bounded: only the first RK_ATTN_MASS_PASSES fired passes and at most
  RK_ATTN_MASS_MAXREQ requests per pass are processed.
- Must be driven with the TRITON draft backend (the hook lives in TritonAttnBackend)
  and window OFF (RK_MTP_WINDOW unset / 0) so kv_indices is the FULL context.

All inert unless RK_ATTN_MASS=1.
"""

from __future__ import annotations

import atexit
import json
import math
import os
from typing import Dict, List, Tuple

import torch

# ------------------------------------------------------------------ config (env)
ENABLED = os.environ.get("RK_ATTN_MASS", "0") not in ("", "0")
_MAX_PASSES = int(os.environ.get("RK_ATTN_MASS_PASSES", "128"))   # total fired decode passes
_MAX_REQ = int(os.environ.get("RK_ATTN_MASS_MAXREQ", "4"))        # reqs processed per pass
_MIN_SEQ = int(os.environ.get("RK_ATTN_MASS_MINSEQ", "4096"))     # skip short-ctx passes
_OUT = os.environ.get(
    "RK_ATTN_MASS_OUT",
    "attn_mass_%s.json" % os.environ.get("RK_ATTN_MASS_TAG", "run"),
)
# TP>1: each rank has a disjoint head shard and would otherwise clobber the same
# file -> give every rank its own JSON (the aggregator globs + merges them).
_RANK = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or os.environ.get(
    "SGLANG_TP_RANK") or ""
if _RANK not in ("", "0"):
    root, ext = os.path.splitext(_OUT)
    _OUT = f"{root}_rank{_RANK}{ext}"


def _int_list(env: str, default: str) -> List[int]:
    return [int(x) for x in os.environ.get(env, default).split(",") if x.strip()]


# windows / sinks to evaluate the SAME captured distribution against (free -- one
# softmax, many band-sums). Defaults span the paper's operating points.
_W_LIST = _int_list("RK_ATTN_MASS_WLIST", "256,512,1024,2048,4032,8192,16384,32768")
_S_LIST = _int_list("RK_ATTN_MASS_SLIST", "0,64")

# Reference operating window/sink for the TAIL DIFFUSENESS metrics. These answer
# the real mechanism question: raw in-window mass can be < 1 yet windowing stays
# lossless IF the dropped (out-of-window) tail is DIFFUSE (near-uniform ~= average
# V ~= no directional signal). We quantify the tail with:
#   tail_mass            : 1 - in_window_mass at (Wref,Sref)
#   tail_norm_entropy    : H(tail)/log(len) in [0,1]; ->1 = uniform/diffuse (safe)
#   tail_participation   : effective #keys / len (PR/len); ->1 diffuse, ->0 peaked
# High entropy + high participation => the far context is uninformative noise, so
# windowing preserves the argmax (the losslessness mechanism). Low values => the
# draft genuinely retrieves specific far tokens (code/retrieval failure regime).
_WREF = int(os.environ.get("RK_ATTN_MASS_WREF", "4032"))
_SREF = int(os.environ.get("RK_ATTN_MASS_SREF", "64"))

# ---- V-aware OUTPUT contribution (RK_ATTN_MASS_VOUT=1, default on) ------------
# Attention MASS can overstate a key's importance: an attention-sink key gets
# large softmax prob but a low-norm / cancelling value vector, so it barely moves
# the layer output o = sum_j p_j v_j. The decision-relevant quantity is how much
# the OUTPUT changes when we window. At the reference (Wref,Sref) we measure:
#   win_perturb   = ||o_full - o_win|| / ||o_full||   (o_win renormalized over
#                   window+sink) -> the actual error windowing injects at this
#                   draft-attn layer. Small even at high tail_mass => sink-like
#                   low-V far keys => windowing safe.
#   tail_out_frac = ||sum_{j in tail} p_j v_j|| / ||o_full||  -> share of the
#                   output magnitude the dropped tail actually contributes.
#   vnorm_{sink,win,tail} = mean ||v_j|| per bucket (tests the sink=low-V story).
_VOUT = os.environ.get("RK_ATTN_MASS_VOUT", "1") not in ("", "0")

# ------------------------------------------------------------------ accumulators
# key = (layer_id, W, S) -> [sum_mass, n_head_samples]
_ACC: Dict[Tuple[int, int, int], List[float]] = {}
_SEQ: Dict[int, List[float]] = {}          # layer_id -> [sum_seq, n]
# layer_id -> [sum_tail_mass, sum_norm_entropy, sum_participation_frac, n]
_TAIL: Dict[int, List[float]] = {}
# layer_id -> [sum_win_perturb, sum_tail_out_frac, sum_vn_sink, sum_vn_win,
#              sum_vn_tail, n, n_sink]  (V-aware output metrics; weighted by group)
_VOUTACC: Dict[int, List[float]] = {}
_STATE = {"passes": 0, "fired_log": 0, "dumped": False}


def _draft_layer_ok(layer) -> bool:
    """In a NEXTN run the only forward_decode caller is the MTP draft, so by
    default we accept every decode pass. RK_ATTN_MASS_LN (comma list) can pin it
    to specific layer_name substrings if a build also decodes the target."""
    want = [s for s in os.environ.get("RK_ATTN_MASS_LN", "").split(",") if s.strip()]
    if not want:
        return True
    ln = getattr(layer, "layer_name", "") or ""
    return any(s in ln for s in want)


@torch.no_grad()
def maybe_record_decode(q, layer, forward_batch, kv_indptr, kv_indices) -> None:
    """Called from TritonAttnBackend.forward_decode BEFORE the kernel. No-op
    unless RK_ATTN_MASS=1. Recomputes and accumulates in-window attention mass."""
    if not ENABLED or _STATE["passes"] >= _MAX_PASSES:
        return
    # eager-only: never record inside a capturing graph (shapes are data-dependent).
    try:
        if torch.cuda.is_current_stream_capturing():
            return
    except Exception:
        pass
    if not _draft_layer_ok(layer):
        return
    try:
        pool = forward_batch.token_to_kv_pool
        key_buffer = pool.get_key_buffer(layer.layer_id)   # [Nslots, Hkv, Dk]
    except Exception:
        return
    if key_buffer is None or key_buffer.dim() != 3:
        return
    value_buffer = None
    if _VOUT:
        try:
            vb = pool.get_value_buffer(layer.layer_id)     # [Nslots, Hkv, Dv]
            if vb is not None and vb.dim() == 3:
                value_buffer = vb
        except Exception:
            value_buffer = None

    Hkv, Dk = key_buffer.shape[1], key_buffer.shape[2]
    Hq = int(getattr(layer, "tp_q_head_num", 0)) or Hkv
    if Hq % max(Hkv, 1) != 0:
        return
    group = Hq // Hkv
    scale = float(getattr(layer, "scaling", 1.0 / (Dk ** 0.5)))

    qf = q.reshape(-1, Hq, Dk)                              # [ntok, Hq, Dk]
    indptr = kv_indptr.to(torch.long)
    bs = indptr.numel() - 1
    lid = int(layer.layer_id)

    recorded_any = False
    for r in range(min(bs, _MAX_REQ)):
        start, end = int(indptr[r]), int(indptr[r + 1])
        seq = end - start
        if seq < _MIN_SEQ or r >= qf.shape[0]:
            continue
        idx = kv_indices[start:end].to(torch.long)
        K = key_buffer[idx]                                # [seq, Hkv, Dk] (kv dtype)
        V = value_buffer[idx] if value_buffer is not None else None  # [seq, Hkv, Dv]
        qr = qf[r].float()                                 # [Hq, Dk]
        # Per-KV-head to AVOID materializing [seq, Hq, Dk] (~16GB @1M) -> OOM.
        # Each GQA group shares one KV head; peak is one [seq, Dk] fp32 (~0.5GB).
        for kh in range(Hkv):
            Kk = K[:, kh, :].float()                       # [seq, Dk]
            qsub = qr[kh * group:(kh + 1) * group, :]      # [group, Dk]
            scores = (qsub @ Kk.t()) * scale               # [group, seq]
            probs = torch.softmax(scores, dim=-1)          # [group, seq]
            for W in _W_LIST:
                win = min(W, seq)
                mass_recent = probs[:, seq - win:].sum(dim=-1)   # [group]
                for S in _S_LIST:
                    sk = min(S, max(seq - win, 0))          # sink disjoint from window
                    mass = mass_recent + (probs[:, :sk].sum(dim=-1) if sk > 0 else 0.0)
                    acc = _ACC.setdefault((lid, int(W), int(S)), [0.0, 0.0])
                    acc[0] += float(mass.sum().item())
                    acc[1] += float(group)

            # ---- tail diffuseness at the reference (Wref, Sref) ----
            rwin = min(_WREF, seq)
            rsk = min(_SREF, max(seq - rwin, 0))
            tlen = seq - rwin - rsk
            if tlen >= 2:
                tail = probs.clone()               # [group, seq]
                tail[:, seq - rwin:] = 0.0
                if rsk > 0:
                    tail[:, :rsk] = 0.0
                tmass = tail.sum(dim=-1)                        # [group]
                p = tail / tmass.clamp_min(1e-12).unsqueeze(-1)  # renorm over tail
                ent = -(p * p.clamp_min(1e-12).log()).sum(dim=-1)   # nats
                norm_ent = ent / math.log(tlen)                # [group] in [0,1]
                pr_frac = 1.0 / ((p * p).sum(dim=-1).clamp_min(1e-12) * tlen)  # PR/len
                tacc = _TAIL.setdefault(lid, [0.0, 0.0, 0.0, 0.0])
                tacc[0] += float(tmass.sum().item())
                tacc[1] += float(norm_ent.sum().item())
                tacc[2] += float(pr_frac.sum().item())
                tacc[3] += float(group)
                del tail, p

                # ---- V-aware OUTPUT contribution at (Wref,Sref) ----
                if V is not None:
                    Vk = V[:, kh, :].float()                       # [seq, Dv]
                    o_full = probs @ Vk                            # [group, Dv]
                    nfull = o_full.norm(dim=-1).clamp_min(1e-12)   # [group]
                    winmask = torch.zeros(seq, dtype=torch.float32, device=probs.device)
                    winmask[seq - rwin:] = 1.0
                    if rsk > 0:
                        winmask[:rsk] = 1.0
                    pw = probs * winmask                           # [group, seq]
                    o_win_unnorm = pw @ Vk                         # [group, Dv]
                    wmass = pw.sum(dim=-1).clamp_min(1e-12)        # [group]
                    o_win = o_win_unnorm / wmass.unsqueeze(-1)     # renormalized over win+sink
                    win_perturb = (o_full - o_win).norm(dim=-1) / nfull       # [group]
                    tail_out_frac = (o_full - o_win_unnorm).norm(dim=-1) / nfull  # [group]
                    vn = Vk.norm(dim=-1)                           # [seq] per-key ||v||
                    vn_win = float(vn[seq - rwin:].mean().item())
                    vn_tail = float(vn[rsk:seq - rwin].mean().item())
                    vacc = _VOUTACC.setdefault(lid, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    vacc[0] += float(win_perturb.sum().item())
                    vacc[1] += float(tail_out_frac.sum().item())
                    vacc[3] += vn_win * group
                    vacc[4] += vn_tail * group
                    vacc[5] += float(group)
                    if rsk > 0:
                        vacc[2] += float(vn[:rsk].mean().item()) * group
                        vacc[6] += float(group)
                    del Vk, o_full, o_win_unnorm, o_win, pw, vn, winmask
            del Kk, scores, probs
        sacc = _SEQ.setdefault(lid, [0.0, 0.0])
        sacc[0] += float(seq)
        sacc[1] += 1.0
        recorded_any = True
        del K
        if V is not None:
            del V

    if recorded_any:
        _STATE["passes"] += 1
        # Dump EVERY pass (tiny JSON): the sglang scheduler subprocess is killed
        # by SIGTERM on shutdown, which skips atexit -> a once-at-exit dump would
        # be lost. Quiet (verbose=False) to avoid log spam; the final atexit dump
        # is verbose.
        dump(verbose=False)
        if _STATE["fired_log"] < 3:
            _STATE["fired_log"] += 1
            print(
                "[attn-mass] FIRED pass=%d layer_id=%s Hq=%d Hkv=%d group=%d "
                "scale=%.4g bs=%d (W_LIST=%s S_LIST=%s)"
                % (_STATE["passes"], layer.layer_id, Hq, Hkv, group, scale, bs,
                   _W_LIST, _S_LIST),
                flush=True,
            )


def dump(verbose: bool = True) -> None:
    # re-entrant: called on every pass (quiet) AND at exit (verbose). Always
    # overwrites with the latest accumulation so a SIGTERM'd process still leaves
    # a complete file.
    if not _ACC:
        return
    per_layer = []
    for (lid, W, S), (sm, n) in sorted(_ACC.items()):
        per_layer.append({"layer_id": lid, "W": W, "sink": S,
                          "in_window_mass": sm / n if n else None, "n": int(n)})
    # aggregate across layers (equal weight per head-sample)
    agg: Dict[Tuple[int, int], List[float]] = {}
    for (lid, W, S), (sm, n) in _ACC.items():
        a = agg.setdefault((W, S), [0.0, 0.0])
        a[0] += sm
        a[1] += n
    overall = [{"W": W, "sink": S, "in_window_mass": sm / n if n else None, "n": int(n)}
               for (W, S), (sm, n) in sorted(agg.items())]
    seq_mean = {str(lid): (sm / n if n else None) for lid, (sm, n) in _SEQ.items()}
    # tail diffuseness: per-layer + overall (weighted by head-samples)
    tail_per_layer = {str(lid): {"tail_mass": s0 / n, "tail_norm_entropy": s1 / n,
                                 "tail_participation_frac": s2 / n, "n": int(n)}
                      for lid, (s0, s1, s2, n) in _TAIL.items() if n}
    tt = [0.0, 0.0, 0.0, 0.0]
    for _lid, (s0, s1, s2, n) in _TAIL.items():
        tt[0] += s0; tt[1] += s1; tt[2] += s2; tt[3] += n
    tail_overall = ({"W": _WREF, "sink": _SREF, "tail_mass": tt[0] / tt[3],
                     "tail_norm_entropy": tt[1] / tt[3],
                     "tail_participation_frac": tt[2] / tt[3], "n": int(tt[3])}
                    if tt[3] else None)
    # V-aware output metrics: per-layer + overall
    def _vrec(win_p, tail_f, vs, vw, vt, n, ns):
        return {"win_perturb": win_p / n, "tail_out_frac": tail_f / n,
                "vnorm_win": vw / n, "vnorm_tail": vt / n,
                "vnorm_sink": (vs / ns if ns else None), "n": int(n)}
    vout_per_layer = {str(lid): _vrec(*v) for lid, v in _VOUTACC.items() if v[5]}
    vt2 = [0.0] * 7
    for _lid, v in _VOUTACC.items():
        for k in range(7):
            vt2[k] += v[k]
    vout_overall = (_vrec(vt2[0], vt2[1], vt2[2], vt2[3], vt2[4], vt2[5], vt2[6])
                    if vt2[5] else None)
    out = {
        "tag": os.environ.get("RK_ATTN_MASS_TAG", "run"),
        "model": os.environ.get("RK_ATTN_MASS_MODEL", ""),
        "passes_recorded": _STATE["passes"],
        "W_list": _W_LIST,
        "sink_list": _S_LIST,
        "seq_mean_per_layer": seq_mean,
        "overall": overall,
        "tail_ref": {"W": _WREF, "sink": _SREF},
        "tail_overall": tail_overall,
        "tail_per_layer": tail_per_layer,
        "vout_overall": vout_overall,
        "vout_per_layer": vout_per_layer,
        "per_layer": per_layer,
    }
    try:
        os.makedirs(os.path.dirname(_OUT), exist_ok=True)
        tmp = _OUT + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, _OUT)                       # atomic: never a half-written file
        if verbose:
            print("[attn-mass] wrote %s (passes=%d)" % (_OUT, _STATE["passes"]), flush=True)
            print("[attn-mass] OVERALL in-window mass:", flush=True)
            for rec in overall:
                print("  W=%-6d sink=%-3d  mass=%.4f  (n=%d)"
                      % (rec["W"], rec["sink"], rec["in_window_mass"], rec["n"]), flush=True)
            if tail_overall:
                print("[attn-mass] TAIL diffuseness @ Wref=%d Sref=%d: tail_mass=%.4f "
                      "norm_entropy=%.4f participation_frac=%.4f (1.0=uniform/diffuse=safe)"
                      % (_WREF, _SREF, tail_overall["tail_mass"],
                         tail_overall["tail_norm_entropy"],
                         tail_overall["tail_participation_frac"]), flush=True)
            if vout_overall:
                vs = vout_overall["vnorm_sink"]
                print("[attn-mass] V-OUTPUT @ Wref=%d Sref=%d: win_perturb=%.4f "
                      "tail_out_frac=%.4f  ||v||(sink/win/tail)=%s/%.3f/%.3f "
                      "(small win_perturb => windowing barely changes the output)"
                      % (_WREF, _SREF, vout_overall["win_perturb"],
                         vout_overall["tail_out_frac"],
                         ("%.3f" % vs) if vs is not None else "NA",
                         vout_overall["vnorm_win"], vout_overall["vnorm_tail"]),
                      flush=True)
    except Exception as e:  # pragma: no cover
        if verbose:
            print("[attn-mass] dump failed: %r" % e, flush=True)


if ENABLED:
    atexit.register(dump)
    print("[attn-mass] ENABLED (RK_ATTN_MASS=1): out=%s passes<=%d maxreq=%d minseq=%d"
          % (_OUT, _MAX_PASSES, _MAX_REQ, _MIN_SEQ), flush=True)
