"""SGLang NEXTN (self-MTP) wall-clock benchmark for Qwen3.x-MoE on one B200 (TP1).

Measures, at a fixed (batch B, input length ISL):
  - TPOT  : per-committed-output-token wall time of the batch running concurrently.
  - AL    : acceptance length = completion_tokens / spec_verify_ct (from meta_info).
  - tok/s : B / TPOT  (aggregate decode throughput).

TPOT is obtained by a TWO-POINT SLOPE to cancel the (huge, at long ctx) prefill:
    wall(N) = prefill(B,ISL) + (N/AL) * t_iter
    => TPOT = d(wall)/dN = (wall_hi - wall_lo) / (N_hi - N_lo)
Each generate() call prefills fresh, so prefill cancels in the difference. With
temperature 0 the committed-token count per seq == max_new_tokens exactly.

Self-MTP (NEXTN) needs NO separate draft model: the MTP weights ship in the
target checkpoint. The MTP draft attention is the window/sink edit target
(qwen3_5.py RadixAttention, gated on is_nextn) for the speedup phase; this script
also forwards RK_MTP_WINDOW/RK_MTP_SINK via env so the same harness works there.

One config per process (clean GPU); appends one JSONL line.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", choices=["none", "nextn"], default="nextn")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--isl", type=int, default=131072)
    ap.add_argument("--n-decode", type=int, default=384,
                    help="committed output tokens timed in the pure-decode phase")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--draft-tokens", type=int, default=5)
    ap.add_argument("--attention-backend", default="flashinfer")
    ap.add_argument("--draft-attention-backend", default="")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor-parallel size (>1 needs matching --gres=gpu:N); ring is TP-invariant")
    ap.add_argument("--mem-frac", type=float, default=0.85)
    ap.add_argument("--mamba-scheduler-strategy", default="extra_buffer")
    # GDN/mamba-hybrid memory knobs. SGLang reserves mamba_full_memory_ratio/(1+r)
    # of post-weights memory for the mamba STATE pool (default r=0.9 => 47%!),
    # which starves the KV pool. Lower r => bigger KV pool => more concurrent reqs.
    ap.add_argument("--mamba-full-memory-ratio", type=float, default=-1.0,
                    help=">=0 overrides SGLang default 0.9 (use ~0.1 to free KV pool; "
                         "0.0 = no full-mamba reserve). -1 (default) = leave SGLang default.")
    ap.add_argument("--max-mamba-cache-size", type=int, default=0,
                    help=">0 sets mamba cache slots explicitly (small => more KV)")
    ap.add_argument("--max-running-requests", type=int, default=0,
                    help=">0 caps running reqs (also shrinks spec mamba-intermediate reserve)")
    ap.add_argument("--max-total-tokens", type=int, default=0,
                    help=">0 forces the KV pool token budget (max_total_tokens). Reclaims "
                         "HBM the spec-mode estimator leaves idle at long context so >1 "
                         "long-context request fits (breaks the B_eff=1 cap).")
    ap.add_argument("--schedule-conservativeness", type=float, default=-1.0,
                    help="<1 admits new reqs more aggressively (concurrency); -1=SGLang default 1.0")
    ap.add_argument("--chunked-prefill-size", type=int, default=0,
                    help="bigger => fewer prefill chunks => faster serial prefill; 0=SGLang default")
    ap.add_argument("--page-size", type=int, default=0)
    ap.add_argument("--disable-cuda-graph", action="store_true")
    ap.add_argument("--disable-radix-cache", action="store_true",
                    help="required for NemotronH + spec (no_buffer mamba path)")
    ap.add_argument("--yarn-factor", type=float, default=0.0,
                    help=">0 enables YaRN rope scaling (4.0 extends native 262144 -> ~1.05M)")
    ap.add_argument("--json-override", default="",
                    help="raw json_model_override_args passthrough (overrides --yarn-factor)")
    ap.add_argument("--dtype", default="")
    ap.add_argument("--real-input", default="",
                    help="path to an InfiniteBench jsonl (e.g. passkey.jsonl); when set, feed REAL "
                         "context tokens (sliced to exact ISL, distinct per seq) instead of random "
                         "ids -> realistic, stable AL for the best-case windowed-MTP corner")
    ap.add_argument("--real-stride", type=int, default=512,
                    help="per-sequence start-offset stride into the real token buffer (distinct "
                         "offsets break radix prefix-sharing across the batch)")
    ap.add_argument("--qa-mode", action="store_true",
                    help="REAL-RESPONSE mode: build each prompt from a RULER record's context+question "
                         "(chat-templated, add_generation_prompt) and DECODE THE ACTUAL ANSWER with "
                         "ignore_eos=False. AL/TPOT then reflect task difficulty (answer-token "
                         "predictability), unlike the fixed-decode continuation path. Use --isl as the "
                         "prompt token budget (records head/tail-truncated to fit); --qa-max-new caps answer.")
    ap.add_argument("--qa-max-new", type=int, default=512,
                    help="qa-mode: max answer tokens to generate (EOS may stop earlier)")
    ap.add_argument("--batch-same", action="store_true",
                    help="B>1: feed the SAME record to every sequence (fixed content) instead of "
                         "cycling distinct records. FORCES --disable-radix-cache (else identical "
                         "prompts collapse to one shared KV -> fake infinite capacity). Gives a clean, "
                         "content-invariant batch-concurrency sweep: constant AL across B, honest B x "
                         "full-context KV, no record-mix drift and no wrap-around dedup when B exceeds "
                         "the record count. Which record: --batch-dup-idx (default = median length).")
    ap.add_argument("--batch-dup-idx", type=int, default=-1,
                    help="--batch-same: index of the record to duplicate; -1 (default) = the "
                         "median-token-length record (representative, matches the hero B=1 choice).")
    ap.add_argument("--force-ignore-eos", action="store_true",
                    help="override the qa-mode default and decode with ignore_eos=True even when a "
                         "real question prompt is built. Use ONLY to probe forced-continuation AL on "
                         "models that EOS too early for a genuine window (e.g. Nemotron on non-cwe): "
                         "pair with --repetition-penalty and check the AL warm-up ramp before trusting "
                         "the number, since forced continuation is the confound the QA path retired.")
    ap.add_argument("--no-multimodal", action="store_true",
                    help="disable the vision tower/processor (text-only bench; needed for a "
                         "local patched-config dir lacking preprocessor_config.json)")
    ap.add_argument("--kv-cache-dtype", default="",
                    help="override KV-cache dtype (e.g. bfloat16) regardless of the checkpoint's "
                         "quant config; used to standardize KV to bf16 across models (the nvidia "
                         "122B ships FP8 KV, the RedHat 35B ships bf16). '' = auto (checkpoint default)")
    ap.add_argument("--random-seed", type=int, default=-1,
                    help="pin sglang server RNG seed (>=0). Under temperature=0 (greedy) this "
                         "does NOT change argmax token selection; it only removes seed as a "
                         "variable across runs. <0 = leave sglang's per-run auto seed.")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="multiplicative repetition penalty on the target logits (1.0 = off; "
                         ">1 e.g. 1.1-1.3 discourages the greedy repetition loops that appear in "
                         "long decodes). Applied identically to draft+target so spec stays lossless "
                         "w.r.t. the penalized distribution; still deterministic under temperature=0.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-output", action="store_true",
                    help="write the generated answer text of the timed run to <out>.gentext.json "
                         "(for native-vs-window output diff / losslessness check)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import sglang as sgl
    from sglang.version import __version__ as sgl_version

    ctx = args.isl + (args.qa_max_new if args.qa_mode else args.n_decode) + 64
    kw = dict(
        model_path=args.model,
        mem_fraction_static=args.mem_frac,
        context_length=ctx,
        tp_size=args.tp,
        trust_remote_code=True,
        attention_backend=args.attention_backend,
        mamba_scheduler_strategy=args.mamba_scheduler_strategy,
        cuda_graph_max_bs=args.batch,
        decode_log_interval=1,  # log gen throughput every decode step (TPOT source)
        log_level="info",       # ensure scheduler "Decode batch ... gen throughput" logs emit
    )
    if args.dtype:
        kw["dtype"] = args.dtype
    if args.kv_cache_dtype:
        kw["kv_cache_dtype"] = args.kv_cache_dtype
    if args.no_multimodal:
        kw["enable_multimodal"] = False
    # YaRN rope scaling to push past the native 262144 context cap. The config
    # nests rope under text_config.rope_parameters (transformers>=5 format); we
    # preserve the mRoPE fields and flip rope_type -> yarn. --json-override lets
    # us iterate on the exact format if the build expects a different key.
    if args.json_override:
        kw["json_model_override_args"] = args.json_override
    elif args.yarn_factor > 0:
        ext = int(262144 * args.yarn_factor)
        rope = {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "yarn",
            "factor": args.yarn_factor,
            "original_max_position_embeddings": 262144,
        }
        # transformers>=5 validate_rope() runs on the (multimodal) config after
        # rope standardization and reads .max_position_embeddings, which the outer
        # config lacks -> set it at BOTH levels so the yarn validator finds it.
        override = {
            "max_position_embeddings": ext,
            "rope_parameters": rope,
            "text_config": {"max_position_embeddings": ext, "rope_parameters": rope},
        }
        kw["json_model_override_args"] = json.dumps(override)
    if args.mamba_full_memory_ratio >= 0:
        kw["mamba_full_memory_ratio"] = args.mamba_full_memory_ratio
    if args.max_mamba_cache_size > 0:
        kw["max_mamba_cache_size"] = args.max_mamba_cache_size
    if args.max_running_requests > 0:
        kw["max_running_requests"] = args.max_running_requests
    if args.max_total_tokens > 0:
        kw["max_total_tokens"] = args.max_total_tokens
    if args.random_seed >= 0:
        kw["random_seed"] = args.random_seed
    if args.schedule_conservativeness >= 0:
        kw["schedule_conservativeness"] = args.schedule_conservativeness
    if args.chunked_prefill_size > 0:
        kw["chunked_prefill_size"] = args.chunked_prefill_size
    if args.page_size > 0:
        kw["page_size"] = args.page_size
    if args.disable_cuda_graph:
        kw["disable_cuda_graph"] = True
    if args.batch_same and not args.disable_radix_cache:
        args.disable_radix_cache = True   # identical prompts + radix ON -> all B share one KV (fake capacity); force off
        print("[batch-same] forcing --disable-radix-cache (identical prompts must not share KV)", flush=True)
    if args.disable_radix_cache:
        kw["disable_radix_cache"] = True
    if args.algo == "nextn":
        kw.update(
            speculative_algorithm="NEXTN",
            speculative_num_steps=args.steps,
            speculative_eagle_topk=args.topk,
            speculative_num_draft_tokens=args.draft_tokens,
        )
        if args.draft_attention_backend:
            kw["speculative_draft_attention_backend"] = args.draft_attention_backend

    window = os.environ.get("RK_MTP_WINDOW", "")
    sink = os.environ.get("RK_MTP_SINK", "")

    t0 = time.time()
    eng = sgl.Engine(**kw)
    load_s = time.time() - t0

    # REAL-INPUT mode: build one long token buffer from a real corpus (e.g. the
    # InfiniteBench passkey filler, which is highly repetitive -> trivially
    # predictable -> high, STABLE AL with real tokens). Each sequence takes a
    # distinct exact-ISL slice at a different start offset, so (a) lengths match
    # the random path exactly (clean cross-method TPOT/B_eff) and (b) distinct
    # token-0 breaks radix prefix-sharing across the batch. This benchmarks the
    # best-case corner (real AL, no random-token AL noise) vs synthetic random.
    real_ids = None
    qa_prompts = None
    input_name = "random"
    if args.real_input and args.qa_mode:
        # QA mode: one prompt per RULER record = chat_template(context + question),
        # truncated to ISL keeping the TAIL (preserves late context + question + the
        # generation prompt). The model then decodes the ACTUAL answer (ignore_eos=False),
        # so AL/TPOT reflect answer-token difficulty rather than haystack continuation.
        from transformers import AutoTokenizer
        input_name = os.path.basename(args.real_input).replace(".jsonl", "")
        rtok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        def build_prompt_ids(text):
            # Render the chat template to a STRING then tokenize to a plain list[int].
            # This model's template (Qwen-VL family) drops string content unless it is
            # given as typed parts, so try list-content first, then string, and finally
            # a manual "Answer:" wrapper. Validate the rendered string actually contains
            # the body (len >= body) before trusting it.
            s = None
            for content in ([{"type": "text", "text": text}], text):
                try:
                    cand = rtok.apply_chat_template(
                        [{"role": "user", "content": content}],
                        add_generation_prompt=True, tokenize=False)
                except Exception:
                    cand = None
                if cand and len(cand) >= len(text):
                    s = cand
                    break
            if not s:
                s = text + "\n\nAnswer:"
            ids = rtok(s, add_special_tokens=False).input_ids
            return list(ids)

        recs = []
        with open(args.real_input) as fh:
            for line in fh:
                r = json.loads(line)
                c = str(r.get("context", "")); q = str(r.get("input", ""))
                if not c and not q:
                    continue
                text = (c + "\n\n" + q).strip() if q else c
                ids = build_prompt_ids(text)
                if len(ids) > args.isl:
                    ids = ids[-args.isl:]
                recs.append(ids)
        if not recs:
            raise SystemExit(f"--real-input {args.real_input} produced no qa prompts")
        qa_prompts = recs
        lens = sorted(len(x) for x in recs)
        print(f"[qa-input] {input_name}: {len(recs)} prompts, tok len "
              f"min/med/max={lens[0]}/{lens[len(lens)//2]}/{lens[-1]} (isl cap={args.isl})", flush=True)
    elif args.real_input:
        from transformers import AutoTokenizer
        input_name = os.path.basename(args.real_input).replace(".jsonl", "")
        rtok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        need = args.isl + (args.batch - 1) * args.real_stride + 64
        buf: list[int] = []
        with open(args.real_input) as fh:
            for line in fh:
                r = json.loads(line)
                txt = str(r.get("context", "")) or str(r.get("input", ""))
                if not txt:
                    continue
                buf.extend(rtok(txt, add_special_tokens=False).input_ids)
                if len(buf) >= need:
                    break
        if not buf:
            raise SystemExit(f"--real-input {args.real_input} produced no tokens")
        while len(buf) < need:  # tile if the corpus is shorter than B x ISL needs
            buf.extend(buf[: need - len(buf)])
        real_ids = buf
        print(f"[real-input] {input_name}: buffer={len(real_ids)} tok, stride={args.real_stride}, "
              f"need={need} (isl={args.isl} x B={args.batch})", flush=True)

    # Exact-length prompts: real distinct slices, or synthetic random ids
    # (avoid low special-token ids).
    def make_prompts(seed):
        if qa_prompts is not None:
            if args.batch_same:
                idx = args.batch_dup_idx
                if idx < 0:   # median-token-length record (representative)
                    order = sorted(range(len(qa_prompts)), key=lambda i: len(qa_prompts[i]))
                    idx = order[len(order) // 2]
                idx %= len(qa_prompts)
                print(f"[batch-same] duplicating record idx={idx} (len={len(qa_prompts[idx])} tok) "
                      f"across B={args.batch} (radix off -> honest B x KV)", flush=True)
                return [qa_prompts[idx] for _ in range(args.batch)]
            return [qa_prompts[b % len(qa_prompts)] for b in range(args.batch)]
        if real_ids is not None:
            st = args.real_stride
            if args.batch_same:   # same slice for every sequence (fixed content)
                return [real_ids[: args.isl] for _ in range(args.batch)]
            return [real_ids[b * st : b * st + args.isl] for b in range(args.batch)]
        rng = np.random.default_rng(seed)
        return [
            rng.integers(low=1000, high=120000, size=args.isl).tolist()
            for _ in range(args.batch)
        ]

    def flush():
        try:
            eng.flush_cache()
        except Exception:
            pass

    def gen(prompts, n_new, ignore_eos=True):
        sp = {"temperature": 0.0, "max_new_tokens": n_new, "ignore_eos": ignore_eos}
        if args.repetition_penalty and args.repetition_penalty != 1.0:
            sp["repetition_penalty"] = args.repetition_penalty
        t = time.time()
        outs = eng.generate(input_ids=prompts, sampling_params=sp)
        dt = time.time() - t
        comp = sum(o.get("meta_info", {}).get("completion_tokens", 0) for o in outs)
        vct = sum(o.get("meta_info", {}).get("spec_verify_ct", 0) or 0 for o in outs)
        cached = sum(o.get("meta_info", {}).get("cached_tokens", 0) or 0 for o in outs)
        return dt, comp, vct, cached, outs

    # PRIMARY TPOT = SGLang-native steady-state decode throughput. The scheduler
    # logs `Decode batch, #running-req: B, ... gen throughput (token/s): X` every
    # decode step (decode_log_interval=1). When all B reqs decode concurrently
    # (#running-req==B), X is the prefill-free decode rate, so TPOT = B / X. For
    # spec decode X already counts ACCEPTED tokens/s, so B/X is TPOT per OUTPUT
    # token directly. We bracket the timed decode with sentinels and a sibling
    # parser (parse_gt.py) extracts the median X at #running-req==B per tag.
    # decode_wall/prefill_wall are still recorded as a cross-check.
    import sys
    # Lightweight warmup: a TINY prompt is enough to trigger JIT/autotune (CUDA
    # graphs are captured at engine init). A full-ISL warmup was a wasted multi-
    # million-token prefill at long context.
    warm = [list(range(1000, 1064)) for _ in range(args.batch)]
    gen(warm, 8)
    flush()
    # ONE timed run = one full prefill + the timed decode. parse_gt.py extracts the
    # prefill-free steady-state decode throughput from the scheduler log, so we do
    # NOT need a separate prefill-only measurement (it was a 2nd full prefill).
    prompts = make_prompts(1)
    pf_dt = float("nan")
    n_new_timed = args.qa_max_new if args.qa_mode else args.n_decode
    sys.stderr.write(f"RK_DEC_BEGIN tag={args.tag} B={args.batch}\n"); sys.stderr.flush()
    ignore_eos = (not args.qa_mode) or args.force_ignore_eos
    dec_dt, comp, vct, cached, timed_outs = gen(prompts, n_new_timed, ignore_eos=ignore_eos)
    sys.stderr.write(f"RK_DEC_END tag={args.tag}\n"); sys.stderr.flush()
    if getattr(args, "dump_output", False):
        try:
            dump_path = args.out + ".gentext.json"
            import json as _json
            recs = [{"seq": i,
                     "completion_tokens": o.get("meta_info", {}).get("completion_tokens", 0),
                     "spec_verify_ct": o.get("meta_info", {}).get("spec_verify_ct", 0),
                     # per-req accept histogram: H[n] = #verify steps that accepted exactly n
                     # draft tokens (SGLang meta_info; alias spec_accept_histogram). Raw data for
                     # per-position marginal/conditional acceptance (see accept_hist in point rec).
                     "spec_correct_drafts_histogram": (
                         o.get("meta_info", {}).get("spec_correct_drafts_histogram")
                         or o.get("meta_info", {}).get("spec_accept_histogram")),
                     "finish_reason": o.get("meta_info", {}).get("finish_reason", None),
                     "text": o.get("text", "")} for i, o in enumerate(timed_outs)]
            with open(dump_path, "w") as _f:
                _json.dump(recs, _f, ensure_ascii=False, indent=1)
            print(f"[dump-output] wrote generated text -> {dump_path}", flush=True)
        except Exception as _e:
            print(f"[dump-output] FAILED: {_e}", flush=True)
    print(f"[diag] decode_wall={dec_dt:.3f}s  cached_tokens={cached} "
          f"(isl*batch={args.isl*args.batch})  [TPOT via parse_gt]", flush=True)

    n_per_seq = comp / args.batch if args.batch else float("nan")
    decode_only_s = dec_dt - pf_dt
    # cross-check TPOT via prefill-subtraction (noise-sensitive at long ctx)
    tpot_sub_ms = (decode_only_s / n_per_seq * 1000.0) if n_per_seq else float("nan")
    tpot_s = decode_only_s / n_per_seq if n_per_seq else float("nan")
    tpot_ms = tpot_sub_ms
    tok_s = (comp / decode_only_s) if decode_only_s > 0 else float("nan")
    accept_len = (comp / vct) if vct else 1.0

    # ---- per-position acceptance from the SGLang accept histogram --------------------
    # H[n] = # verify steps that accepted exactly n draft tokens (summed over all reqs).
    # marginal m_j   = P(depth j accepted)          = (sum_{n>=j} H[n]) / total
    # conditional a_j= P(accept j | reached j)       = (sum_{n>=j} H[n]) / (sum_{n>=j-1} H[n])
    # AL cross-check = 1 + (sum_n n*H[n]) / total     (should match accept_len above)
    hist_agg = []
    for o in timed_outs:
        mi = o.get("meta_info", {})
        h = mi.get("spec_correct_drafts_histogram") or mi.get("spec_accept_histogram")
        if not h:
            continue
        if len(h) > len(hist_agg):
            hist_agg.extend([0] * (len(h) - len(hist_agg)))
        for n, c in enumerate(h):
            hist_agg[n] += int(c or 0)
    marginal_accept = conditional_accept = None
    al_from_hist = None
    if hist_agg and sum(hist_agg) > 0:
        total = sum(hist_agg)
        D = len(hist_agg) - 1                       # max accepted drafts per step
        # tail[j] = # steps with n >= j accepted (tail[0] = total)
        tail = [0] * (len(hist_agg) + 1)
        for j in range(len(hist_agg) - 1, -1, -1):
            tail[j] = tail[j + 1] + hist_agg[j]
        marginal_accept = [tail[j] / total for j in range(1, D + 1)]
        conditional_accept = [
            (tail[j] / tail[j - 1]) if tail[j - 1] > 0 else 0.0 for j in range(1, D + 1)
        ]
        al_from_hist = 1.0 + sum(n * c for n, c in enumerate(hist_agg)) / total

    rec = {
        "tag": args.tag,
        "sgl_version": sgl_version,
        "algo": args.algo,
        "model": args.model,
        "input": input_name,
        "batch": args.batch,
        "isl": args.isl,
        "steps": args.steps,
        "topk": args.topk,
        "draft_tokens": args.draft_tokens,
        "attention_backend": args.attention_backend,
        "draft_attention_backend": args.draft_attention_backend or args.attention_backend,
        "mtp_window": window,
        "mtp_sink": sink,
        "n_decode": args.n_decode,
        "qa_mode": bool(args.qa_mode),
        "qa_max_new": args.qa_max_new if args.qa_mode else 0,
        "ignore_eos": ignore_eos,
        "repetition_penalty": args.repetition_penalty,
        "comp_tokens": comp,
        "verify_ct": vct,
        "decode_wall_s": dec_dt,
        "prefill_wall_s": pf_dt,
        "decode_only_s": decode_only_s,
        "cached_tokens": cached,
        "comp_tokens": comp,
        "verify_ct": vct,
        "tpot_ms": tpot_ms,
        "tok_s": tok_s,
        # aggregate = tok_s (all B seqs); per-user = aggregate / batch. NOTE: derived from
        # the prefill-subtraction tpot_ms (noise-sensitive); the authoritative throughput is
        # in the _gt_ jsonl (tok_s_user/tok_s_gpu from parse_gt's steady-state TPOT).
        "tok_s_user": (tok_s / args.batch) if (args.batch and tok_s == tok_s) else float("nan"),
        "tok_s_gpu": tok_s,
        "accept_len": accept_len,
        "accept_hist": hist_agg,                 # H[n] summed over reqs
        "al_from_hist": al_from_hist,            # cross-check of accept_len
        "marginal_accept": marginal_accept,      # m_j, j=1..D
        "conditional_accept": conditional_accept,# a_j, j=1..D
        "load_s": load_s,
    }
    print(
        f"[{args.tag or args.algo} B{args.batch} isl{args.isl} bk={args.attention_backend}"
        f"/draft={rec['draft_attention_backend']} win={window or '-'} sink={sink or '-'}] "
        f"TPOT={tpot_ms:.3f} ms  AL={accept_len:.3f}  tok/s/gpu={tok_s:.1f}  "
        f"tok/s/user={(tok_s/args.batch) if (args.batch and tok_s==tok_s) else float('nan'):.1f}",
        flush=True,
    )
    if conditional_accept:
        print("[per-pos] conditional a_j=["
              + ", ".join(f"{a:.3f}" for a in conditional_accept)
              + f"]  marginal m_j=[" + ", ".join(f"{m:.3f}" for m in marginal_accept)
              + f"]  AL(hist)={al_from_hist:.3f}", flush=True)

    eng.shutdown()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
