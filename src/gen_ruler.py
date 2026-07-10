"""RULER-style long-context benchmark generator (self-contained, dependency-free).

Faithful re-implementation of the core RULER task families (Hsieh et al., 2024, arXiv:2404.06654)
so we can synthesize long-context inputs at ANY length (incl. 1M) for the windowed self-MTP study,
without the official repo's heavy deps. Covers the three RULER families:

  retrieval (NIAH):   niah_single, niah_multikey, niah_multivalue, niah_multiquery
  multi-hop tracing:  vt   (variable tracking)
  aggregation:        cwe  (common-word extraction), fwe (frequent-word extraction)

Output is JSONL with fields consumed by BOTH our harnesses:
  - context : the haystack (+needles).  bench_sglang_mtp.py --real-input tiles/slices this to exact ISL.
  - input   : the question.             sglang_al_workload.py concatenates "{context}\\n\\n{input}".
  - answer / outputs : gold (for AL/quality scoring later).
  - task, target_tokens, approx_tokens, depth(s).

Length is sized by WORDS using an estimated tokens/word ratio (default 1.35 for this noise corpus),
so the generator runs on a CPU login node with no transformers/torch. Pass --tok-per-word to tune,
or --measure-with <model_or_tok_dir> to record exact token counts (needs transformers). The bench
harness re-tokenizes and slices to exact ISL regardless, so approximate sizing is fine for throughput.

Examples:
  # pipeclean (instant, CPU): tiny 4k set, 4 samples, a few tasks
  python src/gen_ruler.py --lengths 4096 --num-samples 4 \
      --tasks niah_single,niah_multikey,vt,fwe --out-dir data/ruler

  # hero lengths for the smoke
  python src/gen_ruler.py --lengths 131072,262144,1040000 --num-samples 8 \
      --tasks niah_single,niah_multikey,niah_multivalue,vt,fwe --out-dir data/ruler
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string

# Short declarative noise sentences (RULER "needle-in-noise" haystack style).
NOISE = [
    "The grass is green.", "The sky is blue.", "The sun is yellow.",
    "Here we go.", "There and back again.", "The clouds drift slowly overhead.",
    "Birds sing in the early morning.", "The river flows down to the sea.",
    "Mountains rise in the far distance.", "The wind moves through the tall trees.",
    "A dog barked somewhere down the street.", "The kettle boiled on the stove.",
    "Leaves fell gently to the ground.", "The lamp flickered in the quiet room.",
    "Snow covered the silent fields.", "The train departed right on time.",
]

WORDS = (
    "apple river table cloud stone garden bottle planet candle window forest pencil "
    "silver copper marble valley meadow harbor anchor pillow ticket basket lantern "
    "compass feather thunder crystal velvet pepper sugar ribbon puzzle saddle"
).split()


def rnd_key(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(6, 9)))


def rnd_val(rng: random.Random) -> str:
    return str(rng.randint(1000000, 9999999))


def _approx_tokens(text: str, tok_per_word: float) -> int:
    return int(len(text.split()) * tok_per_word)


def _haystack_words_needed(target_tokens: int, tok_per_word: float) -> int:
    return max(1, int(target_tokens / tok_per_word))


# Filler mode controls the GENERATION-ENTROPY (and thus the draft AL) of the haystack.
# At the throughput corner (bench --real-input slices `context` as a flat token buffer),
# AL is driven by filler entropy, NOT by the NIAH/vt/fwe task type -> this is the real
# easy/medium/hard knob:
#   noise -> repetitive declarative sentences   -> HIGH AL (easy)
#   words -> shuffled diverse-vocab word salad  -> MEDIUM AL
#   prose -> real natural-language text (books) -> LOW AL (hard, genuine)
HAYSTACK = "noise"
_PROSE_SENTS: list[str] = []


def _load_prose(paths: list[str]) -> list[str]:
    import re as _re
    sents: list[str] = []
    for p in paths:
        try:
            txt = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for s in _re.split(r"(?<=[.!?])\s+", txt.replace("\n", " ")):
            s = s.strip()
            if len(s.split()) >= 4:
                sents.append(s)
    return sents


def _filler_sentence(rng: random.Random) -> str:
    if HAYSTACK == "prose" and _PROSE_SENTS:
        return rng.choice(_PROSE_SENTS)
    if HAYSTACK == "words":
        return " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 16))) + "."
    return rng.choice(NOISE)


def build_haystack(rng: random.Random, target_tokens: int, tok_per_word: float,
                   inserts: list[tuple[float, str]]) -> str:
    """Build a haystack of ~target_tokens words (filler per HAYSTACK mode) and splice
    `inserts` at depth fractions."""
    need_words = _haystack_words_needed(target_tokens, tok_per_word)
    sents: list[str] = []
    w = 0
    while w < need_words:
        s = _filler_sentence(rng)
        sents.append(s)
        w += len(s.split())
    for depth, text in sorted(inserts, key=lambda x: x[0]):
        pos = min(len(sents), max(0, int(depth * len(sents))))
        sents.insert(pos, text)
    return " ".join(sents)


# ---- RULER task families -------------------------------------------------------------------

def gen_niah_single(rng, target_tokens, tpw):
    key, val = rnd_key(rng), rnd_val(rng)
    needle = f"One of the special magic numbers for {key} is: {val}."
    ctx = build_haystack(rng, target_tokens, tpw, [(rng.uniform(0.1, 0.9), needle)])
    q = f"What is the special magic number for {key} mentioned in the provided text?"
    return ctx, q, [val]


def gen_niah_multikey(rng, target_tokens, tpw, n_distract=3):
    key, val = rnd_key(rng), rnd_val(rng)
    needles = [f"One of the special magic numbers for {key} is: {val}."]
    for _ in range(n_distract):
        needles.append(f"One of the special magic numbers for {rnd_key(rng)} is: {rnd_val(rng)}.")
    inserts = [(rng.uniform(0.05, 0.95), n) for n in needles]
    ctx = build_haystack(rng, target_tokens, tpw, inserts)
    q = f"What is the special magic number for {key} mentioned in the provided text?"
    return ctx, q, [val]


def gen_niah_multivalue(rng, target_tokens, tpw, n_val=4):
    key = rnd_key(rng)
    vals = [rnd_val(rng) for _ in range(n_val)]
    inserts = [(rng.uniform(0.05, 0.95), f"One of the special magic numbers for {key} is: {v}.")
               for v in vals]
    ctx = build_haystack(rng, target_tokens, tpw, inserts)
    q = (f"What are all the special magic numbers for {key} mentioned in the provided text? "
         f"List every one of them.")
    return ctx, q, vals


def gen_niah_multiquery(rng, target_tokens, tpw, n_q=4):
    keys = [rnd_key(rng) for _ in range(n_q)]
    vals = [rnd_val(rng) for _ in range(n_q)]
    inserts = [(rng.uniform(0.05, 0.95), f"One of the special magic numbers for {k} is: {v}.")
               for k, v in zip(keys, vals)]
    ctx = build_haystack(rng, target_tokens, tpw, inserts)
    q = ("What are the special magic numbers for the following keys mentioned in the provided text: "
         + ", ".join(keys) + "?")
    return ctx, q, vals


def gen_vt(rng, target_tokens, tpw, chain_len=4, n_chains=2):
    """Variable tracking: X1=NUM; X2=X1; ... ; find all vars equal to the target value."""
    target_val = rnd_val(rng)
    inserts, gold = [], []
    for c in range(n_chains):
        names = [f"VAR_{rng.choice(string.ascii_uppercase)}{rng.randint(10,99)}" for _ in range(chain_len)]
        val = target_val if c == 0 else rnd_val(rng)
        stmts = [f"{names[0]} = {val}."]
        for i in range(1, chain_len):
            stmts.append(f"{names[i]} = {names[i-1]}.")
        if c == 0:
            gold = names
        for s in stmts:
            inserts.append((rng.uniform(0.05, 0.95), s))
    ctx = build_haystack(rng, target_tokens, tpw, inserts)
    q = (f"Find all variables that are assigned the value {target_val}, directly or through a chain "
         f"of assignments. List all such variable names.")
    return ctx, q, gold


def gen_cwe(rng, target_tokens, tpw, n_common=10, common_freq=30):
    """Common-word extraction (aggregation): some words appear far more often; extract them."""
    common = rng.sample(WORDS, k=min(n_common, len(WORDS)))
    need_words = _haystack_words_needed(target_tokens, tpw)
    seq = []
    for w in common:
        seq += [w] * common_freq
    while len(seq) < need_words:
        seq.append(rng.choice(WORDS))
    rng.shuffle(seq)
    ctx = " ".join(seq)
    q = (f"Among the words in the list above, which {n_common} words appear most frequently? "
         f"List the {n_common} most common words.")
    return ctx, q, common


def gen_fwe(rng, target_tokens, tpw, top_k=3):
    """Frequent-word extraction (aggregation, zipfian): return the top-k most frequent words."""
    need_words = _haystack_words_needed(target_tokens, tpw)
    vocab = WORDS[:]
    rng.shuffle(vocab)
    weights = [1.0 / (i + 1) for i in range(len(vocab))]  # zipfian
    seq = rng.choices(vocab, weights=weights, k=need_words)
    counts = {}
    for w in seq:
        counts[w] = counts.get(w, 0) + 1
    gold = [w for w, _ in sorted(counts.items(), key=lambda x: -x[1])[:top_k]]
    ctx = " ".join(seq)
    q = f"What are the {top_k} most frequently appearing words in the list above?"
    return ctx, q, gold


TASKS = {
    "niah_single": gen_niah_single,
    "niah_multikey": gen_niah_multikey,
    "niah_multivalue": gen_niah_multivalue,
    "niah_multiquery": gen_niah_multiquery,
    "vt": gen_vt,
    "cwe": gen_cwe,
    "fwe": gen_fwe,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="4096",
                    help="comma-sep target token lengths, e.g. 4096,131072,1040000")
    ap.add_argument("--tasks", default=",".join(TASKS),
                    help=f"comma-sep subset of: {','.join(TASKS)}")
    ap.add_argument("--num-samples", type=int, default=8, help="examples per (task,length)")
    ap.add_argument("--tok-per-word", type=float, default=1.35,
                    help="estimated tokens/word for sizing (noise corpus ~1.35 on Qwen BPE)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/ruler")
    ap.add_argument("--haystack", default="noise", choices=["noise", "words", "prose"],
                    help="filler entropy = difficulty: noise(easy)/words(medium)/prose(hard)")
    ap.add_argument("--prose-files", default="data/book1.txt,data/book2.txt",
                    help="comma-sep real-text files for --haystack prose")
    ap.add_argument("--label", default="",
                    help="optional filename tag (e.g. easy/medium/hard) -> {task}_{label}_{L}.jsonl")
    # Answer-length knobs (for qa-mode: longer answers dominate the decode window so
    # AL/TPOT reflect task difficulty rather than first-token transients).
    ap.add_argument("--niah-vals", type=int, default=4, help="niah_multivalue: # values to list")
    ap.add_argument("--niah-queries", type=int, default=4, help="niah_multiquery: # keys to query/list")
    ap.add_argument("--fwe-topk", type=int, default=3,
                    help="fwe: # top frequent words to return (capped at the 36-word vocab)")
    ap.add_argument("--cwe-common", type=int, default=10, help="cwe: # common words to list")
    ap.add_argument("--vt-chain", type=int, default=4, help="vt: chain length")
    ap.add_argument("--vt-chains", type=int, default=2, help="vt: # chains")
    ap.add_argument("--measure-with", default="",
                    help="optional model/tokenizer dir; if set, record EXACT token counts (needs transformers)")
    args = ap.parse_args()

    global HAYSTACK, _PROSE_SENTS
    HAYSTACK = args.haystack
    if args.haystack == "prose":
        _PROSE_SENTS = _load_prose([p.strip() for p in args.prose_files.split(",") if p.strip()])
        if not _PROSE_SENTS:
            raise SystemExit(f"--haystack prose but no sentences loaded from {args.prose_files}")

    measure = None
    if args.measure_with:
        from transformers import AutoTokenizer
        measure = AutoTokenizer.from_pretrained(args.measure_with, trust_remote_code=True)

    os.makedirs(args.out_dir, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in TASKS:
            raise SystemExit(f"unknown task '{t}'; choose from {','.join(TASKS)}")

    task_kw = {
        "niah_multivalue": {"n_val": args.niah_vals},
        "niah_multiquery": {"n_q": args.niah_queries},
        "fwe": {"top_k": args.fwe_topk},
        "cwe": {"n_common": args.cwe_common},
        "vt": {"chain_len": args.vt_chain, "n_chains": args.vt_chains},
    }
    for L in lengths:
        for t in tasks:
            rng = random.Random(f"{args.seed}-{t}-{L}-{args.haystack}")
            lab = f"_{args.label}" if args.label else ""
            path = os.path.join(args.out_dir, f"{t}{lab}_{L}.jsonl")
            n_written = 0
            with open(path, "w") as fh:
                for i in range(args.num_samples):
                    ctx, q, gold = TASKS[t](random.Random(rng.random()), L, args.tok_per_word,
                                            **task_kw.get(t, {}))
                    rec = {
                        "task": t, "target_tokens": L, "haystack": args.haystack,
                        "label": args.label,
                        "approx_tokens": _approx_tokens(ctx + " " + q, args.tok_per_word),
                        "context": ctx, "input": q,
                        "answer": gold[0] if len(gold) == 1 else None,
                        "outputs": gold,
                    }
                    if measure is not None:
                        rec["exact_tokens"] = len(measure(ctx + "\n\n" + q,
                                                          add_special_tokens=False).input_ids)
                    fh.write(json.dumps(rec) + "\n")
                    n_written += 1
            tail = ""
            if measure is None:
                # cheap report from the last record
                tail = f"~{rec['approx_tokens']} tok (est)"
            else:
                tail = f"{rec['exact_tokens']} tok (exact)"
            print(f"[ruler] wrote {n_written:>3} x {t:<16} L={L:<8} -> {path}  ({tail})")


if __name__ == "__main__":
    main()
