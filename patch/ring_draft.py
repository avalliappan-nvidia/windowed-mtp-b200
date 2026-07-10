"""[Windowed-MTP] draft KV *ring buffer*.

This turns the windowed-MTP draft's KV pool from a full-length buffer (physically
`max_total_num_tokens` slots, of which only `sink + W` are ever read) into a compact
per-request ring of `S + W + D` slots, reclaiming the draft pool's ~8-11% of total KV.

Design (kernel-computed ring):
  * The draft keeps SHARING the target's `req_to_token` map and the KV allocator
    (so mamba spec-state handling on the GDN/Mamba2 models is untouched). Only the
    draft's *separate* KV pool is shrunk, and the draft's KV indices / write locations
    are remapped by a pure function of (req_pool_index, token_position).
  * ring_slot(r, p) = r * slots_per_req + ( p < S ? p : S + ((p - S) mod (W + D)) )
      - sink band  [0, S)          : positions 0..S-1, pinned, written once.
      - recent band[S, S + W + D)  : circular over the last (W + D) positions.
    Reads only ever touch [seq - W, seq + D] (window + speculative tree) U sink, a span
    of <= W + D, so within one modulus period => all distinct => never read an evicted
    slot. RoPE is baked per-token into K, so reusing a physical slot for a new position
    is score-correct.
  * The target verify path is untouched => output distribution is exactly the target's
    => lossless (bit-exact under greedy).

Everything here is inert unless RK_DRAFT_RING=1. The window / sink are the same knobs
used by the compute-only windowing (RK_MTP_WINDOW / RK_MTP_SINK), so the ring is a
*memory* realization of the exact same windowed draft.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils.common import get_bool_env_var, get_int_env_var


@dataclass(frozen=True)
class RingConfig:
    enabled: bool
    window: int  # W: most-recent tokens the draft attends to
    sink: int  # S: attention-sink tokens pinned at the front
    draft_tokens: int  # D: speculative tree tokens per verify (reserved beyond window)

    @property
    def modulus(self) -> int:
        # recent-band period: window + tree reservation so tree writes never clobber
        # a still-in-window slot.
        return self.window + self.draft_tokens

    @property
    def slots_per_req(self) -> int:
        return self.sink + self.window + self.draft_tokens

    def pool_size(self, max_num_reqs: int) -> int:
        return self.slots_per_req * int(max_num_reqs)


def resolve_ring_config(draft_tokens: int) -> RingConfig:
    """Read the ring config from env. `draft_tokens` = speculative_num_draft_tokens."""
    enabled = get_bool_env_var("RK_DRAFT_RING")
    window = get_int_env_var("RK_MTP_WINDOW", 0)
    sink = get_int_env_var("RK_MTP_SINK", 0)
    # Ring requires a finite window; if RK_DRAFT_RING is set without a window it is a
    # config error (the pool would still have to be full-length).
    if enabled and window <= 0:
        raise ValueError(
            "RK_DRAFT_RING=1 requires RK_MTP_WINDOW>0 (the ring buffer is a memory "
            "realization of the draft KV window)."
        )
    return RingConfig(
        enabled=enabled,
        window=int(window),
        sink=int(sink),
        draft_tokens=int(draft_tokens),
    )


def ring_slot_torch(
    req_pool_index: torch.Tensor,
    position: torch.Tensor,
    cfg: RingConfig,
) -> torch.Tensor:
    """Vectorized ring-slot mapping. `req_pool_index` and `position` broadcast together.

    Returns int64 physical slot ids into the compact draft pool.
    """
    S = cfg.sink
    M = cfg.modulus
    spr = cfg.slots_per_req
    req_pool_index = req_pool_index.to(torch.int64)
    position = position.to(torch.int64)
    base = req_pool_index * spr
    recent = S + torch.remainder(position - S, M)
    within = torch.where(position < S, position, recent)
    return base + within


# ---------------------------------------------------------------------------
# Draft-extend READ: windowed + ring prefix indices
# ---------------------------------------------------------------------------
#
# The draft-extend attention re-reads the request's prefix KV. Under the ring
# that prefix physically lives in only `sink + W` distinct slots, so we cannot
# gather the full [0, seq) span (it would index the compact pool out of bounds
# *and* re-read evicted positions). Instead we emit, per request, exactly the
# kept positions -- sink [0, s_eff) followed by the recent window
# [seq - recent, seq) -- each mapped to its ring slot. The window is `sink + W`
# (the D tree tokens are the *new* extend queries, written separately via
# out_cache_loc, so they are not part of the prefix key set). Ring-slot
# arithmetic still uses modulus M = W + D so it matches how those positions were
# physically written.


@triton.jit
def _ring_windowed_kv_indices_kernel(
    req_pool_indices,  # [bs] int
    seq_lens,  # [bs] int (length per req over which to window)
    kv_indptr,  # [bs+1] int32, prefix-sum of kept lengths (precomputed)
    kv_indices,  # [>= total_kept] int64 (output; may be an oversized graph buffer)
    sink: tl.constexpr,  # S
    keep_cap: tl.constexpr,  # max kept positions per req (S+W eager prefix; S+W+D graph)
    modulus: tl.constexpr,  # M = W + D
    slots_per_req: tl.constexpr,  # S + W + D
    CAP: tl.constexpr,  # next-pow2 >= keep_cap (loop bound)
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    seq = tl.load(seq_lens + bid)
    kept = tl.minimum(seq, keep_cap)
    s_eff = tl.minimum(sink, seq)
    recent = kept - s_eff
    recent_start = seq - recent
    out_base = tl.load(kv_indptr + bid)
    ring_base = tl.load(req_pool_indices + bid).to(tl.int64) * slots_per_req
    for off in range(0, CAP, BLOCK):
        j = off + tl.arange(0, BLOCK)
        mask = j < kept
        # position kept at output index j: sink band then recent band
        pos = tl.where(j < s_eff, j, recent_start + (j - s_eff))
        recent_slot = sink + (pos - sink) % modulus
        slot = ring_base + tl.where(pos < sink, pos, recent_slot).to(tl.int64)
        tl.store(kv_indices + out_base + j, slot, mask=mask)


def fill_ring_windowed_kv_indices(
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    cfg: RingConfig,
    keep_cap: int,
    kv_indptr_out: torch.Tensor,
    kv_indices_out: torch.Tensor,
) -> None:
    """In-place windowed+ring index build (no allocation, no host sync).

    Writes `kv_indptr_out[0..bs]` (cumsum of kept lengths) and
    `kv_indices_out[0..total)` (ring slots). Works with oversized persistent
    buffers (cuda-graph path). `seq_lens` is the span to window per req;
    `keep_cap` = S+W for the eager prefix read, S+W+D for the graph read (which
    includes the D tree tokens in kv_indices).
    """
    bs = req_pool_indices.shape[0]
    kept = torch.clamp(seq_lens.to(torch.int64), max=keep_cap)
    kv_indptr_out[0] = 0
    kv_indptr_out[1 : bs + 1] = torch.cumsum(kept, dim=0).to(kv_indptr_out.dtype)
    cap = triton.next_power_of_2(max(keep_cap, 1))
    _ring_windowed_kv_indices_kernel[(bs,)](
        req_pool_indices,
        seq_lens,
        kv_indptr_out,
        kv_indices_out,
        cfg.sink,
        keep_cap,
        cfg.modulus,
        cfg.slots_per_req,
        CAP=cap,
        BLOCK=min(cap, 256),
    )


def build_ring_windowed_kv_indices(
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    cfg: RingConfig,
    device: torch.device,
):
    """Eager draft-extend prefix read: allocate and build (kv_indices, kv_indptr).

    `prefix_lens` = per-request prefix length (excl. the D tree tokens, which the
    eager path supplies via out_cache_loc). keep_cap = S+W.
    """
    bs = req_pool_indices.shape[0]
    keep_cap = cfg.sink + cfg.window
    kept = torch.clamp(prefix_lens.to(torch.int64), max=keep_cap)
    kv_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
    total = int(kept.sum().item())
    kv_indices = torch.empty((max(total, 1),), dtype=torch.int64, device=device)
    if total == 0:
        kv_indptr[1:] = torch.cumsum(kept, dim=0).to(torch.int32)
        return kv_indices[:0], kv_indptr
    fill_ring_windowed_kv_indices(
        req_pool_indices, prefix_lens, cfg, keep_cap, kv_indptr, kv_indices
    )
    return kv_indices, kv_indptr


# ---------------------------------------------------------------------------
# WRITE remaps (out_cache_loc)
# ---------------------------------------------------------------------------


def ring_extend_out_cache_loc(
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    num_draft_tokens: int,
    cfg: RingConfig,
) -> torch.Tensor:
    """out_cache_loc for the draft-extend-for-decode write.

    The extend writes KV for the `num_draft_tokens` committed positions
    [prefix_len, prefix_len + D) of each request, laid out as contiguous per-req
    blocks (matching batch.input_ids). Returns int64 ring slots, shape [bs * D].
    """
    device = req_pool_indices.device
    bs = req_pool_indices.shape[0]
    k = torch.arange(num_draft_tokens, device=device).view(1, num_draft_tokens)
    pos = prefix_lens.to(torch.int64).view(bs, 1) + k  # [bs, D]
    req = req_pool_indices.view(bs, 1).expand(bs, num_draft_tokens)
    return ring_slot_torch(req, pos, cfg).reshape(-1)


def ring_out_cache_loc_from_positions(
    reqs_per_token: torch.Tensor,
    positions: torch.Tensor,
    cfg: RingConfig,
) -> torch.Tensor:
    """out_cache_loc for a token-flat batch (used by prefill-fill).

    `reqs_per_token[i]` = req_pool_index of token i, `positions[i]` = its true
    (RoPE) position. Under the ring the KV write for a token goes to its ring
    slot. Prefill writes every prefill position (wrapping in the recent band),
    which is value-correct because draft K/V are pure projections of each
    position's input (target hidden + embedding), independent of attention.
    """
    return ring_slot_torch(reqs_per_token, positions, cfg)


# ---------------------------------------------------------------------------
# Support guard
# ---------------------------------------------------------------------------


def assert_ring_supported(server_args, cfg: RingConfig) -> None:
    """Fail loudly on config combos the PoC ring does not yet remap.

    Decode tree read/write (in-kernel) and the draft-extend read (capture/replay
    index build) are cuda-graph-safe; the extend/prefill writes flow through
    out_cache_loc which the graph copies each replay. topk>1 with page_size>1 is
    still not mapped (paged tree slot layout).
    """
    if not cfg.enabled:
        return
    # The extend/prefill prefix-read remap targets forward_metadata.kv_indices,
    # which the *triton* backend reads at kernel launch. The flashinfer backend
    # bakes indices into a planned wrapper at init time, so the override is a
    # no-op there and the compact pool overflows (OOB) on long-context extends.
    draft_bk = (
        getattr(server_args, "speculative_draft_attention_backend", None)
        or getattr(server_args, "attention_backend", None)
    )
    if draft_bk not in ("triton", None):
        raise ValueError(
            f"RK_DRAFT_RING=1 (PoC) requires --draft-attention-backend triton "
            f"(got {draft_bk!r}); the flashinfer draft backend plans kv_indices "
            f"at init, so the windowed+ring prefix-read override does not apply "
            f"and the compact pool overflows at long context."
        )
    topk = getattr(server_args, "speculative_eagle_topk", 1) or 1
    page_size = getattr(server_args, "page_size", 1) or 1
    if topk > 1 and page_size > 1:
        raise ValueError(
            "RK_DRAFT_RING=1 (PoC) does not support speculative_eagle_topk>1 with "
            "page_size>1 (paged tree slot mapping is not ring-remapped)."
        )
