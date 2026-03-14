"""Topological compressor — shorten braid generator sequences.

Three compression levels that preserve topological invariants:

* **Level 0** — Inverse cancellation: σᵢσᵢ⁻¹ → e.
* **Level 1** — Far-commutativity + re-cancel: cancel distant inverse pairs
  separated by commuting generators (|iₘ − iₚ| ≥ 2).
* **Level 2** — Yang–Baxter BFS: apply YBE rewrites σᵢσᵢ₊₁σᵢ ↔ σᵢ₊₁σᵢσᵢ₊₁
  then re-compress via L0+L1, tracking the shortest result.

All levels preserve writhe, Jones polynomial, and matrix trace.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from braidcodec._exceptions import CompressionError, RewriteVerificationError
from braidcodec.algebra.braid_equations import (
    BraidEquation,
    contract_braid_tensor,
    jones_polynomial,
    simplify_braid,
    writhe,
)
from braidcodec.algebra.yang_baxter import verify_yang_baxter_morphism

if TYPE_CHECKING:
    from braidcodec.codec.schema import EncodedStream
    from braidcodec.crypto.keys import BraidKey


# ── Level 0 — Inverse cancellation ───────────────────────────────────────


def _compress_level0(braid: BraidEquation) -> BraidEquation:
    """Cancel adjacent inverse pairs σᵢσᵢ⁻¹ → e.

    Wraps the existing ``simplify_braid`` implementation.
    """
    return simplify_braid(braid)


# ── Level 1 — Far-commutativity + re-cancel ──────────────────────────────


def _compress_level1(braid: BraidEquation) -> BraidEquation:
    """Cancel distant inverse pairs separated by commuting generators.

    For each cancellable pair (p, q) where gens[p] == -gens[q], check if
    every generator between p and q commutes with gens[p] — i.e.
    |index(gens[m]) − index(gens[p])| ≥ 2 for all p < m < q.

    If so, bubble gens[q] leftward through the commuting region and cancel
    with gens[p].  Repeat until stable, then re-run Level 0.
    """
    gens = list(braid.generators)

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(gens):
            target = -gens[i]
            idx_i = abs(gens[i])
            # Scan forward for the nearest matching inverse.
            for j in range(i + 1, len(gens)):
                if gens[j] == target and all(
                    abs(abs(gens[m]) - idx_i) >= 2 for m in range(i + 1, j)
                ):
                    del gens[j]
                    del gens[i]
                    changed = True
                    break
                # If we hit a non-commuting generator with the same index,
                # this pair can't be cancelled via far-commutativity.
                if abs(abs(gens[j]) - idx_i) < 2 and gens[j] != target:
                    break
            if changed:
                break
            i += 1

    # Re-run Level 0 on the result to catch any newly adjacent pairs.
    result = BraidEquation(
        braid.n_strands,
        gens,
        sector=braid.sector,
        _sector_params=braid._sector_params,
    )
    return _compress_level0(result)


# ── Level 2 — Yang–Baxter BFS ────────────────────────────────────────────


def _find_ybe_sites(gens: list[int]) -> list[tuple[int, list[int]]]:
    """Find all YBE-rewritable sites in a generator sequence.

    Detects patterns (i, i+1, i) and (-(i), -(i+1), -(i)) and returns
    candidate rewrites as (position, replacement_triple).
    """
    sites: list[tuple[int, list[int]]] = []
    for p in range(len(gens) - 2):
        a, b, c = gens[p], gens[p + 1], gens[p + 2]
        # YBE: (i, i±1, i) ↔ (i±1, i, i±1) — positive or all-negative triples
        if (
            a == c
            and abs(b - a) == 1
            and ((a > 0 and b > 0 and c > 0) or (a < 0 and b < 0 and c < 0))
        ):
            sites.append((p, [b, a, b]))
    return sites


def _compress_level2(
    braid: BraidEquation,
    *,
    max_rewrites: int = 100,
) -> BraidEquation:
    """Apply YBE rewrites via bounded BFS, seeking the shortest result.

    At each state, find all YBE sites, apply each rewrite, compress via
    Level 0 + Level 1, and keep the shortest.  Bounded by ``max_rewrites``
    total states explored.
    """
    best = _compress_level1(_compress_level0(braid))
    best_len = len(best.generators)

    # BFS queue: list of generator sequences to explore.
    queue: list[list[int]] = [list(braid.generators)]
    seen: set[tuple[int, ...]] = {tuple(braid.generators)}
    states_explored = 0

    while queue and states_explored < max_rewrites:
        current_gens = queue.pop(0)
        sites = _find_ybe_sites(current_gens)

        for pos, replacement in sites:
            if states_explored >= max_rewrites:
                break
            states_explored += 1

            # Apply YBE rewrite.
            new_gens = list(current_gens)
            new_gens[pos : pos + 3] = replacement

            key = tuple(new_gens)
            if key in seen:
                continue
            seen.add(key)

            # Compress the rewritten sequence.
            candidate_braid = BraidEquation(
                braid.n_strands,
                new_gens,
                sector=braid.sector,
                _sector_params=braid._sector_params,
            )
            compressed = _compress_level1(_compress_level0(candidate_braid))

            if len(compressed.generators) < best_len:
                best = compressed
                best_len = len(compressed.generators)

            # Enqueue the rewritten (un-compressed) form for further exploration.
            queue.append(new_gens)

    return best


# ── Public API — low-level ────────────────────────────────────────────────


def compress_braid(
    braid: BraidEquation,
    *,
    level: int = 1,
    max_rewrites: int = 100,
) -> BraidEquation:
    """Compress a single braid equation.

    Parameters
    ----------
    braid:
        The braid to compress.
    level:
        Compression level (0, 1, or 2).
    max_rewrites:
        Maximum BFS states for Level 2.

    Returns
    -------
    BraidEquation
        A topologically equivalent braid with (ideally) fewer generators.

    Raises
    ------
    CompressionError
        If level is out of range.
    RewriteVerificationError
        If the compressed result is not matrix-equivalent to the original.
    """
    if level not in {0, 1, 2}:
        raise CompressionError(
            f"Compression level must be 0, 1, or 2, got {level}",
            level=level,
        )

    if not braid.generators:
        return braid

    if level == 0:
        result = _compress_level0(braid)
    elif level == 1:
        result = _compress_level1(braid)
    else:
        result = _compress_level2(braid, max_rewrites=max_rewrites)

    # Verify topological equivalence via matrix contraction.
    if result.generators and braid.generators and not verify_yang_baxter_morphism(braid, result):
        raise RewriteVerificationError(
            "Compressed braid is not matrix-equivalent to original",
            original_len=len(braid.generators),
            compressed_len=len(result.generators),
        )

    return result


# ── Public API — stream-level ─────────────────────────────────────────────


def compress(
    stream: EncodedStream,
    key: BraidKey,
    *,
    level: int = 1,
    max_rewrites: int = 100,
) -> EncodedStream:
    """Compress an encoded stream's braid generator sequences.

    For each block, applies the requested compression level.  If the block
    is shortened, the original generators are stored in ``decode_generators``
    so the decoder can recover the original bytes.

    Parameters
    ----------
    stream:
        An ``EncodedStream`` produced by ``encode()``.
    key:
        The ``BraidKey`` used during encoding.
    level:
        Compression level (0, 1, or 2).
    max_rewrites:
        Maximum BFS states for Level 2.

    Returns
    -------
    EncodedStream
        A new stream with compressed generator sequences and updated
        invariant values recomputed from the compressed braids.
    """
    from braidcodec.codec.schema import EncodedBlock

    sector_params = key.sector_params
    new_blocks: list[EncodedBlock] = []

    for block in stream.blocks:
        if not block.generators:
            new_blocks.append(block)
            continue

        original_braid = BraidEquation(
            block.n_strands,
            block.generators,
            sector=block.sector,
            _sector_params=sector_params,
        )
        compressed = compress_braid(
            original_braid,
            level=level,
            max_rewrites=max_rewrites,
        )

        # If compression didn't shorten, keep as-is.
        if len(compressed.generators) >= len(block.generators):
            new_blocks.append(block)
            continue

        # Recompute invariants from the compressed braid.
        w = writhe(compressed)
        jones_real: float | None = None
        jones_imag: float | None = None
        trace_real: float | None = None
        trace_imag: float | None = None

        if block.invariant_tier == 2:
            j = jones_polynomial(compressed)
            jones_real = j.real
            jones_imag = j.imag
        elif block.invariant_tier == 3:
            matrix = contract_braid_tensor(compressed)
            tr = complex(np.trace(matrix))
            trace_real = tr.real
            trace_imag = tr.imag

        new_block = EncodedBlock(
            generators=compressed.generators,
            n_strands=block.n_strands,
            sector=block.sector,
            writhe=w,
            block_index=block.block_index,
            original_length=block.original_length,
            invariant_tier=block.invariant_tier,
            jones_real=jones_real,
            jones_imag=jones_imag,
            trace_real=trace_real,
            trace_imag=trace_imag,
            decode_generators=block.effective_decode_generators,
        )
        new_blocks.append(new_block)

    return replace(stream, blocks=tuple(new_blocks))


# ── Utility ───────────────────────────────────────────────────────────────


def compression_ratio(original: EncodedStream, compressed: EncodedStream) -> float:
    """Compute the compression ratio (compressed / original generator count).

    Returns 1.0 if the original has no generators (no compression possible).
    """
    orig_total = sum(len(b.generators) for b in original.blocks)
    comp_total = sum(len(b.generators) for b in compressed.blocks)
    if orig_total == 0:
        return 1.0
    return comp_total / orig_total
