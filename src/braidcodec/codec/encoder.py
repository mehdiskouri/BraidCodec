"""Encoder — full pipeline from bytes to ``EncodedStream``.

Maps raw data through chunking → braid construction → invariant computation
→ optional simplification → ``EncodedStream`` assembly.  Parallel block
encoding via ``ProcessPoolExecutor`` for multi-block payloads.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import blake3
import numpy as np

from braidcodec.algebra.braid_equations import (
    BraidEquation,
    contract_braid_tensor,
    jones_polynomial,
    simplify_braid,
    writhe,
)
from braidcodec.codec.chunker import (
    bytes_to_generators,
    chunk_stream,
    compute_block_size,
)
from braidcodec.codec.schema import EncodedBlock, EncodedStream

if TYPE_CHECKING:
    from braidcodec.crypto.keys import BraidKey

# ── Tier threshold ────────────────────────────────────────────────────────

_TIER_2_MAX_GENERATORS: int = 24


# ── Module-level block encoder (picklable for ProcessPoolExecutor) ────────


def _encode_block(
    chunk: bytes,
    n_strands: int,
    sector: str,
    sector_params: dict[str, float] | None,
    generators_per_block: int,
    block_index: int,
    original_length: int,
) -> EncodedBlock:
    """Encode a single chunk into an ``EncodedBlock``.

    This function is module-level (not a closure) so it can be pickled by
    ``ProcessPoolExecutor``.
    """
    generators = bytes_to_generators(chunk, n_strands, generators_per_block)

    braid = BraidEquation(
        n_strands,
        generators,
        sector=sector,
        _sector_params=sector_params,
    )

    matrix = contract_braid_tensor(braid)
    w = writhe(braid)

    # Tier selection: Jones (expensive state-sum) for short braids only.
    tier = 2 if len(generators) <= _TIER_2_MAX_GENERATORS else 3

    jones_real: float | None = None
    jones_imag: float | None = None
    trace_real: float | None = None
    trace_imag: float | None = None

    if tier == 2:
        j = jones_polynomial(braid)
        jones_real = j.real
        jones_imag = j.imag
    else:
        tr = complex(np.trace(matrix))
        trace_real = tr.real
        trace_imag = tr.imag

    simplified = simplify_braid(braid)

    return EncodedBlock(
        generators=generators,
        n_strands=n_strands,
        sector=sector,
        writhe=w,
        block_index=block_index,
        original_length=original_length,
        invariant_tier=tier,
        jones_real=jones_real,
        jones_imag=jones_imag,
        trace_real=trace_real,
        trace_imag=trace_imag,
    )


# ── Public API ────────────────────────────────────────────────────────────


def encode(
    data: bytes,
    key: BraidKey,
    *,
    generators_per_block: int = 32,
    max_workers: int | None = None,
) -> EncodedStream:
    """Encode raw data into an ``EncodedStream``.

    Parameters
    ----------
    data:
        Raw bytes to encode.
    key:
        A ``BraidKey`` controlling the R-matrix parametrisation.
    generators_per_block:
        Number of generators per block (default 32).
    max_workers:
        Maximum number of parallel workers.  ``None`` uses
        ``min(cpu_count()-1, 8)``.  Set to 1 for serial execution.

    Returns
    -------
    EncodedStream
        Complete encoded payload ready for wire serialization.
    """
    block_size = compute_block_size(key.n_strands, generators_per_block)
    checksum = blake3.blake3(data).digest()
    chunks = list(chunk_stream(data, block_size))

    sector_params = key.sector_params

    # Build argument tuples for _encode_block.
    args = [
        (chunk, key.n_strands, key.sector, sector_params, generators_per_block, idx, orig_len)
        for idx, chunk, orig_len in chunks
    ]

    # Determine effective worker count.
    if max_workers is None:
        effective_workers = min(max((os.cpu_count() or 1) - 1, 1), 8, len(args))
    else:
        effective_workers = min(max(max_workers, 1), 8, len(args))

    if len(args) <= 1 or effective_workers <= 1:
        blocks = [_encode_block(*a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures = [pool.submit(_encode_block, *a) for a in args]
            blocks = [f.result() for f in futures]

    return EncodedStream(
        blocks=tuple(blocks),
        n_strands=key.n_strands,
        sector=key.sector,
        total_bytes=len(data),
        checksum=checksum,
    )
