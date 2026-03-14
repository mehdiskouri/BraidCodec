"""Decoder — full pipeline from ``EncodedStream`` back to bytes.

Reverses the encode path: per-block generator extraction → writhe pre-check →
optional invariant recomputation → mixed-radix inverse → chunk reassembly →
final BLAKE3 checksum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import blake3
import numpy as np

from braidcodec._exceptions import (
    ChecksumError,
    JonesError,
    KeyMismatchError,
    TraceError,
    WritheError,
)
from braidcodec._types import JONES_TOLERANCE, MATRIX_TOLERANCE
from braidcodec.algebra.braid_equations import (
    BraidEquation,
    contract_braid_tensor,
    jones_polynomial,
    writhe,
)
from braidcodec.codec.chunker import generators_to_bytes

if TYPE_CHECKING:
    from braidcodec.codec.schema import EncodedBlock, EncodedStream
    from braidcodec.crypto.keys import BraidKey


# ── Helpers ───────────────────────────────────────────────────────────────


def _check_writhe(block: EncodedBlock, sector_params: dict[str, float] | None) -> None:
    """Recompute writhe and raise ``WritheError`` on mismatch."""
    braid = BraidEquation(
        block.n_strands,
        block.generators,
        sector=block.sector,
        _sector_params=sector_params,
    )
    actual = writhe(braid)
    if actual != block.writhe:
        raise WritheError(
            f"Block {block.block_index}: writhe mismatch "
            f"(stored={block.writhe}, computed={actual})",
            block_index=block.block_index,
            expected=block.writhe,
            actual=actual,
        )


def _check_invariant(block: EncodedBlock, sector_params: dict[str, float] | None) -> None:
    """Recompute the tier-appropriate invariant and raise on mismatch."""
    braid = BraidEquation(
        block.n_strands,
        block.generators,
        sector=block.sector,
        _sector_params=sector_params,
    )

    if block.invariant_tier == 2:
        stored = block.jones
        if stored is None:
            raise JonesError(
                f"Block {block.block_index}: tier 2 but no Jones value stored",
                block_index=block.block_index,
            )
        actual = jones_polynomial(braid)
        if abs(actual - stored) > JONES_TOLERANCE:
            raise JonesError(
                f"Block {block.block_index}: Jones mismatch "
                f"|Δ|={abs(actual - stored):.2e} > {JONES_TOLERANCE}",
                block_index=block.block_index,
                expected=stored,
                actual=actual,
            )
    elif block.invariant_tier == 3:
        stored = block.trace_invariant
        if stored is None:
            raise TraceError(
                f"Block {block.block_index}: tier 3 but no trace value stored",
                block_index=block.block_index,
            )
        matrix = contract_braid_tensor(braid)
        actual = complex(np.trace(matrix))
        if abs(actual - stored) > MATRIX_TOLERANCE:
            raise TraceError(
                f"Block {block.block_index}: trace mismatch "
                f"|Δ|={abs(actual - stored):.2e} > {MATRIX_TOLERANCE}",
                block_index=block.block_index,
                expected=stored,
                actual=actual,
            )
    # Tier 1: writhe only — no additional invariant to check.


# ── Public API ────────────────────────────────────────────────────────────


def decode(
    stream: EncodedStream,
    key: BraidKey,
    *,
    verify: bool = True,
) -> bytes:
    """Decode an ``EncodedStream`` back to the original bytes.

    Parameters
    ----------
    stream:
        Encoded payload (from ``encode`` or ``EncodedStream.from_bytes``).
    key:
        The ``BraidKey`` used during encoding.
    verify:
        If ``True`` (default), recompute and verify Jones/trace invariants
        per block.  Writhe is **always** checked regardless of this flag.

    Returns
    -------
    bytes
        The original plaintext data.

    Raises
    ------
    KeyMismatchError
        If the key's sector or strand count doesn't match the stream.
    WritheError
        If any block's writhe doesn't match recomputation.
    JonesError
        If a tier-2 block's Jones polynomial doesn't match.
    TraceError
        If a tier-3 block's matrix trace doesn't match.
    ChecksumError
        If the reassembled data's BLAKE3 digest doesn't match.
    """
    # ── Step 1: key validation ────────────────────────────────────────────
    if stream.sector != key.sector:
        raise KeyMismatchError(
            f"Sector mismatch: stream='{stream.sector}', key='{key.sector}'",
            stream_sector=stream.sector,
            key_sector=key.sector,
        )
    if stream.n_strands != key.n_strands:
        raise KeyMismatchError(
            f"Strand count mismatch: stream={stream.n_strands}, key={key.n_strands}",
            stream_n_strands=stream.n_strands,
            key_n_strands=key.n_strands,
        )

    sector_params = key.sector_params

    # ── Step 2: sort blocks defensively ───────────────────────────────────
    sorted_blocks = sorted(stream.blocks, key=lambda b: b.block_index)

    # ── Step 3: per-block decode ──────────────────────────────────────────
    chunks: list[bytes] = []
    for block in sorted_blocks:
        # Writhe pre-check — always, O(k)
        _check_writhe(block, sector_params)

        # Invariant check — optional but on by default
        if verify:
            _check_invariant(block, sector_params)

        # Inverse mapping: generators → bytes
        chunk = generators_to_bytes(
            block.effective_decode_generators,
            block.n_strands,
            block.original_length,
        )
        chunks.append(chunk)

    # ── Step 4: reassemble ────────────────────────────────────────────────
    result = b"".join(chunks)

    # ── Step 5: BLAKE3 final checksum ─────────────────────────────────────
    actual_checksum = blake3.blake3(result).digest()
    if actual_checksum != stream.checksum:
        raise ChecksumError(
            "Reassembled data BLAKE3 checksum mismatch",
            expected=stream.checksum.hex(),
            actual=actual_checksum.hex(),
        )

    return result
