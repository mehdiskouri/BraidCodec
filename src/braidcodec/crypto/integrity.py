"""Integrity verification — 5-channel validation for ``EncodedStream``.

Channels (per Architecture §6):
    1. **Structural** — generator indices in valid range via ``validate_braid_category``
    2. **Writhe** — recompute crossing number, compare to stored value
    3. **Invariant** — Jones (tier 2) or matrix-trace (tier 3) recomputation
    4. **Fermion** (optional) — σᵢ → occupy(i), σᵢ⁻¹ → vacate(i); Pauli exclusion
    5. **Checksum** — decode all blocks → BLAKE3 vs ``stream.checksum``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import blake3
import numpy as np

from braidcodec._types import JONES_TOLERANCE, MATRIX_TOLERANCE
from braidcodec.algebra.braid_equations import (
    BraidEquation,
    contract_braid_tensor,
    jones_polynomial,
    writhe,
)
from braidcodec.algebra.fermion_bounds import (
    create_fermion_bounds,
    occupy,
    vacate,
    validate_fermionic_state,
)
from braidcodec.algebra.yang_baxter import validate_braid_category
from braidcodec.codec.chunker import generators_to_bytes

if TYPE_CHECKING:
    from braidcodec.codec.schema import EncodedBlock, EncodedStream
    from braidcodec.crypto.keys import BraidKey


# ── Result dataclass ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Aggregated result from all verification channels.

    Attributes
    ----------
    valid:
        ``True`` only if **every** enabled channel passed.
    structural_passed:
        All generator indices in ``[1, n_strands-1]``.
    writhe_passed:
        Writhe recomputation matches stored values.
    invariant_passed:
        Jones (tier 2) or trace (tier 3) recomputation matches.
    fermion_passed:
        Fermion occupation walk completed without Pauli exclusion
        violations.  ``None`` if the fermion channel was not run.
    checksum_passed:
        Reassembled plaintext BLAKE3 matches ``stream.checksum``.
    failed_blocks:
        Block indices that failed at least one check.
    details:
        Per-block diagnostic strings (only for failures).
    """

    valid: bool
    structural_passed: bool
    writhe_passed: bool
    invariant_passed: bool
    fermion_passed: bool | None
    checksum_passed: bool
    failed_blocks: tuple[int, ...] = ()
    details: tuple[str, ...] = field(default_factory=tuple)


# ── Channel implementations ──────────────────────────────────────────────


def _check_structural(
    block: EncodedBlock,
) -> tuple[bool, str | None]:
    """Channel 1: validate generator indices are in range."""
    try:
        braid = BraidEquation(
            block.n_strands,
            block.generators,
            sector=block.sector,
        )
    except (ValueError, TypeError):
        return False, f"Block {block.block_index}: structural validation failed"
    ok = validate_braid_category(braid)
    if not ok:
        return False, f"Block {block.block_index}: structural validation failed"
    return True, None


def _check_writhe(
    block: EncodedBlock,
    sector_params: dict[str, float] | None,
) -> tuple[bool, str | None]:
    """Channel 2: recompute writhe and compare."""
    braid = BraidEquation(
        block.n_strands,
        block.generators,
        sector=block.sector,
        _sector_params=sector_params,
    )
    actual = writhe(braid)
    if actual != block.writhe:
        return (
            False,
            f"Block {block.block_index}: writhe mismatch "
            f"(stored={block.writhe}, computed={actual})",
        )
    return True, None


def _check_invariant(
    block: EncodedBlock,
    sector_params: dict[str, float] | None,
) -> tuple[bool, str | None]:
    """Channel 3: recompute Jones (tier 2) or trace (tier 3)."""
    braid = BraidEquation(
        block.n_strands,
        block.generators,
        sector=block.sector,
        _sector_params=sector_params,
    )

    if block.invariant_tier == 2:
        stored = block.jones
        if stored is None:
            return False, f"Block {block.block_index}: tier 2 but no Jones value"
        actual = jones_polynomial(braid)
        delta = abs(actual - stored)
        if delta > JONES_TOLERANCE:
            return (
                False,
                f"Block {block.block_index}: Jones mismatch |Δ|={delta:.2e}",
            )
    elif block.invariant_tier == 3:
        stored = block.trace_invariant
        if stored is None:
            return False, f"Block {block.block_index}: tier 3 but no trace value"
        matrix = contract_braid_tensor(braid)
        actual = complex(np.trace(matrix))
        delta = abs(actual - stored)
        if delta > MATRIX_TOLERANCE:
            return (
                False,
                f"Block {block.block_index}: trace mismatch |Δ|={delta:.2e}",
            )
    # Tier 1: writhe only — nothing extra to check here.
    return True, None


def _check_fermion(block: EncodedBlock) -> tuple[bool, str | None]:
    """Channel 4: walk generators through fermion occupation model.

    σᵢ   (positive) → occupy site *i*
    σᵢ⁻¹ (negative) → vacate site *|i|*

    A Pauli exclusion violation (occupy already-occupied or vacate
    already-empty) is flagged as a failure.
    """
    n_sites = block.n_strands - 1  # generators act on strand pairs
    bounds = create_fermion_bounds(n_sites)

    for idx, gen in enumerate(block.generators):
        site = abs(gen)
        if site > n_sites:
            return (
                False,
                f"Block {block.block_index}: generator {gen} at position {idx} "
                f"exceeds fermion site count {n_sites}",
            )
        if gen > 0:
            occupy(bounds, site)
        else:
            vacate(bounds, site)

    # Validate final state consistency
    result = validate_fermionic_state(bounds)
    if not result.valid:
        return (
            False,
            f"Block {block.block_index}: fermion state validation failed "
            f"(parity_consistent={result.parity_consistent})",
        )
    return True, None


def _check_checksum(
    stream: EncodedStream,
    structurally_failed: set[int],
) -> tuple[bool, str | None]:
    """Channel 5: decode all blocks and verify BLAKE3 checksum.

    Blocks that failed structural validation are skipped — the checksum
    channel cannot run if any block has invalid generators.
    """
    if structurally_failed:
        return (
            False,
            f"BLAKE3 checksum skipped: structurally invalid blocks {sorted(structurally_failed)}",
        )

    sorted_blocks = sorted(stream.blocks, key=lambda b: b.block_index)
    chunks: list[bytes] = []
    for block in sorted_blocks:
        try:
            chunk = generators_to_bytes(
                block.effective_decode_generators,
                block.n_strands,
                block.original_length,
            )
        except (ValueError, TypeError, OverflowError):
            return False, f"BLAKE3 checksum failed: cannot decode block {block.block_index}"
        chunks.append(chunk)

    reassembled = b"".join(chunks)
    actual = blake3.blake3(reassembled).digest()
    if actual != stream.checksum:
        return False, "BLAKE3 checksum mismatch on reassembled data"
    return True, None


# ── Public API ────────────────────────────────────────────────────────────


def verify(
    stream: EncodedStream,
    key: BraidKey,
    *,
    fermion_check: bool = False,
) -> VerificationResult:
    """Run multi-channel integrity verification on an encoded stream.

    Parameters
    ----------
    stream:
        The ``EncodedStream`` to verify.
    key:
        The ``BraidKey`` used for encoding.
    fermion_check:
        If ``True``, run the fermion occupation channel (channel 4).
        Disabled by default since it only detects a narrow class of
        corruptions and adds overhead.

    Returns
    -------
    VerificationResult
        Aggregated result with per-channel pass/fail and diagnostics.
    """
    sector_params = key.sector_params
    failed_blocks: set[int] = set()
    structurally_failed: set[int] = set()
    details: list[str] = []

    structural_passed = True
    writhe_passed = True
    invariant_passed = True
    fermion_passed: bool | None = True if fermion_check else None
    checksum_passed = True

    sorted_blocks = sorted(stream.blocks, key=lambda b: b.block_index)

    for block in sorted_blocks:
        # Channel 1: structural — short-circuit on failure
        ok, msg = _check_structural(block)
        if not ok:
            structural_passed = False
            failed_blocks.add(block.block_index)
            structurally_failed.add(block.block_index)
            if msg:
                details.append(msg)
            continue  # skip downstream checks for this block

        # Channel 2: writhe
        ok, msg = _check_writhe(block, sector_params)
        if not ok:
            writhe_passed = False
            failed_blocks.add(block.block_index)
            if msg:
                details.append(msg)

        # Channel 3: invariant
        ok, msg = _check_invariant(block, sector_params)
        if not ok:
            invariant_passed = False
            failed_blocks.add(block.block_index)
            if msg:
                details.append(msg)

        # Channel 4: fermion (optional)
        if fermion_check:
            ok, msg = _check_fermion(block)
            if not ok:
                fermion_passed = False
                failed_blocks.add(block.block_index)
                if msg:
                    details.append(msg)

    # Channel 5: checksum (global, not per-block)
    ok, msg = _check_checksum(stream, structurally_failed)
    if not ok:
        checksum_passed = False
        if msg:
            details.append(msg)

    valid = (
        structural_passed
        and writhe_passed
        and invariant_passed
        and (fermion_passed is not False)
        and checksum_passed
    )

    return VerificationResult(
        valid=valid,
        structural_passed=structural_passed,
        writhe_passed=writhe_passed,
        invariant_passed=invariant_passed,
        fermion_passed=fermion_passed,
        checksum_passed=checksum_passed,
        failed_blocks=tuple(sorted(failed_blocks)),
        details=tuple(details),
    )
