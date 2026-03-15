"""Integrity verification — multi-channel validation for ``EncodedStream``.

Channels (per Architecture §6):
    1. **Structural** — generator indices in valid range via ``validate_braid_category``
    2. **Writhe** — recompute crossing number, compare to stored value
    3. **Invariant** — Jones (tier 2) or matrix-trace (tier 3) recomputation
    4. **Fermion** (optional) — σᵢ → occupy(i), σᵢ⁻¹ → vacate(i); Pauli exclusion
    5. **Topology** (optional) — recompute BFPS/layer profile, compare metadata
    6. **Checksum** — decode all blocks → BLAKE3 vs ``stream.checksum``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import blake3
import numpy as np

from braidcodec._exceptions import FormatError
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
from braidcodec.codec.chunker import compute_block_size, generators_to_bytes
from braidcodec.codec.preprocessing import (
    build_chunk_profiles,
    morton_key_1d,
    morton_key_2d,
    recover_legacy_generators,
    recover_legacy_generators_v2,
    topology_commitment,
    topology_commitment_v2,
)
from braidcodec.codec.reconstructive_compact import synthesize_reconstructive_bytes
from braidcodec.codec.reconstructive_solver import validate_reconstructive_solver_payload
from braidcodec.codec.reconstructive_transform import reconstructive_inverse_generators
from braidcodec.codec.schema import (
    parse_reconstructive_payload_metadata,
    validate_reconstructive_commitment_metadata,
    validate_reconstructive_metadata,
    validate_reconstructive_payload_metadata,
)

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
    topology_passed:
        Topology profile metadata matches recomputed BFPS/layer values.
        ``None`` if the topology channel was not run.
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
    topology_passed: bool | None
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
    """Channel 6: decode all blocks and verify BLAKE3 checksum.

    Blocks that failed structural validation are skipped — the checksum
    channel cannot run if any block has invalid generators.
    """
    if structurally_failed:
        return (
            False,
            f"BLAKE3 checksum skipped: structurally invalid blocks {sorted(structurally_failed)}",
        )

    if stream.metadata.get("preprocessing_mode") == "reconstructive" and len(stream.blocks) == 0:
        try:
            payload = parse_reconstructive_payload_metadata(stream.metadata)
            reassembled = synthesize_reconstructive_bytes(payload)
        except (ValueError, TypeError, OverflowError, FormatError):
            return False, "BLAKE3 checksum failed: cannot reconstruct compact stream"

        actual = blake3.blake3(reassembled).digest()
        if actual != stream.checksum:
            return False, "BLAKE3 checksum mismatch on reassembled data"
        return True, None

    sorted_blocks = sorted(stream.blocks, key=lambda b: b.block_index)
    chunks: list[bytes] = []
    for block in sorted_blocks:
        try:
            decode_generators = _decode_generators_for_block(stream, block)
            chunk = generators_to_bytes(decode_generators, block.n_strands, block.original_length)
        except (ValueError, TypeError, OverflowError, FormatError):
            return False, f"BLAKE3 checksum failed: cannot decode block {block.block_index}"
        chunks.append(chunk)

    reassembled = b"".join(chunks)
    actual = blake3.blake3(reassembled).digest()
    if actual != stream.checksum:
        return False, "BLAKE3 checksum mismatch on reassembled data"
    return True, None


def _check_topology(
    stream: EncodedStream,
    structurally_failed: set[int],
) -> tuple[bool | None, tuple[int, ...], tuple[str, ...]]:
    """Channel 5: verify stored topology metadata against recomputation.

    Returns
    -------
    tuple
        ``(status, failed_blocks, details)`` where ``status`` is:
        - ``True`` when topology metadata is present and valid,
        - ``False`` on mismatch,
        - ``None`` when channel is intentionally skipped.
    """
    mode = stream.metadata.get("preprocessing_mode")
    if stream.version < 2 or mode != "topology":
        return None, (), ()

    if structurally_failed:
        return (
            False,
            tuple(sorted(structurally_failed)),
            (
                "Topology check skipped: structurally invalid blocks "
                f"{sorted(structurally_failed)}",
            ),
        )

    sorted_blocks = sorted(stream.blocks, key=lambda b: b.block_index)
    chunks_for_profiles: list[tuple[int, bytes, int]] = []

    for block in sorted_blocks:
        try:
            decode_generators = _decode_generators_for_block(stream, block)
            chunk = generators_to_bytes(decode_generators, block.n_strands, block.original_length)
        except (ValueError, TypeError, OverflowError, FormatError):
            return (
                False,
                (block.block_index,),
                (f"Block {block.block_index}: topology decode failed",),
            )
        padded_len = compute_block_size(block.n_strands, len(block.generators))
        padded_chunk = chunk.ljust(padded_len, b"\x00")
        chunks_for_profiles.append((block.block_index, padded_chunk, block.original_length))

    gpb = len(sorted_blocks[0].generators) if sorted_blocks else 1
    expected_profiles = build_chunk_profiles(
        chunks_for_profiles,
        n_strands=stream.n_strands,
        generators_per_block=gpb,
    )
    expected_by_index = {p.block_index: p for p in expected_profiles}

    failed: set[int] = set()
    details: list[str] = []

    synthesis_version = stream.metadata.get("topology_synthesis_version", "1")

    for block in sorted_blocks:
        p = expected_by_index.get(block.block_index)
        if p is None:
            failed.add(block.block_index)
            details.append(f"Block {block.block_index}: missing recomputed topology profile")
            continue

        expected_values: tuple[tuple[str, int, int | None], ...]

        if synthesis_version == "2":
            expected_morton = morton_key_1d(block.block_index, p.layer_index)
            expected_commitment = topology_commitment_v2(
                morton_key=expected_morton,
                layer_index=p.layer_index,
                hash32=p.signature.hash32,
            )
            expected_values = (
                ("topology_layer_index", p.layer_index, block.topology_layer_index),
                ("topology_hash32", p.signature.hash32, block.topology_hash32),
                ("topology_morton_key", expected_morton, block.topology_morton_key),
                ("topology_commitment", expected_commitment, block.topology_commitment),
            )
        else:
            expected_morton = morton_key_2d(p.layer_index, block.block_index)
            expected_commitment = topology_commitment(
                morton_key=expected_morton,
                layer_index=p.layer_index,
                layer_n_chunks=p.layer_n_chunks,
                nnz_bits=p.nnz_bits,
                hash32=p.signature.hash32,
                density_fp=p.signature.density_fp,
                centroid_fp=p.signature.centroid_fp,
                variance_fp=p.signature.variance_fp,
            )
            expected_values = (
                ("topology_layer_index", p.layer_index, block.topology_layer_index),
                ("topology_layer_n_chunks", p.layer_n_chunks, block.topology_layer_n_chunks),
                ("topology_nnz_bits", p.nnz_bits, block.topology_nnz_bits),
                ("topology_hash32", p.signature.hash32, block.topology_hash32),
                ("topology_density_fp", p.signature.density_fp, block.topology_density_fp),
                ("topology_centroid_fp", p.signature.centroid_fp, block.topology_centroid_fp),
                ("topology_variance_fp", p.signature.variance_fp, block.topology_variance_fp),
                ("topology_morton_key", expected_morton, block.topology_morton_key),
                ("topology_commitment", expected_commitment, block.topology_commitment),
            )

        for field_name, expected, actual in expected_values:
            if actual != expected:
                failed.add(block.block_index)
                details.append(
                    f"Block {block.block_index}: {field_name} mismatch "
                    f"(stored={actual}, expected={expected})"
                )

        if synthesis_version != "2" and (
            block.topology_dt_scale is None
            or abs(block.topology_dt_scale - p.dt_scale) > 1e-12
        ):
            failed.add(block.block_index)
            details.append(
                f"Block {block.block_index}: topology_dt_scale mismatch "
                f"(stored={block.topology_dt_scale}, expected={p.dt_scale})"
            )

    if failed:
        return False, tuple(sorted(failed)), tuple(details)
    return True, (), ()


def _decode_generators_for_block(stream: EncodedStream, block: EncodedBlock) -> list[int]:
    """Resolve the generator sequence used for byte reconstruction."""
    if block.decode_generators is not None:
        return block.decode_generators

    mode = stream.metadata.get("preprocessing_mode")
    if mode == "reconstructive":
        payload = parse_reconstructive_payload_metadata(stream.metadata)
        seed_vector = payload.get("km_seed_vector", "")
        return reconstructive_inverse_generators(
            block.generators,
            n_strands=block.n_strands,
            seed_vector=seed_vector,
            block_index=block.block_index,
        )

    if mode != "topology":
        return block.generators

    synthesis_version = stream.metadata.get("topology_synthesis_version", "1")
    if synthesis_version == "2":
        if (
            block.topology_layer_index is None
            or block.topology_hash32 is None
            or block.topology_morton_key is None
        ):
            return block.generators

        return recover_legacy_generators_v2(
            topology_generators=block.generators,
            n_strands=block.n_strands,
            layer_index=block.topology_layer_index,
            signature_hash32=block.topology_hash32,
            morton_key=block.topology_morton_key,
        )

    if (
        block.topology_layer_index is None
        or block.topology_layer_n_chunks is None
        or block.topology_nnz_bits is None
        or block.topology_dt_scale is None
        or block.topology_hash32 is None
        or block.topology_density_fp is None
        or block.topology_centroid_fp is None
        or block.topology_variance_fp is None
    ):
        return block.generators

    return recover_legacy_generators(
        topology_generators=block.generators,
        n_strands=block.n_strands,
        layer_index=block.topology_layer_index,
        layer_n_chunks=block.topology_layer_n_chunks,
        nnz_bits=block.topology_nnz_bits,
        dt_scale=block.topology_dt_scale,
        signature_hash32=block.topology_hash32,
        signature_density_fp=block.topology_density_fp,
        signature_centroid_fp=block.topology_centroid_fp,
        signature_variance_fp=block.topology_variance_fp,
    )


# ── Public API ────────────────────────────────────────────────────────────


def verify(
    stream: EncodedStream,
    key: BraidKey,
    *,
    fermion_check: bool = False,
    topology_check: bool = False,
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
    topology_check:
        If ``True``, run topology metadata verification (channel 5) for
        topology-mode streams.

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
    topology_passed: bool | None = True if topology_check else None
    checksum_passed = True

    sorted_blocks = sorted(stream.blocks, key=lambda b: b.block_index)

    # Reconstructive contract validation is treated as a structural check.
    if stream.metadata.get("preprocessing_mode") == "reconstructive":
        try:
            validate_reconstructive_metadata(stream.metadata)
            payload = validate_reconstructive_payload_metadata(stream.metadata)
            validate_reconstructive_commitment_metadata(stream.metadata, stream.blocks)
            validate_reconstructive_solver_payload(payload)
        except FormatError as exc:
            structural_passed = False
            details.append(f"Reconstructive metadata invalid: {exc}")

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

    # Channel 5: topology (global recomputation, optional)
    if topology_check:
        topo_ok, topo_failed_blocks, topo_details = _check_topology(stream, structurally_failed)
        topology_passed = topo_ok
        if topo_ok is False:
            failed_blocks.update(topo_failed_blocks)
            details.extend(topo_details)

    # Channel 6: checksum (global, not per-block)
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
        and (topology_passed is not False)
        and checksum_passed
    )

    return VerificationResult(
        valid=valid,
        structural_passed=structural_passed,
        writhe_passed=writhe_passed,
        invariant_passed=invariant_passed,
        fermion_passed=fermion_passed,
        topology_passed=topology_passed,
        checksum_passed=checksum_passed,
        failed_blocks=tuple(sorted(failed_blocks)),
        details=tuple(details),
    )
