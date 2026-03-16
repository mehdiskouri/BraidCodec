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
    FormatError,
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
from braidcodec.codec.preprocessing import (
    recover_legacy_generators,
    recover_legacy_generators_v2,
)
from braidcodec.codec.reconstructive_compact import synthesize_reconstructive_bytes
from braidcodec.codec.reconstructive_transform import reconstructive_inverse_generators
from braidcodec.codec.schema import (
    get_reconstructive_transport_code,
    validate_reconstructive_compact_transport_metadata,
)

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


def _decode_generators_for_block(stream: EncodedStream, block: EncodedBlock) -> list[int]:
    """Resolve the generator sequence used for byte reconstruction."""
    if block.decode_generators is not None:
        return block.decode_generators

    mode = stream.metadata.get("preprocessing_mode")
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
            nnz_bits=block.topology_nnz_bits or 0,
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


def _decode_reconstructive_generators_for_block(
    block: EncodedBlock,
    payload: dict[str, str],
) -> list[int]:
    """Resolve generator sequence for reconstructive-mode decoding.

    Reconstructive mode uses payload-seeded inverse mapping to recover the
    byte reconstruction generator sequence.
    """
    if payload.get("model_id") != "frequency-manifold":
        raise FormatError("Unsupported reconstructive decode model")
    seed_vector = payload.get("km_seed_vector", "")
    return reconstructive_inverse_generators(
        block.generators,
        n_strands=block.n_strands,
        seed_vector=seed_vector,
        block_index=block.block_index,
    )


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

    mode = stream.metadata.get("preprocessing_mode")
    transport_code = get_reconstructive_transport_code(stream.metadata)
    reconstructive_mode = mode == "reconstructive" or bool(transport_code)
    reconstructive_payload: dict[str, str] | None = None
    if reconstructive_mode:
        reconstructive_payload = validate_reconstructive_compact_transport_metadata(
            stream.metadata,
            stream.blocks,
        )

        if reconstructive_payload is not None and transport_code:
            result = synthesize_reconstructive_bytes(reconstructive_payload)
            actual_checksum = blake3.blake3(result).digest()
            if actual_checksum != stream.checksum:
                raise ChecksumError(
                    "Reassembled data BLAKE3 checksum mismatch",
                    expected=stream.checksum.hex(),
                    actual=actual_checksum.hex(),
                )
            return result

        if not sorted_blocks:
            result = synthesize_reconstructive_bytes(reconstructive_payload)
            actual_checksum = blake3.blake3(result).digest()
            if actual_checksum != stream.checksum:
                raise ChecksumError(
                    "Reassembled data BLAKE3 checksum mismatch",
                    expected=stream.checksum.hex(),
                    actual=actual_checksum.hex(),
                )
            return result

    # ── Step 3: per-block decode ──────────────────────────────────────────
    chunks: list[bytes] = []
    for block in sorted_blocks:
        # Writhe pre-check — always, O(k)
        _check_writhe(block, sector_params)

        # Invariant check — optional but on by default
        if verify:
            _check_invariant(block, sector_params)

        # Inverse mapping: generators → bytes
        if mode == "reconstructive" and reconstructive_payload is not None:
            decode_generators = _decode_reconstructive_generators_for_block(
                block,
                reconstructive_payload,
            )
        else:
            decode_generators = _decode_generators_for_block(stream, block)
        try:
            chunk = generators_to_bytes(decode_generators, block.n_strands, block.original_length)
        except (OverflowError, ValueError) as exc:
            raise FormatError(
                f"Block {block.block_index}: invalid generator sequence for byte recovery",
                block_index=block.block_index,
            ) from exc
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
