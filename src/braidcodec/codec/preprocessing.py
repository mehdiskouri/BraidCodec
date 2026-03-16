"""Topology preprocessing helpers for encode scheduling.

Phase 2 inserts a deterministic chunk -> topology profiling step before braid
construction. The current implementation is intentionally decode-neutral: it does
not mutate payload bytes, but computes per-chunk signatures and sparse layer
profiles used by scheduling and future topology-aware encoding phases.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import blake3


def _topology_shift_terms(
    *,
    n_strands: int,
    layer_index: int,
    layer_n_chunks: int,
    nnz_bits: int,
    dt_scale: float,
    signature_hash32: int,
    signature_density_fp: int,
    signature_centroid_fp: int,
    signature_variance_fp: int,
) -> tuple[int, int, int, int, int, int]:
    span = max(n_strands - 1, 1)
    density_term = signature_density_fp % span
    centroid_term = signature_centroid_fp % span
    variance_term = signature_variance_fp % span
    dt_term = round(dt_scale * 1000.0) % span
    global_shift = (layer_index + layer_n_chunks + density_term + dt_term) % span
    sign_seed = (signature_hash32 ^ signature_density_fp ^ signature_variance_fp) & 1
    return span, centroid_term, variance_term, global_shift, sign_seed, nnz_bits


def synthesize_topology_generators(
    *,
    generators: list[int],
    n_strands: int,
    layer_index: int,
    layer_n_chunks: int,
    nnz_bits: int,
    dt_scale: float,
    signature_hash32: int,
    signature_density_fp: int,
    signature_centroid_fp: int,
    signature_variance_fp: int,
) -> list[int]:
    """Derive topology-mode generators from legacy mixed-radix generators.

    The transform is deterministic and bijective. It can be reversed with
    ``recover_legacy_generators``.
    """
    span, centroid_term, variance_term, global_shift, sign_seed, nnz_local = _topology_shift_terms(
        n_strands=n_strands,
        layer_index=layer_index,
        layer_n_chunks=layer_n_chunks,
        nnz_bits=nnz_bits,
        dt_scale=dt_scale,
        signature_hash32=signature_hash32,
        signature_density_fp=signature_density_fp,
        signature_centroid_fp=signature_centroid_fp,
        signature_variance_fp=signature_variance_fp,
    )

    derived: list[int] = []
    for i, gen in enumerate(generators):
        base = abs(gen) - 1
        local_shift = (global_shift + i + variance_term + (nnz_local & 0x7)) % span
        new_abs = ((base + local_shift + centroid_term) % span) + 1

        sign = 1 if gen > 0 else -1
        if ((signature_hash32 >> (i % 16)) & 1) == 1:
            sign *= -1
        if ((nnz_local + i + sign_seed) & 1) == 1:
            sign *= -1

        derived.append(sign * new_abs)

    return derived


def recover_legacy_generators(
    *,
    topology_generators: list[int],
    n_strands: int,
    layer_index: int,
    layer_n_chunks: int,
    nnz_bits: int,
    dt_scale: float,
    signature_hash32: int,
    signature_density_fp: int,
    signature_centroid_fp: int,
    signature_variance_fp: int,
) -> list[int]:
    """Invert ``synthesize_topology_generators`` back to legacy generators."""
    span, centroid_term, variance_term, global_shift, sign_seed, nnz_local = _topology_shift_terms(
        n_strands=n_strands,
        layer_index=layer_index,
        layer_n_chunks=layer_n_chunks,
        nnz_bits=nnz_bits,
        dt_scale=dt_scale,
        signature_hash32=signature_hash32,
        signature_density_fp=signature_density_fp,
        signature_centroid_fp=signature_centroid_fp,
        signature_variance_fp=signature_variance_fp,
    )

    recovered: list[int] = []
    for i, gen in enumerate(topology_generators):
        local_shift = (global_shift + i + variance_term + (nnz_local & 0x7)) % span
        topo_abs = abs(gen) - 1
        base = (topo_abs - local_shift - centroid_term) % span
        legacy_abs = base + 1

        sign = 1 if gen > 0 else -1
        if ((signature_hash32 >> (i % 16)) & 1) == 1:
            sign *= -1
        if ((nnz_local + i + sign_seed) & 1) == 1:
            sign *= -1

        recovered.append(sign * legacy_abs)

    return recovered


def morton_key_2d(layer_index: int, block_index: int) -> int:
    """Compute a 2D Morton key by interleaving layer/block bits.

    The result is deterministic and stable across runs for the same
    ``(layer_index, block_index)`` pair.
    """
    x = max(layer_index, 0)
    y = max(block_index, 0)
    out = 0
    bit = 0
    while x > 0 or y > 0:
        out |= (x & 1) << (2 * bit)
        out |= (y & 1) << (2 * bit + 1)
        x >>= 1
        y >>= 1
        bit += 1
    return out


def morton_key_1d(block_index: int, lane: int = 0) -> int:
    """Compute a 1D Morton-like key for block ordering within a lane.

    This dilates block-index bits and injects a small lane discriminator in
    low bits for deterministic topology binding with lower metadata overhead.
    """
    x = max(block_index, 0)
    out = 0
    bit = 0
    while x > 0:
        out |= (x & 1) << (2 * bit)
        x >>= 1
        bit += 1
    return (out << 3) | (max(lane, 0) & 0x7)


def topology_commitment(
    *,
    morton_key: int,
    layer_index: int,
    layer_n_chunks: int,
    nnz_bits: int,
    hash32: int,
    density_fp: int,
    centroid_fp: int,
    variance_fp: int,
) -> int:
    """Create a compact deterministic commitment over topology descriptors."""
    payload = (
        f"{morton_key}|{layer_index}|{layer_n_chunks}|{nnz_bits}|"
        f"{hash32}|{density_fp}|{centroid_fp}|{variance_fp}"
    ).encode("ascii")
    digest = blake3.blake3(payload).digest(length=8)
    return int.from_bytes(digest, byteorder="big", signed=False)


def topology_commitment_v2(
    *,
    morton_key: int,
    layer_index: int,
    hash32: int,
) -> int:
    """Compact deterministic commitment for synthesis v2 metadata."""
    payload = f"{morton_key}|{layer_index}|{hash32}".encode("ascii")
    digest = blake3.blake3(payload).digest(length=8)
    return int.from_bytes(digest, byteorder="big", signed=False)


@dataclass(frozen=True, slots=True)
class BinaryFixedPointSignature:
    """Fixed-point signature derived deterministically from a chunk."""

    hash32: int
    density_fp: int
    centroid_fp: int
    variance_fp: int


@dataclass(frozen=True, slots=True)
class ChunkTopologyProfile:
    """Derived topology profile for one chunk.

    This profile is used for cost estimation and layer-aware scheduling only.
    """

    block_index: int
    original_length: int
    layer_index: int
    layer_n_chunks: int
    nnz_bits: int
    dt_scale: float
    estimated_cost: float
    signature: BinaryFixedPointSignature


def compute_bfps(chunk: bytes) -> BinaryFixedPointSignature:
    """Compute a deterministic fixed-point signature for one chunk."""
    if not chunk:
        return BinaryFixedPointSignature(hash32=0, density_fp=0, centroid_fp=0, variance_fp=0)

    bit_counts = [byte.bit_count() for byte in chunk]
    total_bits = 8 * len(chunk)
    ones = sum(bit_counts)

    density = ones / total_bits

    weighted_sum = 0.0
    weighted_sq_sum = 0.0
    for idx, count in enumerate(bit_counts):
        weighted_sum += idx * count
        weighted_sq_sum += (idx * idx) * count

    if ones == 0:
        centroid = 0.0
        variance = 0.0
    else:
        centroid = weighted_sum / ones
        second_moment = weighted_sq_sum / ones
        variance = max(second_moment - (centroid * centroid), 0.0)

    max_index = max(len(chunk) - 1, 1)
    centroid_norm = min(max(centroid / max_index, 0.0), 1.0)
    variance_norm = min(max(variance / (max_index * max_index), 0.0), 1.0)

    digest = blake3.blake3(chunk).digest(length=4)
    hash32 = int.from_bytes(digest, byteorder="big", signed=False)

    # Fixed-point encoding (Q16.16 style for bounded values).
    return BinaryFixedPointSignature(
        hash32=hash32,
        density_fp=round(density * 65535.0),
        centroid_fp=round(centroid_norm * 65535.0),
        variance_fp=round(variance_norm * 65535.0),
    )


def synthesize_topology_generators_v2(
    *,
    generators: list[int],
    n_strands: int,
    layer_index: int,
    signature_hash32: int,
    morton_key: int,
    nnz_bits: int = 0,
) -> list[int]:
    """Derive compact topology generators for synthesis version 2."""
    span = max(n_strands - 1, 1)
    lane_term = (morton_key & 0x1F) % span
    nnz_term = max(nnz_bits, 0) % span
    sign_seed = ((signature_hash32 >> 5) ^ morton_key ^ max(nnz_bits, 0)) & 1

    derived: list[int] = []
    for i, gen in enumerate(generators):
        base = abs(gen) - 1
        shift = ((signature_hash32 >> (i % 24)) + layer_index + lane_term + nnz_term + i) % span
        new_abs = ((base + shift) % span) + 1

        sign = 1 if gen > 0 else -1
        if ((signature_hash32 >> (i % 16)) & 1) == 1:
            sign *= -1
        if ((max(nnz_bits, 0) + i + sign_seed) & 1) == 1:
            sign *= -1

        derived.append(sign * new_abs)

    return derived


def recover_legacy_generators_v2(
    *,
    topology_generators: list[int],
    n_strands: int,
    layer_index: int,
    signature_hash32: int,
    morton_key: int,
    nnz_bits: int = 0,
) -> list[int]:
    """Invert ``synthesize_topology_generators_v2`` back to legacy generators."""
    span = max(n_strands - 1, 1)
    lane_term = (morton_key & 0x1F) % span
    nnz_term = max(nnz_bits, 0) % span
    sign_seed = ((signature_hash32 >> 5) ^ morton_key ^ max(nnz_bits, 0)) & 1

    recovered: list[int] = []
    for i, gen in enumerate(topology_generators):
        topo_abs = abs(gen) - 1
        shift = ((signature_hash32 >> (i % 24)) + layer_index + lane_term + nnz_term + i) % span
        base = (topo_abs - shift) % span
        legacy_abs = base + 1

        sign = 1 if gen > 0 else -1
        if ((signature_hash32 >> (i % 16)) & 1) == 1:
            sign *= -1
        if ((max(nnz_bits, 0) + i + sign_seed) & 1) == 1:
            sign *= -1

        recovered.append(sign * legacy_abs)

    return recovered


def _layer_bucket_count(total_chunks: int) -> int:
    if total_chunks <= 1:
        return 1
    # Keep layer fan-out modest to preserve sparsity and scheduling locality.
    return max(1, min(32, math.ceil(math.sqrt(total_chunks))))


def _estimate_block_cost(
    *,
    n_strands: int,
    generators_per_block: int,
    signature: BinaryFixedPointSignature,
    layer_n_chunks: int,
) -> float:
    """Estimate relative compute cost for scheduling mode selection.

    The model is intentionally simple and monotonic so it can be recalibrated
    later against benchmark telemetry.
    """
    matrix_factor = float(2 ** max(n_strands - 2, 1))
    generator_factor = max(generators_per_block, 1) / 8.0
    density = signature.density_fp / 65535.0
    layer_factor = 1.0 + (layer_n_chunks / 16.0)
    return matrix_factor * generator_factor * (1.0 + density) * layer_factor


def build_chunk_profiles(
    chunks: list[tuple[int, bytes, int]],
    *,
    n_strands: int,
    generators_per_block: int,
) -> list[ChunkTopologyProfile]:
    """Build deterministic sparse-layer topology profiles for all chunks."""
    if not chunks:
        return []

    bucket_count = _layer_bucket_count(len(chunks))

    provisional: list[tuple[int, int, int, int, BinaryFixedPointSignature]] = []
    layer_counts: Counter[int] = Counter()

    for block_index, chunk, original_length in chunks:
        sig = compute_bfps(chunk)
        nnz_bits = sum(byte.bit_count() for byte in chunk)
        # Deterministic sparse assignment from signature hash and block index.
        layer_index = (sig.hash32 ^ (block_index * 2654435761)) % bucket_count
        provisional.append((block_index, original_length, layer_index, nnz_bits, sig))
        layer_counts[layer_index] += 1

    profiles: list[ChunkTopologyProfile] = []
    for block_index, original_length, layer_index, nnz_bits, sig in provisional:
        layer_n_chunks = layer_counts[layer_index]
        density = sig.density_fp / 65535.0
        dt_scale = 1.0 + density + (layer_index / max(bucket_count, 1))

        estimated_cost = _estimate_block_cost(
            n_strands=n_strands,
            generators_per_block=generators_per_block,
            signature=sig,
            layer_n_chunks=layer_n_chunks,
        )

        profiles.append(
            ChunkTopologyProfile(
                block_index=block_index,
                original_length=original_length,
                layer_index=layer_index,
                layer_n_chunks=layer_n_chunks,
                nnz_bits=nnz_bits,
                dt_scale=dt_scale,
                estimated_cost=estimated_cost,
                signature=sig,
            )
        )

    return profiles


def sort_chunk_indices_for_topology(profiles: list[ChunkTopologyProfile]) -> list[int]:
    """Return chunk indices in topology-aware processing order."""
    return [
        p.block_index
        for p in sorted(
            profiles,
            key=lambda p: (p.layer_index, -p.estimated_cost, p.block_index),
        )
    ]


def mean_estimated_cost(profiles: list[ChunkTopologyProfile]) -> float:
    """Mean estimated cost across all chunk profiles."""
    if not profiles:
        return 0.0
    return sum(p.estimated_cost for p in profiles) / len(profiles)
