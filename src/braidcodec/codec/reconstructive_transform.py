"""Deterministic generator transforms for reconstructive mode.

The transform is invertible and keyed by the reconstructive seed vector.
Encode applies the forward map to block generators; decode applies the inverse
map from compact payload metadata.
"""

from __future__ import annotations

from braidcodec._exceptions import FormatError


def parse_seed_vector(seed_vector: str) -> tuple[float, ...]:
    """Parse comma-separated seed vector metadata into floats."""
    raw = seed_vector.strip()
    if not raw:
        raise FormatError("km_seed_vector must be non-empty")

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise FormatError("km_seed_vector must contain numeric values")

    values: list[float] = []
    for item in parts:
        try:
            values.append(float(item))
        except ValueError as exc:
            raise FormatError("km_seed_vector contains non-numeric entries") from exc

    if not values:
        raise FormatError("km_seed_vector must contain at least one value")
    return tuple(values)


def _allowed_generators(n_strands: int) -> tuple[int, ...]:
    if n_strands < 2:
        raise FormatError("n_strands must be >= 2", n_strands=n_strands)
    # Stable ordering shared by encode/decode.
    return tuple(list(range(1, n_strands)) + list(range(-1, -(n_strands), -1)))


def _seed_schedule(seed: tuple[float, ...], block_index: int, position: int) -> tuple[int, bool]:
    """Return per-position offset and involution flag from seed values."""
    seed_i = seed[(block_index + position) % len(seed)]
    quantized = int(abs(seed_i) * 1_000_003.0)
    offset = quantized
    flip = bool((quantized >> 5) & 1)
    return offset, flip


def _map_forward_index(idx: int, size: int, offset: int, flip: bool) -> int:
    mapped = (idx + (offset % size)) % size
    if flip:
        mapped = (size - 1) - mapped
    return mapped


def _map_inverse_index(idx: int, size: int, offset: int, flip: bool) -> int:
    mapped = idx
    if flip:
        mapped = (size - 1) - mapped
    mapped = (mapped - (offset % size)) % size
    return mapped


def reconstructive_forward_generators(
    generators: list[int],
    *,
    n_strands: int,
    seed_vector: str,
    block_index: int,
) -> list[int]:
    """Apply deterministic forward transform for reconstructive mode."""
    allowed = _allowed_generators(n_strands)
    index_by_gen = {g: i for i, g in enumerate(allowed)}
    seed = parse_seed_vector(seed_vector)

    out: list[int] = []
    size = len(allowed)
    for pos, gen in enumerate(generators):
        if gen not in index_by_gen:
            raise FormatError(
                "Generator out of range for reconstructive transform",
                generator=gen,
                n_strands=n_strands,
            )
        offset, flip = _seed_schedule(seed, block_index, pos)
        idx = index_by_gen[gen]
        mapped_idx = _map_forward_index(idx, size, offset, flip)
        out.append(allowed[mapped_idx])
    return out


def reconstructive_inverse_generators(
    generators: list[int],
    *,
    n_strands: int,
    seed_vector: str,
    block_index: int,
) -> list[int]:
    """Apply deterministic inverse transform for reconstructive mode."""
    allowed = _allowed_generators(n_strands)
    index_by_gen = {g: i for i, g in enumerate(allowed)}
    seed = parse_seed_vector(seed_vector)

    out: list[int] = []
    size = len(allowed)
    for pos, gen in enumerate(generators):
        if gen not in index_by_gen:
            raise FormatError(
                "Generator out of range for reconstructive inverse",
                generator=gen,
                n_strands=n_strands,
            )
        offset, flip = _seed_schedule(seed, block_index, pos)
        idx = index_by_gen[gen]
        mapped_idx = _map_inverse_index(idx, size, offset, flip)
        out.append(allowed[mapped_idx])
    return out
