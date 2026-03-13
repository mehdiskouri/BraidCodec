"""Bijective numeration between byte blocks and braid generator sequences.

This module implements the pure-arithmetic bijection at the heart of BraidCodec's
serialization engine.  With *n* strands the valid generators are
``{-(n-1), …, -1, 1, …, (n-1)}`` — *b = 2(n-1)* values.  A block of *k*
generators encodes values in ``[0, b^k - 1]``, holding
``floor(k · log₂(b) / 8)`` bytes.

No algebra imports — only ``math``, ``typing``, and ``collections.abc``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Iterator

    from braidcodec._types import GeneratorSeq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_n_strands(n_strands: int) -> None:
    if n_strands < 2:
        raise ValueError(f"n_strands must be >= 2, got {n_strands}")


def _validate_generators_per_block(generators_per_block: int) -> None:
    if generators_per_block < 1:
        raise ValueError(f"generators_per_block must be >= 1, got {generators_per_block}")


def _base(n_strands: int) -> int:
    """Number of distinct generator symbols for *n_strands* strands."""
    return 2 * (n_strands - 1)


# ---------------------------------------------------------------------------
# Digit ↔ Generator mapping
# ---------------------------------------------------------------------------


def _digit_to_generator(digit: int, n_strands: int) -> int:
    """Map a base-*b* digit to a signed generator index.

    d < (n-1)  → +(d + 1)
    d >= (n-1) → -(d - n + 2)
    """
    half = n_strands - 1
    if digit < half:
        return digit + 1
    return -(digit - half + 1)


def _generator_to_digit(generator: int, n_strands: int) -> int:
    """Map a signed generator index back to a base-*b* digit.

    g > 0 → g - 1
    g < 0 → (n-1) + |g| - 1
    """
    if generator == 0:
        raise ValueError("Generator index must be non-zero")
    if abs(generator) >= n_strands:
        raise ValueError(
            f"Generator |{generator}| out of range for {n_strands} strands "
            f"(must be in 1..{n_strands - 1})"
        )
    if generator > 0:
        return generator - 1
    return (n_strands - 1) + abs(generator) - 1


# ---------------------------------------------------------------------------
# Block-size arithmetic
# ---------------------------------------------------------------------------


def compute_block_size(n_strands: int, generators_per_block: int) -> int:
    """Return the number of payload bytes a single block can hold.

    ``floor(k · log₂(base) / 8)`` where *k* = *generators_per_block* and
    *base* = ``2(n_strands - 1)``.
    """
    _validate_n_strands(n_strands)
    _validate_generators_per_block(generators_per_block)
    b = _base(n_strands)
    return math.floor(generators_per_block * math.log2(b) / 8)


def compute_generators_needed(n_strands: int, block_bytes: int) -> int:
    """Return the minimum generators needed to encode *block_bytes* bytes.

    Inverse of :func:`compute_block_size`: the smallest *k* such that
    ``floor(k · log₂(base) / 8) >= block_bytes``.
    """
    _validate_n_strands(n_strands)
    if block_bytes < 1:
        raise ValueError(f"block_bytes must be >= 1, got {block_bytes}")
    b = _base(n_strands)
    bits_per_gen = math.log2(b)
    return math.ceil(block_bytes * 8 / bits_per_gen)


# ---------------------------------------------------------------------------
# Encode / Decode
# ---------------------------------------------------------------------------


def bytes_to_generators(
    chunk: bytes,
    n_strands: int,
    generators_per_block: int,
) -> GeneratorSeq:
    """Convert a byte chunk to a generator sequence via base-*b* decomposition.

    The chunk is interpreted as a big-endian unsigned integer, decomposed into
    *generators_per_block* digits in base ``2(n_strands - 1)``, then each digit
    is mapped to a signed generator index.

    Raises
    ------
    ValueError
        If *chunk* is too large for the given block parameters.
    """
    _validate_n_strands(n_strands)
    _validate_generators_per_block(generators_per_block)
    b = _base(n_strands)
    block_capacity = compute_block_size(n_strands, generators_per_block)
    if len(chunk) > block_capacity:
        raise ValueError(
            f"Chunk of {len(chunk)} bytes exceeds block capacity of "
            f"{block_capacity} bytes for n_strands={n_strands}, "
            f"generators_per_block={generators_per_block}"
        )

    value = int.from_bytes(chunk, byteorder="big")

    # Decompose into base-b digits, least-significant first, then reverse.
    digits: list[int] = []
    for _ in range(generators_per_block):
        digits.append(value % b)
        value //= b

    # After full decomposition the residual must be zero.
    if value != 0:
        raise ValueError("Chunk value overflows the generator block capacity")

    digits.reverse()  # most-significant digit first
    return [_digit_to_generator(d, n_strands) for d in digits]


def generators_to_bytes(
    generators: GeneratorSeq,
    n_strands: int,
    original_length: int,
) -> bytes:
    """Recover the original byte chunk from a generator sequence.

    Parameters
    ----------
    generators:
        Signed generator indices (length determines *k*).
    n_strands:
        Strand count of the braid group.
    original_length:
        Byte length of the *original* (pre-padding) data in this block.
        Used to strip trailing zero-padding.

    Raises
    ------
    ValueError
        If any generator is zero or out of range.
    """
    _validate_n_strands(n_strands)
    if not generators:
        raise ValueError("Generator sequence must not be empty")
    b = _base(n_strands)

    # Digits → big-endian integer reconstruction (Horner's method).
    value = 0
    for g in generators:
        digit = _generator_to_digit(g, n_strands)
        value = value * b + digit

    # Determine the padded byte length (full block capacity).
    k = len(generators)
    padded_length = compute_block_size(n_strands, k)
    if padded_length < 1:
        raise ValueError(
            f"Block parameters yield zero-byte capacity (n_strands={n_strands}, k={k})"
        )
    if original_length > padded_length:
        raise ValueError(
            f"original_length ({original_length}) exceeds block capacity ({padded_length})"
        )

    raw = value.to_bytes(padded_length, byteorder="big")
    return raw[:original_length]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def chunk_stream(
    data: bytes | BinaryIO,
    block_size: int,
) -> Iterator[tuple[int, bytes, int]]:
    """Yield ``(block_index, padded_chunk, original_length)`` tuples.

    *data* is either a ``bytes`` object or a readable binary stream.
    Each yielded chunk is zero-padded on the right to *block_size* bytes;
    *original_length* records the true payload size (which may be less than
    *block_size* for the last block, or for empty input).

    Parameters
    ----------
    data:
        Raw input data.
    block_size:
        Fixed chunk width in bytes (from :func:`compute_block_size`).

    Raises
    ------
    ValueError
        If *block_size* < 1.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    buf = data if isinstance(data, bytes) else data.read()

    if len(buf) == 0:
        # Yield a single all-zeros block with original_length=0 so that
        # encoding empty input is well-defined.
        yield (0, b"\x00" * block_size, 0)
        return

    idx = 0
    offset = 0
    while offset < len(buf):
        raw_chunk = buf[offset : offset + block_size]
        orig_len = len(raw_chunk)
        padded = raw_chunk.ljust(block_size, b"\x00")
        yield (idx, padded, orig_len)
        idx += 1
        offset += block_size
