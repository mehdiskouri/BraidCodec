"""Tests for braidcodec.codec.chunker — bijective numeration engine.

Covers:
  - Known block-size vectors
  - Encoding / decoding deterministic traces
  - chunk_stream edge cases (exact multiple, remainder, empty, single byte, BinaryIO)
  - Validation error paths
  - Hypothesis round-trip properties (4 strategies)
"""

from __future__ import annotations

import io

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from braidcodec.codec.chunker import (
    bytes_to_generators,
    chunk_stream,
    compute_block_size,
    compute_generators_needed,
    generators_to_bytes,
)

# ═══════════════════════════════════════════════════════════════════════════
# Known vectors — block size
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeBlockSize:
    """Verify block-size arithmetic against hand-computed reference values."""

    @pytest.mark.parametrize(
        ("n_strands", "k", "expected"),
        [
            # n=4, base=6, k=32 → floor(32*log₂(6)/8) = 10
            (4, 32, 10),
            # n=3, base=4, k=16 → floor(16 * 2 / 8) = floor(4) = 4
            (3, 16, 4),
            # n=2, base=2, k=8 → floor(8 * 1 / 8) = 1
            (2, 8, 1),
            # n=5, base=8, k=32 → floor(32 * 3 / 8) = 12
            (5, 32, 12),
            # n=6, base=10, k=32 → floor(32*log₂(10)/8) = 13
            (6, 32, 13),
            # n=5, base=8, k=16 → floor(16 * 3 / 8) = 6
            (5, 16, 6),
            # n=6, base=10, k=16 → floor(16 * 3.3219 / 8) = floor(6.64) = 6
            (6, 16, 6),
            # Minimal: n=2, k=1 → floor(1 * 1 / 8) = 0
            (2, 1, 0),
        ],
    )
    def test_known_vectors(self, n_strands: int, k: int, expected: int) -> None:
        assert compute_block_size(n_strands, k) == expected


class TestComputeGeneratorsNeeded:
    """Verify the inverse computation — minimum k for a given byte capacity."""

    @pytest.mark.parametrize(
        ("n_strands", "block_bytes", "expected_k"),
        [
            # n=4, base=6 — need k s.t. floor(k * 2.585 / 8) >= 10 → k=31 gives 9, k=32 gives 10 ✓
            (4, 10, 31),
            # n=3, base=4 — need k s.t. floor(k * 2 / 8) >= 4 → k=16 gives exactly 4 ✓
            (3, 4, 16),
            # n=2, base=2 — 1 byte → k=8
            (2, 1, 8),
        ],
    )
    def test_known_inverses(self, n_strands: int, block_bytes: int, expected_k: int) -> None:
        k = compute_generators_needed(n_strands, block_bytes)
        assert k == expected_k
        # The returned k must actually satisfy the capacity requirement.
        assert compute_block_size(n_strands, k) >= block_bytes

    @pytest.mark.parametrize(
        ("n_strands", "block_bytes"),
        [(3, 1), (4, 5), (5, 12), (6, 13)],
    )
    def test_inverse_always_sufficient(self, n_strands: int, block_bytes: int) -> None:
        k = compute_generators_needed(n_strands, block_bytes)
        assert compute_block_size(n_strands, k) >= block_bytes
        # k-1 must be insufficient (otherwise k is not minimal).
        if k > 1:
            assert compute_block_size(n_strands, k - 1) < block_bytes


# ═══════════════════════════════════════════════════════════════════════════
# Encoding / Decoding — deterministic traces
# ═══════════════════════════════════════════════════════════════════════════


class TestBytesToGenerators:
    """Test bytes → generator mapping."""

    def test_all_zeros_map_to_all_plus_one(self) -> None:
        """All-zero chunk → every digit is 0 → every generator is +1."""
        n, k = 4, 32
        chunk = b"\x00" * compute_block_size(n, k)
        gens = bytes_to_generators(chunk, n, k)
        assert len(gens) == k
        assert all(g == 1 for g in gens)

    def test_max_value_chunk(self) -> None:
        """Maximum-valued chunk → highest digit in every position."""
        n, k = 3, 16
        block_size = compute_block_size(n, k)
        base = 2 * (n - 1)
        max_val = base**k - 1
        chunk = max_val.to_bytes(block_size, byteorder="big")
        gens = bytes_to_generators(chunk, n, k)
        assert len(gens) == k
        # Every digit should be base-1, which maps to the last negative generator.
        last_neg = -(n - 1)
        assert all(g == last_neg for g in gens)

    def test_hello_known_trace(self) -> None:
        """Trace 'Hello' through the chunker and verify round-trip."""
        n, k = 4, 32
        block_size = compute_block_size(n, k)
        # "Hello" is 5 bytes, block_size is 10 → pad to 10
        data = b"Hello"
        padded = data.ljust(block_size, b"\x00")
        gens = bytes_to_generators(padded, n, k)
        assert len(gens) == k
        recovered = generators_to_bytes(gens, n, len(data))
        assert recovered == data

    def test_single_byte_min(self) -> None:
        """Single zero byte, padded to block size."""
        n, k = 4, 32
        block_size = compute_block_size(n, k)
        padded = b"\x00" * block_size
        gens = bytes_to_generators(padded, n, k)
        assert len(gens) == k
        # All digits zero → all +1 generators.
        assert all(g == 1 for g in gens)

    def test_single_byte_max(self) -> None:
        """Single 0xff byte, padded to block size — mirrors chunk_stream."""
        n, k = 4, 32
        block_size = compute_block_size(n, k)
        padded = b"\xff" + b"\x00" * (block_size - 1)
        gens = bytes_to_generators(padded, n, k)
        assert len(gens) == k
        recovered = generators_to_bytes(gens, n, 1)
        assert recovered == b"\xff"


class TestGeneratorsToBytes:
    """Test generator sequence → bytes recovery."""

    def test_all_plus_one_gives_zeros(self) -> None:
        """All +1 generators → digit 0 everywhere → value = 0 → all zero bytes."""
        n, k = 4, 32
        block_size = compute_block_size(n, k)
        gens = [1] * k
        result = generators_to_bytes(gens, n, block_size)
        assert result == b"\x00" * block_size

    def test_original_length_truncation(self) -> None:
        """original_length < padded size → trailing padding stripped."""
        n, k = 4, 32
        block_size = compute_block_size(n, k)
        chunk = b"AB"
        padded = chunk.ljust(block_size, b"\x00")
        gens = bytes_to_generators(padded, n, k)
        recovered = generators_to_bytes(gens, n, original_length=2)
        assert recovered == b"AB"


# ═══════════════════════════════════════════════════════════════════════════
# chunk_stream
# ═══════════════════════════════════════════════════════════════════════════


class TestChunkStream:
    """Test the streaming block iterator."""

    def test_exact_multiple(self) -> None:
        """Data length is an exact multiple of block_size."""
        block_size = 4
        data = b"ABCDEFGH"  # 8 bytes = 2 blocks
        chunks = list(chunk_stream(data, block_size))
        assert len(chunks) == 2
        assert chunks[0] == (0, b"ABCD", 4)
        assert chunks[1] == (1, b"EFGH", 4)

    def test_remainder_padded(self) -> None:
        """Last block is padded when data doesn't fill it."""
        block_size = 4
        data = b"ABCDE"  # 5 bytes → 2 blocks, second has 1 real byte
        chunks = list(chunk_stream(data, block_size))
        assert len(chunks) == 2
        assert chunks[0] == (0, b"ABCD", 4)
        assert chunks[1] == (1, b"E\x00\x00\x00", 1)

    def test_empty_input(self) -> None:
        """Empty input yields a single zero block with original_length=0."""
        chunks = list(chunk_stream(b"", 4))
        assert len(chunks) == 1
        idx, padded, orig_len = chunks[0]
        assert idx == 0
        assert padded == b"\x00\x00\x00\x00"
        assert orig_len == 0

    def test_single_byte(self) -> None:
        """Single byte → single block."""
        chunks = list(chunk_stream(b"\x42", 4))
        assert len(chunks) == 1
        assert chunks[0] == (0, b"\x42\x00\x00\x00", 1)

    def test_binary_io_input(self) -> None:
        """BinaryIO (BytesIO) accepted transparently."""
        data = b"Hello, World!"
        stream = io.BytesIO(data)
        chunks_from_bytes = list(chunk_stream(data, 5))
        stream.seek(0)
        chunks_from_io = list(chunk_stream(stream, 5))
        assert chunks_from_bytes == chunks_from_io

    def test_block_indices_sequential(self) -> None:
        """Block indices are sequential starting from 0."""
        data = bytes(range(256)) * 4  # 1024 bytes
        block_size = 10
        chunks = list(chunk_stream(data, block_size))
        indices = [idx for idx, _, _ in chunks]
        assert indices == list(range(len(chunks)))

    def test_reassembly(self) -> None:
        """Concatenating original_length bytes from each block recovers data."""
        data = b"The quick brown fox jumps over the lazy dog"
        block_size = 7
        parts: list[bytes] = []
        for _, padded, orig_len in chunk_stream(data, block_size):
            parts.append(padded[:orig_len])
        assert b"".join(parts) == data


# ═══════════════════════════════════════════════════════════════════════════
# Validation — error paths
# ═══════════════════════════════════════════════════════════════════════════


class TestValidation:
    """Verify that invalid inputs raise ValueError with clear messages."""

    def test_n_strands_too_low(self) -> None:
        with pytest.raises(ValueError, match="n_strands must be >= 2"):
            compute_block_size(1, 8)

    def test_n_strands_zero(self) -> None:
        with pytest.raises(ValueError, match="n_strands must be >= 2"):
            compute_block_size(0, 8)

    def test_generators_per_block_too_low(self) -> None:
        with pytest.raises(ValueError, match="generators_per_block must be >= 1"):
            compute_block_size(3, 0)

    def test_block_bytes_too_low(self) -> None:
        with pytest.raises(ValueError, match="block_bytes must be >= 1"):
            compute_generators_needed(3, 0)

    def test_oversized_chunk(self) -> None:
        n, k = 3, 16
        block_size = compute_block_size(n, k)
        with pytest.raises(ValueError, match="exceeds block capacity"):
            bytes_to_generators(b"\xff" * (block_size + 1), n, k)

    def test_generator_zero(self) -> None:
        """Generator index 0 is never valid."""
        with pytest.raises(ValueError, match="non-zero"):
            generators_to_bytes([1, 0, 2], n_strands=3, original_length=1)

    def test_generator_out_of_range_positive(self) -> None:
        """Generator |g| >= n_strands is invalid."""
        with pytest.raises(ValueError, match="out of range"):
            generators_to_bytes([4], n_strands=4, original_length=1)

    def test_generator_out_of_range_negative(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            generators_to_bytes([-4], n_strands=4, original_length=1)

    def test_empty_generators(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            generators_to_bytes([], n_strands=3, original_length=1)

    def test_original_length_exceeds_capacity(self) -> None:
        with pytest.raises(ValueError, match="exceeds block capacity"):
            generators_to_bytes([1] * 8, n_strands=2, original_length=2)

    def test_chunk_stream_block_size_zero(self) -> None:
        with pytest.raises(ValueError, match="block_size must be >= 1"):
            list(chunk_stream(b"abc", 0))

    def test_zero_capacity_block(self) -> None:
        """generators_to_bytes with k too small for even 1 byte."""
        # n=2, k=1 → block_size = floor(1 * 1 / 8) = 0
        with pytest.raises(ValueError, match="zero-byte capacity"):
            generators_to_bytes([1], n_strands=2, original_length=1)


# ═══════════════════════════════════════════════════════════════════════════
# Hypothesis round-trip properties
# ═══════════════════════════════════════════════════════════════════════════


# Strategy: n_strands ∈ [2, 6] and generators_per_block chosen so block
# capacity ≥ 1 byte.
_n_strands_st = st.integers(min_value=2, max_value=6)
_k_st = st.integers(min_value=1, max_value=64)


def _valid_n_k() -> st.SearchStrategy[tuple[int, int]]:
    """Generate (n_strands, k) pairs where the block holds ≥ 1 byte."""
    return st.tuples(_n_strands_st, _k_st).filter(lambda nk: compute_block_size(nk[0], nk[1]) >= 1)


class TestHypothesisRoundTrips:
    """Property-based round-trip tests (4 strategies per Phase 1 spec)."""

    @given(data=st.data())
    @settings(max_examples=1000, deadline=None)
    def test_bytes_to_generators_to_bytes(self, data: st.DataObject) -> None:
        """bytes → generators → bytes is identity for arbitrary chunks."""
        n, k = data.draw(_valid_n_k())
        block_size = compute_block_size(n, k)
        # Draw a chunk of exactly block_size random bytes.
        chunk = data.draw(st.binary(min_size=block_size, max_size=block_size))
        gens = bytes_to_generators(chunk, n, k)
        recovered = generators_to_bytes(gens, n, original_length=block_size)
        assert recovered == chunk

    @given(data=st.data())
    @settings(max_examples=1000, deadline=None)
    def test_generators_to_bytes_to_generators(self, data: st.DataObject) -> None:
        """generators → bytes → generators is identity for reachable sequences.

        Not every generator sequence is reachable: ``base^k`` exceeds
        ``256^block_size``.  We test only sequences produced by
        ``bytes_to_generators`` (which are precisely the reachable ones).
        """
        n, k = data.draw(_valid_n_k())
        block_size = compute_block_size(n, k)
        chunk = data.draw(st.binary(min_size=block_size, max_size=block_size))
        gens = bytes_to_generators(chunk, n, k)
        recovered_bytes = generators_to_bytes(gens, n, original_length=block_size)
        recovered_gens = bytes_to_generators(recovered_bytes, n, k)
        assert recovered_gens == gens

    @given(data=st.data())
    @settings(max_examples=1000, deadline=None)
    def test_all_generators_in_valid_range(self, data: st.DataObject) -> None:
        """Every output generator satisfies 1 ≤ |g| < n_strands."""
        n, k = data.draw(_valid_n_k())
        block_size = compute_block_size(n, k)
        chunk = data.draw(st.binary(min_size=block_size, max_size=block_size))
        gens = bytes_to_generators(chunk, n, k)
        for g in gens:
            assert g != 0, "Generator must be non-zero"
            assert 1 <= abs(g) < n, f"|{g}| not in [1, {n - 1}]"

    @given(data=st.data())
    @settings(max_examples=1000, deadline=None)
    def test_chunk_stream_reassembly(self, data: st.DataObject) -> None:
        """chunk_stream reassembly recovers original data for inputs up to 10 KB."""
        raw = data.draw(st.binary(min_size=0, max_size=10240))
        n = data.draw(_n_strands_st)
        # Pick a k so block_size ≥ 1
        k = data.draw(
            st.integers(min_value=1, max_value=64).filter(
                lambda kk: compute_block_size(n, kk) >= 1
            )
        )
        block_size = compute_block_size(n, k)

        parts: list[bytes] = []
        for _, padded, orig_len in chunk_stream(raw, block_size):
            parts.append(padded[:orig_len])
        assert b"".join(parts) == raw


# ═══════════════════════════════════════════════════════════════════════════
# Extra edge-case coverage
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Additional edge cases for full coverage."""

    def test_n_strands_2_base_2_bijection(self) -> None:
        """With 2 strands (base=2), generators are {+1, -1} — binary."""
        n, k = 2, 8
        # block_size = 1 byte, so encode a single byte.
        chunk = b"\xa5"  # 10100101 in binary
        gens = bytes_to_generators(chunk, n, k)
        assert len(gens) == 8
        # Expected: digit 1→-1, 0→+1 pattern for 0xA5 = 165 = b10100101
        recovered = generators_to_bytes(gens, n, 1)
        assert recovered == chunk

    def test_compute_generators_needed_consistency(self) -> None:
        """For every (n, block_bytes) the computed k satisfies the capacity."""
        for n in range(2, 7):
            for target in range(1, 20):
                k = compute_generators_needed(n, target)
                assert compute_block_size(n, k) >= target

    def test_large_block(self) -> None:
        """Encode and decode a full 10-byte block (n=4, k=32)."""
        n, k = 4, 32
        block_size = compute_block_size(n, k)
        assert block_size == 10
        data = bytes(range(10))
        gens = bytes_to_generators(data, n, k)
        recovered = generators_to_bytes(gens, n, block_size)
        assert recovered == data

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_digit_generator_roundtrip_exhaustive(self, n: int) -> None:
        """Every digit in [0, base) round-trips through digit↔generator."""
        from braidcodec.codec.chunker import _digit_to_generator, _generator_to_digit

        base = 2 * (n - 1)
        for d in range(base):
            g = _digit_to_generator(d, n)
            assert g != 0
            assert 1 <= abs(g) < n
            assert _generator_to_digit(g, n) == d

    def test_chunk_stream_large_data(self) -> None:
        """Verify chunk_stream handles large data correctly."""
        data = bytes(range(256)) * 40  # 10240 bytes
        block_size = 10
        chunks = list(chunk_stream(data, block_size))
        total = sum(orig for _, _, orig in chunks)
        assert total == len(data)
        # Reassembly
        parts = [padded[:orig] for _, padded, orig in chunks]
        assert b"".join(parts) == data
