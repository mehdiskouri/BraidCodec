"""Tests for the encode pipeline."""

from __future__ import annotations

import blake3
import numpy as np
import pytest

from braidcodec.algebra.braid_equations import (
    BraidEquation,
    contract_braid_tensor,
    jones_polynomial,
    writhe,
)
from braidcodec.codec.chunker import bytes_to_generators, compute_block_size
from braidcodec.codec.encoder import encode
from braidcodec.codec.schema import EncodedStream
from braidcodec.crypto.keys import BraidKey, keygen

# ── Helpers ───────────────────────────────────────────────────────────────

# Use generators_per_block=8 for fast tier-2 tests (2^8 = 256 state iterations).
_K_SMALL: int = 8
# Default generators_per_block=32 triggers tier 3 (trace only, fast).
_K_DEFAULT: int = 32


def _make_key(sector: str = "TSR", theta_offset: float = 1.0, n_strands: int = 4) -> BraidKey:
    return keygen(sector=sector, n_strands=n_strands, theta_offset=theta_offset)


# ── Basic encoding ────────────────────────────────────────────────────────


class TestEncodeBasic:
    def test_encode_returns_stream(self) -> None:
        key = _make_key()
        stream = encode(b"Hello", key, generators_per_block=_K_SMALL)
        assert isinstance(stream, EncodedStream)

    def test_stream_fields(self) -> None:
        key = _make_key()
        stream = encode(b"Hello", key, generators_per_block=_K_SMALL)
        assert stream.total_bytes == 5
        assert stream.n_strands == key.n_strands
        assert stream.sector == key.sector
        assert stream.version == 1

    def test_checksum_matches(self) -> None:
        data = b"Hello, World!"
        key = _make_key()
        stream = encode(data, key, generators_per_block=_K_SMALL)
        assert stream.checksum == blake3.blake3(data).digest()

    def test_generators_valid(self) -> None:
        key = _make_key()
        stream = encode(b"\x01\x02", key, generators_per_block=_K_SMALL)
        for block in stream.blocks:
            for g in block.generators:
                assert g != 0
                assert abs(g) < key.n_strands

    def test_writhe_correctness(self) -> None:
        """Writhe is computed on the *original* generators (pre-simplification)."""
        key = _make_key()
        data = b"\x01\x02"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        block = stream.blocks[0]

        # Independently compute writhe on original generators.
        bs = compute_block_size(key.n_strands, _K_SMALL)
        padded = data.ljust(bs, b"\x00")
        original_gens = bytes_to_generators(padded, key.n_strands, _K_SMALL)
        expected_writhe = writhe(BraidEquation(key.n_strands, original_gens, sector=key.sector))
        assert block.writhe == expected_writhe


# ── Jones correctness (tier 2) ────────────────────────────────────────────


class TestJonesCorrectness:
    def test_jones_matches_independent_computation(self) -> None:
        key = _make_key()
        data = b"\x01\x02"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        block = stream.blocks[0]
        assert block.invariant_tier == 2

        # Independently replicate the encoder pipeline.
        bs = compute_block_size(key.n_strands, _K_SMALL)
        padded = data.ljust(bs, b"\x00")
        gens = bytes_to_generators(padded, key.n_strands, _K_SMALL)
        braid = BraidEquation(
            key.n_strands,
            gens,
            sector=key.sector,
            _sector_params=key.sector_params,
        )
        expected = jones_polynomial(braid)

        assert block.jones is not None
        assert abs(block.jones - expected) < 1e-10


# ── Trace correctness (tier 3) ────────────────────────────────────────────


class TestTraceCorrectness:
    def test_trace_matches_independent_computation(self) -> None:
        key = _make_key()
        data = b"Hello"
        stream = encode(data, key, generators_per_block=_K_DEFAULT)
        block = stream.blocks[0]
        assert block.invariant_tier == 3

        # Independently replicate.
        bs = compute_block_size(key.n_strands, _K_DEFAULT)
        padded = data.ljust(bs, b"\x00")
        gens = bytes_to_generators(padded, key.n_strands, _K_DEFAULT)
        braid = BraidEquation(
            key.n_strands,
            gens,
            sector=key.sector,
            _sector_params=key.sector_params,
        )
        matrix = contract_braid_tensor(braid)
        expected = complex(np.trace(matrix))

        assert block.trace_invariant is not None
        assert abs(block.trace_invariant - expected) < 1e-10


# ── Tier selection ────────────────────────────────────────────────────────


class TestTierSelection:
    def test_tier_2_for_small_k(self) -> None:
        key = _make_key()
        stream = encode(b"\x01\x02", key, generators_per_block=_K_SMALL)
        for block in stream.blocks:
            assert block.invariant_tier == 2
            assert block.jones is not None
            assert block.trace_invariant is None

    def test_tier_3_for_large_k(self) -> None:
        key = _make_key()
        stream = encode(b"Hello", key, generators_per_block=_K_DEFAULT)
        for block in stream.blocks:
            assert block.invariant_tier == 3
            assert block.jones is None
            assert block.trace_invariant is not None


# ── Sectors ───────────────────────────────────────────────────────────────


class TestEncodeSectors:
    @pytest.mark.parametrize("sector", ["Identity", "TSR", "Ising", "Fibonacci", "SU2k2"])
    def test_sector_produces_valid_stream(self, sector: str) -> None:
        key = _make_key(sector=sector)
        stream = encode(b"\x01\x02", key, generators_per_block=_K_SMALL)
        assert stream.sector == sector
        assert len(stream.blocks) >= 1

    def test_different_keys_different_invariants(self) -> None:
        key1 = _make_key(theta_offset=0.5)
        key2 = _make_key(theta_offset=3.0)
        data = b"\x01\x02"
        s1 = encode(data, key1, generators_per_block=_K_SMALL)
        s2 = encode(data, key2, generators_per_block=_K_SMALL)
        # Same data, different keys → different Jones polynomials.
        assert s1.blocks[0].jones != s2.blocks[0].jones


# ── Multi-block ───────────────────────────────────────────────────────────


class TestEncodeMultiBlock:
    def test_sequential_block_indices(self) -> None:
        key = _make_key()
        stream = encode(b"A" * 100, key, generators_per_block=_K_DEFAULT)
        indices = [b.block_index for b in stream.blocks]
        assert indices == list(range(len(indices)))
        assert len(indices) > 1

    def test_empty_data(self) -> None:
        key = _make_key()
        stream = encode(b"", key, generators_per_block=_K_SMALL)
        assert stream.total_bytes == 0
        assert len(stream.blocks) == 1
        assert stream.blocks[0].original_length == 0

    def test_block_count(self) -> None:
        key = _make_key()
        bs = compute_block_size(key.n_strands, _K_DEFAULT)
        data = b"X" * (bs * 3)  # exactly 3 full blocks
        stream = encode(data, key, generators_per_block=_K_DEFAULT)
        assert len(stream.blocks) == 3


# ── Parallelism ───────────────────────────────────────────────────────────


class TestEncodeParallelism:
    def test_serial_parallel_blocks_match(self) -> None:
        data = b"A" * 100
        key = _make_key()
        serial = encode(data, key, generators_per_block=_K_DEFAULT, max_workers=1)
        parallel = encode(data, key, generators_per_block=_K_DEFAULT, max_workers=2)
        assert len(serial.blocks) == len(parallel.blocks)
        for sb, pb in zip(serial.blocks, parallel.blocks, strict=True):
            assert sb.generators == pb.generators
            assert sb.writhe == pb.writhe
            assert sb.invariant_tier == pb.invariant_tier
            assert sb.block_index == pb.block_index

    def test_max_workers_1(self) -> None:
        key = _make_key()
        stream = encode(b"Hello", key, generators_per_block=_K_SMALL, max_workers=1)
        assert isinstance(stream, EncodedStream)
