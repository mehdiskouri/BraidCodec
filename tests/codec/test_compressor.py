"""Tests for the topological compressor."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from braidcodec._exceptions import CompressionError
from braidcodec.algebra.braid_equations import (
    BraidEquation,
    contract_braid_tensor,
    writhe,
)
from braidcodec.codec.compressor import (
    _compress_level0,
    _compress_level1,
    _compress_level2,
    _find_ybe_sites,
    compress,
    compress_braid,
    compression_ratio,
)
from braidcodec.codec.decoder import decode
from braidcodec.codec.encoder import encode
from braidcodec.codec.preprocessing import recover_legacy_generators_v2
from braidcodec.crypto.keys import keygen

# ── Helpers ───────────────────────────────────────────────────────────────

_K_SMALL: int = 8


def _braid(gens: list[int], n: int = 4, sector: str = "TSR") -> BraidEquation:
    return BraidEquation(n, gens, sector=sector)


def _matrix_equivalent(b1: BraidEquation, b2: BraidEquation) -> bool:
    m1 = contract_braid_tensor(b1)
    m2 = contract_braid_tensor(b2)
    return bool(float(abs(m1 - m2).max()) < 1e-10)


# ═════════════════════════════════════════════════════════════════════════
# Level 0 — Inverse cancellation
# ═════════════════════════════════════════════════════════════════════════


class TestLevel0:
    def test_adjacent_inverse_cancelled(self) -> None:
        b = _braid([1, -1, 2])
        result = _compress_level0(b)
        assert result.generators == [2]

    def test_nested_cancellation(self) -> None:
        """After first cancel, newly adjacent pair cancels too."""
        b = _braid([1, 2, -2, -1])
        result = _compress_level0(b)
        assert result.generators == []

    def test_no_op_when_nothing_to_cancel(self) -> None:
        b = _braid([1, 2, 3], n=5)
        result = _compress_level0(b)
        assert result.generators == [1, 2, 3]

    def test_matrix_preserved(self) -> None:
        b = _braid([1, -1, 2, 3, -3], n=5)
        result = _compress_level0(b)
        assert _matrix_equivalent(b, result)

    def test_empty_braid(self) -> None:
        b = _braid([1, -1])
        result = _compress_level0(b)
        assert result.generators == []


# ═════════════════════════════════════════════════════════════════════════
# Level 1 — Far-commutativity
# ═════════════════════════════════════════════════════════════════════════


class TestLevel1:
    def test_distant_commute_cancel(self) -> None:
        """[1, 3, -1, 2] on 5 strands: gen 3 commutes with gen 1 (|3-1|=2)."""
        b = _braid([1, 3, -1, 2], n=5)
        result = _compress_level1(b)
        assert 1 not in result.generators
        assert -1 not in result.generators
        assert len(result.generators) <= 2

    def test_no_commute_case(self) -> None:
        """[1, 2, -1] stays — gen 2 does NOT commute with 1 (|2-1|=1 < 2)."""
        b = _braid([1, 2, -1])
        result = _compress_level1(b)
        assert result.generators == [1, 2, -1]

    def test_multiple_distant_pairs(self) -> None:
        """Multiple far pairs should all cancel."""
        # [1, 3, -1, 3, -3] on 5 strands: (1, -1) across gen 3 which commutes,
        # then (3, -3) adjacent.
        b = _braid([1, 3, -1, 3, -3], n=5)
        result = _compress_level1(b)
        # 1,-1 cancel (gen 3 commutes); then 3,-3 cancel
        assert len(result.generators) == 0 or len(result.generators) < 5

    def test_jones_preserved(self) -> None:
        b = _braid([1, 3, -1, 2], n=5)
        result = _compress_level1(b)
        if result.generators:
            assert _matrix_equivalent(b, result)

    def test_single_generator(self) -> None:
        b = _braid([1])
        result = _compress_level1(b)
        assert result.generators == [1]


# ═════════════════════════════════════════════════════════════════════════
# Level 2 — Yang–Baxter BFS
# ═════════════════════════════════════════════════════════════════════════


class TestLevel2:
    def test_ybe_enables_cancel(self) -> None:
        """[1, 2, 1, -2] → YBE yields [2, 1, 2, -2] → cancel (2,-2) → [2, 1]."""
        b = _braid([1, 2, 1, -2])
        result = _compress_level2(b)
        assert len(result.generators) <= 2

    def test_all_negative_ybe(self) -> None:
        """All-negative YBE variant: [-1, -2, -1, 2]."""
        b = _braid([-1, -2, -1, 2])
        result = _compress_level2(b)
        assert len(result.generators) <= 2

    def test_bounded_search(self) -> None:
        """With max_rewrites=0, no exploration happens — only L0+L1."""
        b = _braid([1, 2, 1, -2])
        result_bounded = _compress_level2(b, max_rewrites=0)
        result_full = _compress_level2(b, max_rewrites=100)
        assert len(result_full.generators) <= len(result_bounded.generators)

    def test_matrix_preserved(self) -> None:
        b = _braid([1, 2, 1, -2])
        result = _compress_level2(b)
        if result.generators:
            assert _matrix_equivalent(b, result)

    def test_ybe_sites_positive(self) -> None:
        gens = [1, 2, 1]
        sites = _find_ybe_sites(gens)
        assert len(sites) == 1
        assert sites[0] == (0, [2, 1, 2])

    def test_ybe_sites_negative(self) -> None:
        gens = [-1, -2, -1]
        sites = _find_ybe_sites(gens)
        assert len(sites) == 1
        assert sites[0] == (0, [-2, -1, -2])

    def test_no_ybe_sites(self) -> None:
        gens = [1, 3, 2]
        sites = _find_ybe_sites(gens)
        assert len(sites) == 0


# ═════════════════════════════════════════════════════════════════════════
# compress_braid (public API)
# ═════════════════════════════════════════════════════════════════════════


class TestCompressBraid:
    def test_invalid_level(self) -> None:
        b = _braid([1, 2])
        with pytest.raises(CompressionError, match="level"):
            compress_braid(b, level=3)

    def test_empty_braid_passthrough(self) -> None:
        b = _braid([1, -1])
        simplified = _compress_level0(b)
        result = compress_braid(simplified, level=0)
        assert result.generators == []

    def test_level_progression(self) -> None:
        """Higher levels should produce equal or shorter results."""
        b = _braid([1, 2, 1, -2])
        r0 = compress_braid(b, level=0)
        r1 = compress_braid(b, level=1)
        r2 = compress_braid(b, level=2)
        assert len(r1.generators) <= len(r0.generators)
        assert len(r2.generators) <= len(r1.generators)


# ═════════════════════════════════════════════════════════════════════════
# compress (stream-level)
# ═════════════════════════════════════════════════════════════════════════


class TestCompressStream:
    def test_encode_compress_decode(self) -> None:
        """Full pipeline: encode → compress → decode."""
        key = keygen(sector="TSR", n_strands=4, theta_offset=1.0)
        data = b"Compress me!"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        compressed = compress(stream, key, level=1)
        result = decode(compressed, key)
        assert result == data

    def test_decode_generators_present_after_compression(self) -> None:
        """Compressed blocks with shortened generators carry decode_generators."""
        key = keygen(sector="TSR", n_strands=4, theta_offset=1.0)
        data = b"Hello, topology!"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        compressed = compress(stream, key, level=2)
        synthesis_version = stream.metadata.get("topology_synthesis_version", "1")
        for orig_block, comp_block in zip(stream.blocks, compressed.blocks, strict=True):
            if len(comp_block.generators) < len(orig_block.generators):
                assert comp_block.decode_generators is not None
                if (
                    synthesis_version == "2"
                    and stream.metadata.get("preprocessing_mode") == "topology"
                    and orig_block.topology_layer_index is not None
                    and orig_block.topology_hash32 is not None
                    and orig_block.topology_morton_key is not None
                ):
                    expected = recover_legacy_generators_v2(
                        topology_generators=orig_block.generators,
                        n_strands=orig_block.n_strands,
                        layer_index=orig_block.topology_layer_index,
                        signature_hash32=orig_block.topology_hash32,
                        morton_key=orig_block.topology_morton_key,
                        nnz_bits=orig_block.topology_nnz_bits or 0,
                    )
                    assert comp_block.decode_generators == expected
                else:
                    assert comp_block.decode_generators == orig_block.effective_decode_generators

    def test_ratio_lte_one(self) -> None:
        """Compression ratio ≤ 1.0 (compressed never larger than original)."""
        key = keygen(sector="TSR", n_strands=4, theta_offset=1.0)
        data = b"Test ratio"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        compressed = compress(stream, key, level=1)
        ratio = compression_ratio(stream, compressed)
        assert ratio <= 1.0

    def test_progressive_levels(self) -> None:
        """Level 2 should give equal or better ratio than Level 1."""
        key = keygen(sector="TSR", n_strands=4, theta_offset=1.0)
        data = b"Progressive level test data"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        c1 = compress(stream, key, level=1)
        c2 = compress(stream, key, level=2)
        r1 = compression_ratio(stream, c1)
        r2 = compression_ratio(stream, c2)
        assert r2 <= r1 + 1e-12  # allow float rounding

    def test_empty_data(self) -> None:
        key = keygen(sector="TSR", n_strands=4, theta_offset=1.0)
        data = b""
        stream = encode(data, key, generators_per_block=_K_SMALL)
        compressed = compress(stream, key, level=1)
        assert decode(compressed, key) == data


# ═════════════════════════════════════════════════════════════════════════
# All sectors
# ═════════════════════════════════════════════════════════════════════════


class TestCompressAllSectors:
    @pytest.mark.parametrize("sector", ["Identity", "TSR", "Ising", "Fibonacci", "SU2k2"])
    def test_compress_per_sector(self, sector: str) -> None:
        key = keygen(sector=sector, n_strands=4, theta_offset=1.0)
        data = b"Sector test"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        compressed = compress(stream, key, level=1)
        assert decode(compressed, key) == data


# ═════════════════════════════════════════════════════════════════════════
# Wire-format round-trip with compressed stream
# ═════════════════════════════════════════════════════════════════════════


class TestCompressWireFormat:
    def test_compressed_wire_roundtrip(self) -> None:
        """encode → compress → to_bytes → from_bytes → decode."""
        from braidcodec.codec.schema import EncodedStream

        key = keygen(sector="TSR", n_strands=4, theta_offset=1.0)
        data = b"Wire format test"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        compressed = compress(stream, key, level=1)
        wire = compressed.to_bytes()
        recovered = EncodedStream.from_bytes(wire)
        assert decode(recovered, key) == data


# ═════════════════════════════════════════════════════════════════════════
# Hypothesis property-based tests
# ═════════════════════════════════════════════════════════════════════════


# Strategy for random braids on 4 strands.
_braid_gens_st = st.lists(
    st.sampled_from([-3, -2, -1, 1, 2, 3]),
    min_size=1,
    max_size=20,
)


class TestCompressHypothesis:
    @given(gens=_braid_gens_st)
    @settings(max_examples=200, deadline=None)
    def test_matrix_preserved_random(self, gens: list[int]) -> None:
        """Compressed braid is always matrix-equivalent to original."""
        b = BraidEquation(4, gens, sector="TSR")
        result = compress_braid(b, level=2, max_rewrites=20)
        if result.generators:
            assert _matrix_equivalent(b, result)

    @given(gens=_braid_gens_st)
    @settings(max_examples=200, deadline=None)
    def test_writhe_preserved_random(self, gens: list[int]) -> None:
        """Compressed braid preserves writhe."""
        b = BraidEquation(4, gens, sector="TSR")
        result = compress_braid(b, level=2, max_rewrites=20)
        assert writhe(b) == writhe(result)

    @given(data=st.binary(min_size=1, max_size=128))
    @settings(max_examples=50, deadline=None)
    def test_encode_compress_decode_random(self, data: bytes) -> None:
        """Random data survives encode → compress → decode."""
        key = keygen(sector="TSR", n_strands=4, theta_offset=1.0)
        stream = encode(data, key, generators_per_block=_K_SMALL)
        compressed = compress(stream, key, level=1)
        assert decode(compressed, key) == data
