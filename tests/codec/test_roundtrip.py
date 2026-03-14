"""End-to-end round-trip tests: encode → (wire format) → decode == identity.

Covers multiple sizes, all sectors, wire-format serialization, and
Hypothesis-driven fuzz testing.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from braidcodec.codec.decoder import decode
from braidcodec.codec.encoder import encode
from braidcodec.codec.schema import EncodedStream
from braidcodec.crypto.integrity import verify
from braidcodec.crypto.keys import BraidKey, keygen

# ── Helpers ───────────────────────────────────────────────────────────────

_K_SMALL: int = 8  # tier 2, manageable state-sum


def _make_key(sector: str = "TSR", theta_offset: float = 1.0, n_strands: int = 4) -> BraidKey:
    return keygen(sector=sector, n_strands=n_strands, theta_offset=theta_offset)


# ── Parametrised sizes ────────────────────────────────────────────────────


class TestRoundTripSizes:
    @pytest.mark.parametrize(
        "size",
        [0, 1, 2, 10, 32, 100, 1024],
        ids=["0B", "1B", "2B", "10B", "32B", "100B", "1KB"],
    )
    def test_size_round_trip(self, size: int) -> None:
        key = _make_key()
        data = bytes(range(256)) * (size // 256 + 1)
        data = data[:size]
        stream = encode(data, key, generators_per_block=_K_SMALL)
        assert decode(stream, key, verify=False) == data


# ── All sectors × two sizes ──────────────────────────────────────────────


class TestRoundTripSectors:
    @pytest.mark.parametrize("sector", ["Identity", "TSR", "Ising", "Fibonacci", "SU2k2"])
    @pytest.mark.parametrize("size", [5, 50], ids=["5B", "50B"])
    def test_sector_size(self, sector: str, size: int) -> None:
        key = _make_key(sector=sector)
        data = bytes(range(size))
        stream = encode(data, key, generators_per_block=_K_SMALL)
        assert decode(stream, key) == data


# ── Wire-format round-trip ────────────────────────────────────────────────


class TestWireFormatRoundTrip:
    def test_encode_serialize_deserialize_decode(self) -> None:
        """Full pipeline: encode → to_bytes → from_bytes → decode."""
        key = _make_key()
        data = b"wire format round-trip test"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        wire = stream.to_bytes()
        restored = EncodedStream.from_bytes(wire)
        assert decode(restored, key) == data

    def test_verify_after_deserialization(self) -> None:
        """Verify passes after wire-format round-trip."""
        key = _make_key()
        data = b"integrity after wire format"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        wire = stream.to_bytes()
        restored = EncodedStream.from_bytes(wire)
        result = verify(restored, key)
        assert result.valid is True


# ── Hypothesis: arbitrary bytes (verify=False for speed) ──────────────────


class TestRoundTripHypothesisNoVerify:
    @given(data=st.binary(max_size=1024))
    @settings(max_examples=200, deadline=None)
    def test_arbitrary_bytes(self, data: bytes) -> None:
        key = _make_key()
        stream = encode(data, key, generators_per_block=_K_SMALL)
        assert decode(stream, key, verify=False) == data


# ── Hypothesis: small blocks with full verification ───────────────────────


class TestRoundTripHypothesisVerified:
    @given(data=st.binary(max_size=256))
    @settings(max_examples=100, deadline=None)
    def test_small_verified(self, data: bytes) -> None:
        """Tier-2 round-trip with full Jones verification."""
        key = _make_key()
        stream = encode(data, key, generators_per_block=_K_SMALL)
        assert decode(stream, key, verify=True) == data


# ── Hypothesis: wire-format serialization round-trip ──────────────────────


class TestRoundTripHypothesisWire:
    @given(data=st.binary(max_size=512))
    @settings(max_examples=100, deadline=None)
    def test_wire_round_trip(self, data: bytes) -> None:
        key = _make_key()
        stream = encode(data, key, generators_per_block=_K_SMALL)
        wire = stream.to_bytes()
        restored = EncodedStream.from_bytes(wire)
        assert decode(restored, key, verify=False) == data
