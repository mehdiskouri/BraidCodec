"""Tests for the decode pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import blake3
import pytest

from braidcodec._exceptions import (
    ChecksumError,
    JonesError,
    KeyMismatchError,
    TraceError,
    WritheError,
)
from braidcodec.codec.decoder import decode
from braidcodec.codec.encoder import encode
from braidcodec.crypto.keys import BraidKey, keygen

if TYPE_CHECKING:
    from braidcodec.codec.schema import EncodedStream

# ── Helpers ───────────────────────────────────────────────────────────────

# k=8 → tier 2 (Jones), fast state-sum (2^8 = 256 states).
_K_SMALL: int = 8
# k=32 → tier 3 (trace only).
_K_DEFAULT: int = 32


def _make_key(sector: str = "TSR", theta_offset: float = 1.0, n_strands: int = 4) -> BraidKey:
    return keygen(sector=sector, n_strands=n_strands, theta_offset=theta_offset)


def _tamper_block(stream: EncodedStream, block_idx: int, **overrides: object) -> EncodedStream:
    """Return a new stream with one block's fields overridden (frozen dataclass)."""
    blocks = list(stream.blocks)
    blocks[block_idx] = replace(blocks[block_idx], **overrides)
    return replace(stream, blocks=tuple(blocks))


# ── Basic round-trips ─────────────────────────────────────────────────────


class TestDecodeBasic:
    def test_hello(self) -> None:
        key = _make_key()
        data = b"Hello"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        assert decode(stream, key) == data

    def test_empty(self) -> None:
        key = _make_key()
        data = b""
        stream = encode(data, key, generators_per_block=_K_SMALL)
        assert decode(stream, key) == data

    def test_single_byte(self) -> None:
        key = _make_key()
        data = b"\x42"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        assert decode(stream, key) == data

    def test_exact_block_boundary(self) -> None:
        """Data exactly fills one block — no padding wasted."""
        key = _make_key()
        from braidcodec.codec.chunker import compute_block_size

        bs = compute_block_size(key.n_strands, _K_SMALL)
        data = bytes(range(bs))
        stream = encode(data, key, generators_per_block=_K_SMALL)
        assert decode(stream, key) == data

    def test_multi_block(self) -> None:
        key = _make_key()
        data = b"A" * 200
        stream = encode(data, key, generators_per_block=_K_SMALL)
        assert len(stream.blocks) > 1
        assert decode(stream, key) == data

    def test_tier_3_trace(self) -> None:
        """Tier 3 (trace invariant) round-trips correctly."""
        key = _make_key()
        data = b"Hello, topology!"
        stream = encode(data, key, generators_per_block=_K_DEFAULT)
        # With k=32, tier 3 is used
        for block in stream.blocks:
            assert block.invariant_tier == 3
        assert decode(stream, key) == data


# ── Key mismatch ──────────────────────────────────────────────────────────


class TestDecodeKeyMismatch:
    def test_wrong_sector(self) -> None:
        key = _make_key(sector="TSR")
        stream = encode(b"Hi", key, generators_per_block=_K_SMALL)
        wrong_key = _make_key(sector="Ising")
        with pytest.raises(KeyMismatchError, match="Sector mismatch"):
            decode(stream, wrong_key)

    def test_wrong_strands(self) -> None:
        key = _make_key(n_strands=4)
        stream = encode(b"Hi", key, generators_per_block=_K_SMALL)
        wrong_key = _make_key(n_strands=3)
        with pytest.raises(KeyMismatchError, match="Strand count mismatch"):
            decode(stream, wrong_key)


# ── Tampering detection ──────────────────────────────────────────────────


class TestDecodeTampering:
    def test_flipped_generator_detected_by_writhe(self) -> None:
        """Flipping a positive generator to negative changes writhe."""
        key = _make_key()
        data = b"\x01\x02\x03"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        block = stream.blocks[0]
        gens = list(block.generators)
        # Find a positive generator and negate it
        for i, g in enumerate(gens):
            if g > 0:
                gens[i] = -g
                break
        tampered = _tamper_block(stream, 0, generators=gens)
        with pytest.raises(WritheError, match="writhe mismatch"):
            decode(tampered, key)

    def test_modified_writhe_detected(self) -> None:
        """If stored writhe is wrong, pre-check catches it."""
        key = _make_key()
        data = b"\x01\x02"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        tampered = _tamper_block(stream, 0, writhe=stream.blocks[0].writhe + 999)
        with pytest.raises(WritheError, match="writhe mismatch"):
            decode(tampered, key)

    def test_modified_jones_detected(self) -> None:
        """Corrupted Jones value triggers JonesError on tier-2 block."""
        key = _make_key()
        data = b"\x01\x02"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        block = stream.blocks[0]
        assert block.invariant_tier == 2
        tampered = _tamper_block(
            stream, 0, jones_real=99.0, jones_imag=99.0
        )
        with pytest.raises(JonesError, match="Jones mismatch"):
            decode(tampered, key)

    def test_modified_trace_detected(self) -> None:
        """Corrupted trace value triggers TraceError on tier-3 block."""
        key = _make_key()
        data = b"Hello, topology!"
        stream = encode(data, key, generators_per_block=_K_DEFAULT)
        block = stream.blocks[0]
        assert block.invariant_tier == 3
        tampered = _tamper_block(
            stream, 0, trace_real=99.0, trace_imag=99.0
        )
        with pytest.raises(TraceError, match="trace mismatch"):
            decode(tampered, key)

    def test_modified_checksum_detected(self) -> None:
        """Wrong stream checksum triggers ChecksumError after decode."""
        key = _make_key()
        data = b"Hello"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        bad_checksum = blake3.blake3(b"wrong data").digest()
        tampered = replace(stream, checksum=bad_checksum)
        with pytest.raises(ChecksumError, match="checksum mismatch"):
            decode(tampered, key)


# ── verify=False skips invariant ──────────────────────────────────────────


class TestDecodeVerifyFalse:
    def test_tampered_jones_ok_without_verify(self) -> None:
        """With verify=False, corrupted jones doesn't prevent decode."""
        key = _make_key()
        data = b"\x01\x02"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        tampered = _tamper_block(stream, 0, jones_real=99.0, jones_imag=99.0)
        # Should still succeed — writhe still matches, only Jones is wrong
        result = decode(tampered, key, verify=False)
        assert result == data

    def test_tampered_trace_ok_without_verify(self) -> None:
        """With verify=False, corrupted trace doesn't prevent decode."""
        key = _make_key()
        data = b"Hello, topology!"
        stream = encode(data, key, generators_per_block=_K_DEFAULT)
        tampered = _tamper_block(stream, 0, trace_real=99.0, trace_imag=99.0)
        result = decode(tampered, key, verify=False)
        assert result == data


# ── All sectors ───────────────────────────────────────────────────────────


class TestDecodeAllSectors:
    @pytest.mark.parametrize("sector", ["Identity", "TSR", "Ising", "Fibonacci", "SU2k2"])
    def test_round_trip_per_sector(self, sector: str) -> None:
        key = _make_key(sector=sector)
        data = b"Sector test data"
        stream = encode(data, key, generators_per_block=_K_SMALL)
        assert decode(stream, key) == data
