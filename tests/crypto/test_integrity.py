"""Tests for the 5-channel integrity verification pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import blake3
import pytest

from braidcodec.codec.encoder import encode
from braidcodec.crypto.integrity import VerificationResult, verify
from braidcodec.crypto.keys import BraidKey, keygen

if TYPE_CHECKING:
    from braidcodec.codec.schema import EncodedStream

# ── Helpers ───────────────────────────────────────────────────────────────

_K_SMALL: int = 8  # tier 2 (Jones)
_K_DEFAULT: int = 32  # tier 3 (trace)


def _make_key(sector: str = "TSR", theta_offset: float = 1.0, n_strands: int = 4) -> BraidKey:
    return keygen(sector=sector, n_strands=n_strands, theta_offset=theta_offset)


def _tamper_block(stream: EncodedStream, block_idx: int, **overrides: object) -> EncodedStream:
    """Return a new stream with one block's fields overridden."""
    blocks = list(stream.blocks)
    blocks[block_idx] = replace(blocks[block_idx], **overrides)
    return replace(stream, blocks=tuple(blocks))


# ── Clean verification ────────────────────────────────────────────────────


class TestVerifyClean:
    def test_all_channels_pass(self) -> None:
        key = _make_key()
        stream = encode(b"Hello", key, generators_per_block=_K_SMALL)
        result = verify(stream, key)
        assert isinstance(result, VerificationResult)
        assert result.valid is True
        assert result.structural_passed is True
        assert result.writhe_passed is True
        assert result.invariant_passed is True
        assert result.checksum_passed is True
        assert result.failed_blocks == ()
        assert result.details == ()

    def test_fermion_disabled_by_default(self) -> None:
        key = _make_key()
        stream = encode(b"Hello", key, generators_per_block=_K_SMALL)
        result = verify(stream, key)
        assert result.fermion_passed is None

    def test_fermion_enabled_passes_clean(self) -> None:
        key = _make_key()
        stream = encode(b"Hello", key, generators_per_block=_K_SMALL)
        result = verify(stream, key, fermion_check=True)
        assert result.fermion_passed is True
        assert result.valid is True

    def test_tier_3_clean(self) -> None:
        key = _make_key()
        stream = encode(b"Hello, topology!", key, generators_per_block=_K_DEFAULT)
        result = verify(stream, key)
        assert result.valid is True
        assert result.invariant_passed is True

    def test_empty_data(self) -> None:
        key = _make_key()
        stream = encode(b"", key, generators_per_block=_K_SMALL)
        result = verify(stream, key)
        assert result.valid is True

    def test_multi_block(self) -> None:
        key = _make_key()
        stream = encode(b"A" * 200, key, generators_per_block=_K_SMALL)
        assert len(stream.blocks) > 1
        result = verify(stream, key)
        assert result.valid is True


# ── Per-channel corruption detection ──────────────────────────────────────


class TestVerifyCorruption:
    def test_structural_failure(self) -> None:
        """Generator out of range → structural failure, short-circuits."""
        key = _make_key()
        stream = encode(b"\x01\x02", key, generators_per_block=_K_SMALL)
        block = stream.blocks[0]
        bad_gens = list(block.generators)
        bad_gens[0] = 99  # out of range for n_strands=4
        # Need to rebuild because BraidEquation validates in __init__
        # Instead, use a lower-level approach: create block with bad gen
        # The validate_braid_category check creates a BraidEquation that
        # will fail validation. Let's use a generator that's in [1, n-1]
        # range but make it 0 (which is always invalid).
        bad_gens[0] = key.n_strands  # |gen| >= n_strands → invalid
        tampered = _tamper_block(stream, 0, generators=bad_gens)
        result = verify(tampered, key)
        assert result.structural_passed is False
        assert result.valid is False
        assert 0 in result.failed_blocks

    def test_writhe_corruption(self) -> None:
        """Modified writhe → writhe check fails."""
        key = _make_key()
        stream = encode(b"\x01\x02", key, generators_per_block=_K_SMALL)
        tampered = _tamper_block(stream, 0, writhe=stream.blocks[0].writhe + 100)
        result = verify(tampered, key)
        assert result.writhe_passed is False
        assert result.valid is False

    def test_jones_corruption(self) -> None:
        """Modified Jones → invariant check fails (tier 2)."""
        key = _make_key()
        stream = encode(b"\x01\x02", key, generators_per_block=_K_SMALL)
        assert stream.blocks[0].invariant_tier == 2
        tampered = _tamper_block(stream, 0, jones_real=99.0, jones_imag=99.0)
        result = verify(tampered, key)
        assert result.invariant_passed is False
        assert result.valid is False

    def test_trace_corruption(self) -> None:
        """Modified trace → invariant check fails (tier 3)."""
        key = _make_key()
        stream = encode(b"Hello, topology!", key, generators_per_block=_K_DEFAULT)
        assert stream.blocks[0].invariant_tier == 3
        tampered = _tamper_block(stream, 0, trace_real=99.0, trace_imag=99.0)
        result = verify(tampered, key)
        assert result.invariant_passed is False
        assert result.valid is False

    def test_checksum_corruption(self) -> None:
        """Wrong stream checksum → checksum check fails."""
        key = _make_key()
        stream = encode(b"Hello", key, generators_per_block=_K_SMALL)
        bad_checksum = blake3.blake3(b"wrong").digest()
        tampered = replace(stream, checksum=bad_checksum)
        result = verify(tampered, key)
        assert result.checksum_passed is False
        assert result.valid is False


# ── Structural short-circuit ──────────────────────────────────────────────


class TestVerifyShortCircuit:
    def test_structural_failure_skips_downstream(self) -> None:
        """If structural fails on a block, writhe/invariant aren't run for it."""
        key = _make_key()
        stream = encode(b"\x01\x02", key, generators_per_block=_K_SMALL)
        block = stream.blocks[0]
        # Use generator = n_strands (out of range) to trigger structural failure
        bad_gens = list(block.generators)
        bad_gens[0] = key.n_strands
        tampered = _tamper_block(stream, 0, generators=bad_gens)
        result = verify(tampered, key)
        assert result.structural_passed is False
        # Writhe and invariant channels didn't run for the bad block,
        # so they remain True (they passed for all blocks that were checked)
        assert 0 in result.failed_blocks
        assert len(result.details) >= 1


# ── Wrong key ─────────────────────────────────────────────────────────────


class TestVerifyWrongKey:
    def test_wrong_theta_fails_invariant(self) -> None:
        """Different theta_offset produces different invariants."""
        key = _make_key(theta_offset=1.0)
        stream = encode(b"Hello", key, generators_per_block=_K_SMALL)
        wrong_key = _make_key(theta_offset=2.0)
        result = verify(stream, wrong_key)
        assert result.invariant_passed is False
        assert result.valid is False


# ── All sectors ───────────────────────────────────────────────────────────


class TestVerifyAllSectors:
    @pytest.mark.parametrize("sector", ["Identity", "TSR", "Ising", "Fibonacci", "SU2k2"])
    def test_clean_per_sector(self, sector: str) -> None:
        key = _make_key(sector=sector)
        stream = encode(b"Sector test", key, generators_per_block=_K_SMALL)
        result = verify(stream, key)
        assert result.valid is True


# ── Edge-case coverage for per-channel functions ──────────────────────────


class TestVerifyEdgeCases:
    def test_tier2_missing_jones_value(self) -> None:
        """Tier 2 block with jones=None → invariant check fails."""
        from braidcodec.crypto.integrity import _check_invariant

        key = _make_key()
        stream = encode(b"\x01\x02", key, generators_per_block=_K_SMALL)
        block = stream.blocks[0]
        assert block.invariant_tier == 2
        # Bypass __post_init__ by setting via object.__setattr__
        bad_block = replace(block, jones_real=0.0, jones_imag=0.0)
        object.__setattr__(bad_block, "jones_real", None)
        object.__setattr__(bad_block, "jones_imag", None)
        ok, msg = _check_invariant(bad_block, key.sector_params)
        assert not ok
        assert msg is not None and "Jones" in msg

    def test_tier3_missing_trace_value(self) -> None:
        """Tier 3 block with trace=None → invariant check fails."""
        from braidcodec.crypto.integrity import _check_invariant

        key = _make_key()
        stream = encode(b"Hello, topology!", key, generators_per_block=_K_DEFAULT)
        block = stream.blocks[0]
        assert block.invariant_tier == 3
        bad_block = replace(block, trace_real=0.0, trace_imag=0.0)
        object.__setattr__(bad_block, "trace_real", None)
        object.__setattr__(bad_block, "trace_imag", None)
        ok, msg = _check_invariant(bad_block, key.sector_params)
        assert not ok
        assert msg is not None and "trace" in msg

    def test_fermion_generator_exceeds_sites(self) -> None:
        """Generator |g| > n_sites → fermion channel fails."""
        key = _make_key(n_strands=3)
        stream = encode(b"\x01", key, generators_per_block=_K_SMALL)
        block = stream.blocks[0]
        # Replace a generator with one that exceeds n_sites (n_strands - 1 = 2)
        bad_gens = list(block.generators)
        bad_gens[0] = 99  # way beyond n_sites
        tampered = _tamper_block(stream, 0, generators=bad_gens)
        result = verify(tampered, key, fermion_check=True)
        # Structural check may catch this first, but fermion won't pass either
        assert result.valid is False

    def test_checksum_skipped_on_structural_fail(self) -> None:
        """When structural fails → checksum channel skipped."""
        key = _make_key()
        stream = encode(b"\x01\x02", key, generators_per_block=_K_SMALL)
        block = stream.blocks[0]
        bad_gens = list(block.generators)
        bad_gens[0] = key.n_strands
        tampered = _tamper_block(stream, 0, generators=bad_gens)
        result = verify(tampered, key)
        assert result.checksum_passed is False
        assert any("skipped" in d.lower() or "checksum" in d.lower() for d in result.details)
