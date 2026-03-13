"""Tests for BraidKey generation and serialization."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from braidcodec._exceptions import FormatError, KeyValidationError
from braidcodec.algebra.braid_equations import get_sector_r_matrix
from braidcodec.crypto.keys import (
    BraidKey,
    _compute_key_id,
    key_from_bytes,
    key_to_bytes,
    keygen,
)

_TWO_PI = 2.0 * math.pi


# ── Construction ──────────────────────────────────────────────────────────


class TestKeyConstruction:
    def test_default_keygen(self) -> None:
        key = keygen()
        assert key.sector == "TSR"
        assert key.n_strands == 4
        assert 0.0 <= key.theta_offset < _TWO_PI
        assert len(key.key_id) == 32

    def test_explicit_offset(self) -> None:
        key = keygen(theta_offset=1.0)
        assert key.theta_offset == 1.0

    def test_all_sectors(self) -> None:
        for sector in ("Identity", "TSR", "Ising", "Fibonacci", "SU2k2"):
            key = keygen(sector=sector, theta_offset=1.0)
            assert key.sector == sector

    def test_deterministic_key_id(self) -> None:
        key1 = keygen(theta_offset=1.0)
        key2 = keygen(theta_offset=1.0)
        assert key1.key_id == key2.key_id

    def test_unique_key_ids(self) -> None:
        key1 = keygen(theta_offset=0.5)
        key2 = keygen(theta_offset=3.0)
        assert key1.key_id != key2.key_id

    def test_theta_effective(self) -> None:
        key = keygen(sector="TSR", theta_offset=1.0)
        # theta_effective = base_theta(TSR) + offset
        assert key.theta_effective > 1.0

    def test_sector_params_dict(self) -> None:
        key = keygen(theta_offset=1.0)
        params = key.sector_params
        assert "theta" in params
        assert params["theta"] == key.theta_effective


# ── Validation ────────────────────────────────────────────────────────────


class TestKeyValidation:
    def test_bad_sector(self) -> None:
        with pytest.raises(KeyValidationError, match=r"Unknown sector"):
            BraidKey(sector="BadSector", n_strands=4, theta_offset=1.0, key_id="a" * 32)

    def test_bad_n_strands(self) -> None:
        with pytest.raises(KeyValidationError, match=r"n_strands"):
            BraidKey(sector="TSR", n_strands=1, theta_offset=1.0, key_id="a" * 32)

    def test_offset_negative(self) -> None:
        with pytest.raises(KeyValidationError, match=r"theta_offset"):
            BraidKey(sector="TSR", n_strands=4, theta_offset=-0.1, key_id="a" * 32)

    def test_offset_too_large(self) -> None:
        with pytest.raises(KeyValidationError, match=r"theta_offset"):
            BraidKey(sector="TSR", n_strands=4, theta_offset=7.0, key_id="a" * 32)

    def test_bad_key_id_short(self) -> None:
        with pytest.raises(KeyValidationError, match=r"key_id"):
            BraidKey(sector="TSR", n_strands=4, theta_offset=1.0, key_id="short")

    def test_bad_key_id_uppercase(self) -> None:
        with pytest.raises(KeyValidationError, match=r"key_id"):
            BraidKey(sector="TSR", n_strands=4, theta_offset=1.0, key_id="A" * 32)


# ── Serialization ─────────────────────────────────────────────────────────


class TestKeySerialization:
    def test_50_byte_length(self) -> None:
        key = keygen(theta_offset=1.0)
        data = key_to_bytes(key)
        assert len(data) == 50

    def test_magic(self) -> None:
        key = keygen(theta_offset=1.0)
        data = key_to_bytes(key)
        assert data[:4] == b"BRDK"

    def test_round_trip(self) -> None:
        key = keygen(theta_offset=1.0)
        data = key_to_bytes(key)
        restored = key_from_bytes(data)
        assert restored.sector == key.sector
        assert restored.n_strands == key.n_strands
        assert restored.theta_offset == key.theta_offset
        assert restored.key_id == key.key_id

    def test_all_sectors_round_trip(self) -> None:
        for sector in ("Identity", "TSR", "Ising", "Fibonacci", "SU2k2"):
            key = keygen(sector=sector, theta_offset=2.0)
            restored = key_from_bytes(key_to_bytes(key))
            assert restored.sector == key.sector
            assert restored.key_id == key.key_id

    def test_bad_magic(self) -> None:
        key = keygen(theta_offset=1.0)
        data = key_to_bytes(key)
        bad = b"XXXX" + data[4:]
        with pytest.raises(FormatError, match=r"magic"):
            key_from_bytes(bad)

    def test_corrupted_digest(self) -> None:
        key = keygen(theta_offset=1.0)
        data = bytearray(key_to_bytes(key))
        data[-1] = (data[-1] + 1) % 256
        with pytest.raises(FormatError, match=r"digest"):
            key_from_bytes(bytes(data))

    def test_wrong_length(self) -> None:
        with pytest.raises(FormatError, match=r"50"):
            key_from_bytes(b"short")


# ── Key ID determinism ────────────────────────────────────────────────────


class TestKeyIdDeterminism:
    def test_same_params_same_id(self) -> None:
        id1 = _compute_key_id("TSR", 4, 1.0)
        id2 = _compute_key_id("TSR", 4, 1.0)
        assert id1 == id2

    def test_different_sector_different_id(self) -> None:
        id1 = _compute_key_id("TSR", 4, 1.0)
        id2 = _compute_key_id("Ising", 4, 1.0)
        assert id1 != id2


# ── Hypothesis ────────────────────────────────────────────────────────────


class TestKeyHypothesis:
    @settings(max_examples=200)
    @given(offset=st.floats(min_value=0.0, max_value=6.28, allow_nan=False, allow_infinity=False))
    def test_offset_round_trip(self, offset: float) -> None:
        key = keygen(theta_offset=offset)
        data = key_to_bytes(key)
        restored = key_from_bytes(data)
        assert restored.key_id == key.key_id
        assert restored.theta_offset == key.theta_offset

    @settings(max_examples=200)
    @given(offset=st.floats(min_value=0.0, max_value=6.28, allow_nan=False, allow_infinity=False))
    def test_keyed_r_matrix_unitarity(self, offset: float) -> None:
        key = keygen(theta_offset=offset)
        r_matrix = get_sector_r_matrix(
            key.sector, inverse=False, theta_override=key.theta_effective
        )
        product = r_matrix @ r_matrix.conj().T
        assert np.allclose(product, np.eye(4), atol=1e-10)
