"""Tests for the lightweight Pauli stub in _pauli_compat."""

import pytest

from braidcodec.algebra._pauli_compat import Pauli


class TestPauli:
    def test_valid_label(self) -> None:
        p = Pauli("IXYZ")
        assert p.to_label() == "IXYZ"

    def test_invalid_label_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid Pauli label"):
            Pauli("ABC")

    def test_repr(self) -> None:
        p = Pauli("ZZ")
        assert repr(p) == "Pauli('ZZ')"

    def test_eq_same(self) -> None:
        assert Pauli("XY") == Pauli("XY")

    def test_eq_different(self) -> None:
        assert Pauli("XY") != Pauli("YX")

    def test_eq_not_implemented(self) -> None:
        assert Pauli("X") != "X"

    def test_hash(self) -> None:
        s = {Pauli("X"), Pauli("X"), Pauli("Y")}
        assert len(s) == 2

    def test_single_char_labels(self) -> None:
        for c in "IXYZ":
            p = Pauli(c)
            assert p.to_label() == c
