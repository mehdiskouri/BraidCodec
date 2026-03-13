"""Lightweight Pauli stub for when qiskit is not installed.

Covers only the surface used by fermion_bounds: ``Pauli(label_string)`` constructor
returning an object with a ``.to_label()`` method. No operators or matrix methods.
"""

from __future__ import annotations


class Pauli:
    """Minimal Pauli operator stub (Z-string construction only)."""

    __slots__ = ("_label",)

    def __init__(self, label: str) -> None:
        if not all(c in "IXYZ" for c in label):
            msg = f"Invalid Pauli label: {label!r}"
            raise ValueError(msg)
        self._label = label

    def to_label(self) -> str:
        return self._label

    def __repr__(self) -> str:
        return f"Pauli('{self._label}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pauli):
            return NotImplemented
        return self._label == other._label

    def __hash__(self) -> int:
        return hash(self._label)
