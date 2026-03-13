"""Shared type aliases and tolerance constants for BraidCodec."""

from __future__ import annotations

from typing import Final, Literal

GeneratorSeq = list[int]
"""Sequence of signed braid generators."""

SectorName = Literal["Identity", "TSR", "Ising", "Fibonacci", "SU2k2"]
"""Valid anyon sector names."""

JonesValue = complex
"""Jones polynomial evaluation."""

ByteChunk = bytes
"""Fixed-size input block."""

# Tolerance constants
JONES_TOLERANCE: Final[float] = 1e-8
"""Maximum |ΔJ| for Jones polynomial comparison."""

MATRIX_TOLERANCE: Final[float] = 1e-10
"""Maximum Frobenius norm for matrix equality checks."""

UNITARITY_TOLERANCE: Final[float] = 1e-10
"""Maximum deviation from unitarity (||RR† - I||)."""
