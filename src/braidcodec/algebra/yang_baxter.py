"""
yang_baxter.py — Categorical validation for Yang-Baxter equation.

Python mirror of algebra/yang_baxter.jl.
Replaces Catlab dependency with lightweight list-of-lists morphism diagrams
and matrix-level verification.

Yang-Baxter equation:  σᵢσᵢ₊₁σᵢ = σᵢ₊₁σᵢσᵢ₊₁
"""

from __future__ import annotations

from typing import NamedTuple

from numpy.linalg import norm

from .braid_equations import (
    BraidEquation,
    contract_braid_tensor,
    simplify_braid,
)

# =============================================================================
# Braid Diagrams (lightweight Catlab replacement)
# =============================================================================


def create_braid_diagram(generators: list[int], n: int) -> list[list[str]]:
    """Create morphism diagram for braid using categorical composition.

    Each generator σᵢ is represented as a list of morphism labels:
    ``[id, ..., id, σ/σ_inv, id, ..., id]``.

    Parameters
    ----------
    generators : list[int]
        Signed generator indices. Positive = σᵢ, negative = σᵢ⁻¹.
    n : int
        Number of strands.

    Returns
    -------
    list[list[str]]
        One entry per generator; each entry is a list of labels.
    """
    morphisms: list[list[str]] = []

    for gen in generators:
        i = abs(gen)
        is_inv = gen < 0

        left_ids = ["id"] * (i - 1) if i > 1 else []
        sigma = "σ_inv" if is_inv else "σ"
        right_ids = ["id"] * (n - i - 1) if i < n - 1 else []

        morphisms.append([*left_ids, sigma, *right_ids])

    return morphisms


# =============================================================================
# YB Morphism Verification
# =============================================================================


def verify_yang_baxter_morphism(b1: BraidEquation, b2: BraidEquation) -> bool:
    """Verify two braids are matrix-equivalent (categorical isomorphism).

    Contracts both braids and checks ||M₁ − M₂|| < 1e-10.
    """
    m1 = contract_braid_tensor(b1)
    m2 = contract_braid_tensor(b2)
    return bool(norm(m1 - m2) < 1e-10)


# =============================================================================
# Category Axiom Validation
# =============================================================================


def validate_braid_category(braid: BraidEquation) -> bool:
    """Validate braid satisfies category axioms.

    Checks:
    1. Generator indices in range [1, n-1].
    2. No structural violations.
    """
    n = braid.n_strands
    for g in braid.generators:
        i = abs(g)
        if i < 1 or i >= n:
            return False
    return True


# =============================================================================
# Catlab Equivalence
# =============================================================================


def check_yb_equivalence_catlab(b1: BraidEquation, b2: BraidEquation) -> bool:
    """Check equivalence under Yang-Baxter using simplification + contraction.

    Simplifies both braids, contracts, and compares matrices.
    """
    if b1.n_strands != b2.n_strands:
        msg = f"Strand count mismatch: {b1.n_strands} vs {b2.n_strands}"
        raise ValueError(msg)

    s1 = simplify_braid(b1)
    s2 = simplify_braid(b2)

    m1 = contract_braid_tensor(s1)
    m2 = contract_braid_tensor(s2)

    return bool(norm(m1 - m2) < 1e-10)


# =============================================================================
# Complexity Metrics
# =============================================================================


class BraidComplexity(NamedTuple):
    length: int
    unique_generators: int
    inverse_count: int
    max_generator: int


def compute_braid_complexity(braid: BraidEquation) -> BraidComplexity:
    """Compute categorical complexity metrics for a braid."""
    gens = braid.generators
    return BraidComplexity(
        length=len(gens),
        unique_generators=len(set(abs(g) for g in gens)),
        inverse_count=sum(1 for g in gens if g < 0),
        max_generator=max((abs(g) for g in gens), default=0),
    )
