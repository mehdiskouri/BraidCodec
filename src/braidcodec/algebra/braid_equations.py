"""
braid_equations.py — Braid group representations with Jones polynomial invariants.

Python mirror of algebra/braid_equations.jl.
Uses numpy for Kronecker products and matrix operations.

Theoretical Foundation (from GCO proof):
    The validity of a proof is its topological invariant.
    For a proof represented by a braid L, its truth value is V(L) (Jones polynomial).

    NO SWAP MATRICES — SWAP is topologically trivial.
    All R-matrices satisfy: R₁₂R₁₃R₂₃ = R₂₃R₁₃R₁₂ (Yang-Baxter)
"""

from __future__ import annotations

import math
import re

import numpy as np

from .tsr_constants import C_CONSTANT

# =============================================================================
# Valid sectors
# =============================================================================

VALID_SECTORS = frozenset({"Identity", "TSR", "Ising", "Fibonacci", "SU2k2"})


# =============================================================================
# BraidEquation
# =============================================================================


class BraidEquation:
    """Braid group element with tensor representation.

    Parameters
    ----------
    n_strands : int
        Number of strands (≥ 2).
    generators : list[int]
        Generator sequence. Positive = σᵢ, negative = σᵢ⁻¹. Must be non-zero
        with |g| < n_strands.
    sector : str
        Anyon sector: "Identity", "TSR", "Ising", "Fibonacci", "SU2k2".
    simplified : bool
        Whether this braid has been simplified.
    """

    __slots__ = (
        "_sector_params",
        "braid_matrix",
        "generators",
        "n_strands",
        "sector",
        "simplified",
    )

    def __init__(
        self,
        n_strands: int,
        generators: list[int],
        *,
        sector: str = "Identity",
        simplified: bool = False,
        _sector_params: dict[str, float] | None = None,
    ):
        if n_strands < 2:
            msg = f"Need ≥2 strands, got {n_strands}"
            raise ValueError(msg)
        if not all(g != 0 and abs(g) < n_strands for g in generators):
            msg = f"Invalid generator: must be non-zero and |g| < n_strands={n_strands}"
            raise ValueError(msg)
        if sector not in VALID_SECTORS:
            msg = f"Unknown sector '{sector}'. Valid: {sorted(VALID_SECTORS)}"
            raise ValueError(msg)

        self.n_strands: int = n_strands
        self.generators: list[int] = list(generators)
        self.braid_matrix: np.ndarray | None = None
        self.sector: str = sector
        self.simplified: bool = simplified
        self._sector_params: dict[str, float] | None = _sector_params

    # -- dunder methods -------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BraidEquation):
            return NotImplemented
        return (
            self.n_strands == other.n_strands
            and self.generators == other.generators
            and self.sector == other.sector
        )

    def __len__(self) -> int:
        return len(self.generators)

    def __mul__(self, other: BraidEquation) -> BraidEquation:
        return compose(self, other)

    def __invert__(self) -> BraidEquation:
        return BraidEquation(
            self.n_strands,
            [-g for g in reversed(self.generators)],
            sector=self.sector,
            simplified=False,
            _sector_params=self._sector_params,
        )

    def __repr__(self) -> str:
        return (
            f"BraidEquation(n_strands={self.n_strands}, "
            f"generators={self.generators}, sector='{self.sector}')"
        )

    def __str__(self) -> str:
        return braid_to_string(self)

    # -- predicates -----------------------------------------------------------

    def is_identity(self) -> bool:
        return len(self.generators) == 0


# =============================================================================
# R-matrix construction
# =============================================================================


# ── Matrix caches ────────────────────────────────────────────────────────
# Keyed on (sector, inverse, theta_override) and (i, n, inverse, sector, theta)
# respectively.  Entries are immutable once inserted.
_r_matrix_cache: dict[tuple[str, bool, float | None], np.ndarray] = {}
_gen_matrix_cache: dict[tuple[int, int, bool, str, float | None], np.ndarray] = {}


def get_sector_r_matrix(
    sector: str,
    inverse: bool,
    *,
    theta_override: float | None = None,
) -> np.ndarray:
    """4×4 unitary phased-SWAP R-matrix for a given anyon sector.

    R(α,β,γ) = diag(e^{iα}, 0, 0, e^{iγ}) with off-diag e^{iβ} SWAP block.

    Parameters
    ----------
    sector : str
        One of "Identity", "TSR", "Ising", "Fibonacci", "SU2k2".
    inverse : bool
        If True, return R⁻¹ = R†.
    theta_override : float | None
        If set, use this theta instead of the sector-derived value.

    Returns
    -------
    np.ndarray
        4×4 complex unitary matrix.
    """
    cache_key = (sector, inverse, theta_override)
    cached = _r_matrix_cache.get(cache_key)
    if cached is not None:
        return cached

    if theta_override is not None:
        theta = theta_override
    elif sector in ("Identity", "TSR"):
        theta = math.pi * C_CONSTANT
    elif sector == "Ising":
        theta = math.pi / 4
    elif sector == "Fibonacci":
        theta = 4 * math.pi / 5
    elif sector == "SU2k2":
        theta = math.pi / 2
    else:
        raise ValueError(f"Unknown sector: {sector}. Valid: {sorted(VALID_SECTORS)}")

    if inverse:
        theta = -theta

    alpha = theta
    beta = theta
    gamma = 3 * theta

    R = np.array(
        [
            [np.exp(1j * alpha), 0, 0, 0],
            [0, 0, np.exp(1j * beta), 0],
            [0, np.exp(1j * beta), 0, 0],
            [0, 0, 0, np.exp(1j * gamma)],
        ],
        dtype=np.complex128,
    )
    _r_matrix_cache[cache_key] = R
    return R


def get_braid_generator_matrix(
    i: int,
    n: int,
    inverse: bool,
    sector: str,
    *,
    theta_override: float | None = None,
) -> np.ndarray:
    """Full d^n × d^n unitary matrix for generator σᵢ on n strands.

    Builds I^{⊗(i-1)} ⊗ R ⊗ I^{⊗(n-i-1)} via Kronecker products.

    Parameters
    ----------
    i : int
        Generator index (1-based, 1 ≤ i < n).
    n : int
        Number of strands.
    inverse : bool
        If True, use R⁻¹.
    sector : str
        Anyon sector.
    theta_override : float | None
        If set, override the sector-derived theta for R-matrix construction.

    Returns
    -------
    np.ndarray
        (2^n × 2^n) complex unitary matrix.
    """
    if not (1 <= i < n):
        msg = f"Generator index i={i} out of range [1, {n - 1}]"
        raise ValueError(msg)

    cache_key = (i, n, inverse, sector, theta_override)
    cached = _gen_matrix_cache.get(cache_key)
    if cached is not None:
        return cached

    d = 2
    R = get_sector_r_matrix(sector, inverse, theta_override=theta_override)
    I2 = np.eye(d, dtype=np.complex128)

    result: np.ndarray = np.array([[1.0 + 0j]], dtype=np.complex128)

    # Left identities: strands 1..(i-1)
    for _ in range(i - 1):
        result = np.asarray(np.kron(result, I2), dtype=np.complex128)

    # R-matrix on strands i and i+1
    result = np.asarray(np.kron(result, R), dtype=np.complex128)

    # Right identities: strands (i+2)..n
    for _ in range(i + 1, n):
        result = np.asarray(np.kron(result, I2), dtype=np.complex128)

    _gen_matrix_cache[cache_key] = result
    return result


# =============================================================================
# Contraction
# =============================================================================


def contract_braid_tensor(braid: BraidEquation) -> np.ndarray:
    """Contract all braid generators into a single unitary matrix.

    Left-to-right application: for generators [g1, g2, g3],
    the result is M(g3) @ M(g2) @ M(g1).

    Caches result on ``braid.braid_matrix``.
    """
    if braid.braid_matrix is not None:
        return braid.braid_matrix

    n = braid.n_strands
    D = 2**n

    theta_ov = braid._sector_params.get("theta") if braid._sector_params else None

    # Pre-build a local lookup: signed generator → cached matrix.
    # Avoids abs()/comparison per iteration and reduces to int dict lookup.
    gen_mats: dict[int, np.ndarray] = {}
    for gen in braid.generators:
        if gen not in gen_mats:
            gen_mats[gen] = get_braid_generator_matrix(
                abs(gen), n, gen < 0, braid.sector, theta_override=theta_ov
            )

    result = np.eye(D, dtype=np.complex128)
    for gen in braid.generators:
        result = gen_mats[gen] @ result

    braid.braid_matrix = result
    return result


# =============================================================================
# Simplification
# =============================================================================


def simplify_braid(braid: BraidEquation) -> BraidEquation:
    """Cancel inverse pairs σᵢσᵢ⁻¹ → e. Returns new BraidEquation."""
    gens = list(braid.generators)

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(gens) - 1:
            if gens[i] == -gens[i + 1]:
                del gens[i : i + 2]
                changed = True
                break
            i += 1

    return BraidEquation(
        braid.n_strands,
        gens,
        sector=braid.sector,
        simplified=True,
        _sector_params=braid._sector_params,
    )


# =============================================================================
# Factory
# =============================================================================


def create_braid_equation(
    generators: list[int] | list[str],
    n: int,
    *,
    sector: str = "Identity",
) -> BraidEquation:
    """Create a BraidEquation from int list or symbol string list.

    Symbol format examples: "σ1", "σ2inv", "s_1", "s_2inv", "e", "identity".
    """
    if not generators:
        return BraidEquation(n, [], sector=sector)

    # If first element is a string, parse symbols
    first = generators[0]
    if isinstance(first, str):
        str_gens: list[str] = [str(g) for g in generators]
        gen_indices: list[int] = []
        for s in str_gens:
            if s in ("e", "identity"):
                continue
            elif "inv" in s:
                idx_str = re.sub(r"[σs_]", "", s.replace("inv", ""))
                idx = int(idx_str)
                if not (1 <= idx < n):
                    msg = f"Invalid generator index: {idx} (must be 1 <= idx < {n})"
                    raise ValueError(msg)
                gen_indices.append(-idx)
            else:
                idx_str = re.sub(r"[σs_]", "", s)
                idx = int(idx_str)
                if not (1 <= idx < n):
                    msg = f"Invalid generator index: {idx} (must be 1 <= idx < {n})"
                    raise ValueError(msg)
                gen_indices.append(idx)
        return BraidEquation(n, gen_indices, sector=sector)
    else:
        return BraidEquation(n, [int(g) for g in generators], sector=sector)


# =============================================================================
# Composition / Inverse
# =============================================================================


def compose(b1: BraidEquation, b2: BraidEquation) -> BraidEquation:
    """Compose two braids (concatenate generators)."""
    if b1.n_strands != b2.n_strands or b1.sector != b2.sector:
        msg = "Braids must have same n_strands and sector"
        raise ValueError(msg)
    return BraidEquation(
        b1.n_strands,
        b1.generators + b2.generators,
        sector=b1.sector,
        simplified=False,
        _sector_params=b1._sector_params,
    )


# =============================================================================
# String representation
# =============================================================================


def braid_to_string(braid: BraidEquation) -> str:
    """Human-readable string for a braid: σ1σ2⁻¹ [TSR]."""
    if not braid.generators:
        return "e"
    parts = [f"σ{abs(g)}{'⁻¹' if g < 0 else ''}" for g in braid.generators]
    tag = "" if braid.sector == "Identity" else f" [{braid.sector}]"
    return "".join(parts) + tag


# =============================================================================
# Jones Polynomial / Kauffman Bracket / Writhe
# =============================================================================


def writhe(braid: BraidEquation) -> int:
    """Algebraic crossing number: w(L) = Σ sign(σᵢ)."""
    return sum(1 if g > 0 else -1 for g in braid.generators)


def get_kauffman_A(sector: str) -> complex:
    """Kauffman bracket A parameter. Jones variable: t = A⁻⁴."""
    if sector in ("Identity", "TSR"):
        return complex(np.exp(1j * math.pi * C_CONSTANT / 4))
    elif sector == "Ising":
        return complex(np.exp(1j * math.pi / 16))
    elif sector == "Fibonacci":
        return complex(np.exp(1j * math.pi / 5))
    elif sector == "SU2k2":
        return complex(np.exp(1j * math.pi / 8))
    else:
        raise ValueError(f"Unknown sector: {sector}")


def _get_effective_A(braid: BraidEquation) -> complex:
    """Kauffman A parameter, respecting keyed ``_sector_params``."""
    if braid._sector_params and "theta" in braid._sector_params:
        return complex(np.exp(1j * braid._sector_params["theta"] / 4))
    return get_kauffman_A(braid.sector)


def count_loops_in_smoothing(n: int, gens: list[int], state: int) -> int:
    """Count closed loops when braid is closed with given smoothing state."""
    # connections[i] = what strand i maps to (0-indexed internally)
    connections = list(range(n))

    for idx, gen in enumerate(gens):
        i = abs(gen) - 1  # 0-indexed strand position
        bit = (state >> idx) & 1
        is_positive = gen > 0

        # A-smoothing (bit=0 for positive): strands pass through
        # B-smoothing (bit=1 for positive): strands reconnect
        do_reconnect = (is_positive and bit == 1) or (not is_positive and bit == 0)

        if do_reconnect:
            connections[i], connections[i + 1] = (
                connections[i + 1],
                connections[i],
            )

    # Count cycles in the permutation
    visited = [False] * n
    n_loops = 0
    for start in range(n):
        if not visited[start]:
            n_loops += 1
            current = start
            while not visited[current]:
                visited[current] = True
                current = connections[current]
    return n_loops


# ── Cycle-count lookup table for vectorised Kauffman bracket ──────────────
_cycle_lut_cache: dict[int, np.ndarray] = {}


def _get_cycle_lut(n: int) -> np.ndarray:
    """Return a LUT mapping permutation encoding → cycle count for n elements."""
    lut = _cycle_lut_cache.get(n)
    if lut is not None:
        return lut
    from itertools import permutations

    lut = np.zeros(n**n, dtype=np.int32)
    factors = [n ** (n - 1 - j) for j in range(n)]
    for perm in permutations(range(n)):
        key = sum(p * f for p, f in zip(perm, factors, strict=False))
        visited = [False] * n
        cycles = 0
        for start in range(n):
            if not visited[start]:
                cycles += 1
                cur = start
                while not visited[cur]:
                    visited[cur] = True
                    cur = perm[cur]
        lut[key] = cycles
    _cycle_lut_cache[n] = lut
    return lut


def _kauffman_bracket_vectorized(
    A: complex,
    d: complex,
    n: int,
    gens: list[int],
    m: int,
) -> complex:
    """Vectorised Kauffman bracket using numpy — O(m·2^m) with low constant."""
    S = 1 << m
    states = np.arange(S, dtype=np.int32)
    gen_arr = np.array(gens, dtype=np.int32)

    # a_power per state: (S,)
    bits = (states[:, None] >> np.arange(m, dtype=np.int32)[None, :]) & 1
    positive = gen_arr[None, :] > 0
    a_powers = np.where(positive, np.where(bits == 0, 1, -1), np.where(bits == 0, -1, 1)).sum(
        axis=1
    )

    # Connection swapping: track permutations for all states at once
    connections = np.tile(np.arange(n, dtype=np.int32), (S, 1))
    for idx in range(m):
        g = gens[idx]
        i = abs(g) - 1
        bit = (states >> idx) & 1
        do_swap = (bit == 1) if g > 0 else (bit == 0)
        ci = connections[:, i].copy()
        ci1 = connections[:, i + 1].copy()
        connections[:, i] = np.where(do_swap, ci1, ci)
        connections[:, i + 1] = np.where(do_swap, ci, ci1)

    # Cycle count via pre-computed LUT
    lut = _get_cycle_lut(n)
    factors = np.array([n ** (n - 1 - j) for j in range(n)], dtype=np.int64)
    perm_keys = (connections.astype(np.int64) * factors[None, :]).sum(axis=1)
    n_loops = lut[perm_keys]

    bracket = np.sum(A ** a_powers.astype(np.float64) * d ** (n_loops - 1).astype(np.float64))
    return complex(bracket)


def kauffman_bracket(braid: BraidEquation) -> complex:
    """Kauffman bracket ⟨L⟩ via state-sum model.

    Satisfies skein relation:
        ⟨crossing⟩ = A⟨0-smoothing⟩ + A⁻¹⟨1-smoothing⟩

    Uses a vectorised numpy implementation for m ≤ 20 generators.
    """
    A = _get_effective_A(braid)
    n = braid.n_strands
    gens = braid.generators

    d = -(A**2) - A ** (-2)  # loop value

    if not gens:
        return d ** (n - 1)

    m = len(gens)

    if m <= 20:
        return _kauffman_bracket_vectorized(A, d, n, gens, m)

    # Scalar fallback for very long braids (>2^20 states).
    bracket = 0.0 + 0.0j

    for state in range(1 << m):
        a_power = 0
        for idx, gen in enumerate(gens):
            bit = (state >> idx) & 1
            if gen > 0:
                a_power += 1 if bit == 0 else -1
            else:
                a_power += -1 if bit == 0 else 1

        n_loops = count_loops_in_smoothing(n, gens, state)
        bracket += A**a_power * d ** (n_loops - 1)

    return complex(bracket)


def jones_polynomial(braid: BraidEquation) -> complex:
    """Jones polynomial V(L) = (-A³)^{-w(L)} · ⟨L⟩.

    Topological invariant: unchanged under Reidemeister moves.
    """
    A = _get_effective_A(braid)
    w = writhe(braid)
    bracket = kauffman_bracket(braid)
    return complex(((-(A**3)) ** (-w)) * bracket)


# =============================================================================
# Registry
# =============================================================================


class BraidEquationRegistry:
    """Thread-unsafe global registry for braid equations (mirrors Julia)."""

    def __init__(self) -> None:
        self._equations: dict[int, BraidEquation] = {}
        self._next_id: int = 1

    def register(self, braid: BraidEquation) -> int:
        """Register a braid, return its integer ID."""
        id_ = self._next_id
        self._next_id += 1
        self._equations[id_] = braid
        return id_

    def get(self, id_: int) -> BraidEquation | None:
        return self._equations.get(id_)

    def has(self, id_: int) -> bool:
        return id_ in self._equations

    def clear(self) -> None:
        self._equations.clear()
        self._next_id = 1

    def stats(self) -> dict[str, int | float]:
        n = len(self._equations)
        n_simplified = sum(1 for b in self._equations.values() if b.simplified)
        total_len = sum(len(b) for b in self._equations.values())
        return {
            "n_braids": n,
            "n_simplified": n_simplified,
            "avg_length": total_len / n if n > 0 else 0.0,
        }

    def braid_to_string(self, id_: int) -> str:
        b = self.get(id_)
        if b is None:
            return f"Braid ID {id_} not found"
        return braid_to_string(b)


BRAID_REGISTRY = BraidEquationRegistry()
