"""
TSRConstants — Single Source of Truth for Theory of Systemic Resonance Physics.

Python mirror of algebra/TSRConstants.jl.
Provides canonical physics constants and derived functions.

Theoretical Foundation (three proven theorems from ResonanceDynamic.md):

    Theorem 1 — Single Resonator Energy Transfer:
        ẍ + 2ζωₙẋ + ωₙ²x = (F_ext/m)cos(ωt)
        Damping: ζ = √(D'/ℏω)

    Theorem 2 — Coupled Oscillator Synchronization:
        K = √2·Cⁿ·exp(−d/λ)

    Theorem 3 — Many-Body Energy Minimization (Langevin):
        P(S) = Z⁻¹·exp(−E/D')
"""

from __future__ import annotations

import math
import warnings

import numpy as np

# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

C_CONSTANT: float = 0.141416474
"""d-orbital quantum structure constant: C = 1/(5√2) × (1 − 3.452×10⁻⁵)."""

D_PRIME: float = 2.67e-16
"""Decoherence energy scale [J]."""

HBAR: float = 1.054571817e-34
"""Reduced Planck constant [J·s] (CODATA 2018)."""

ALPHA: float = 1.0 / 137.035999084
"""Fine structure constant (CODATA 2018)."""

K_BOLTZMANN: float = 1.380649e-23
"""Boltzmann constant [J/K] (exact, SI 2019)."""

C_LIGHT: float = 299792458.0
"""Speed of light in vacuum [m/s] (exact, SI 2019)."""

# =============================================================================
# OSCILLATOR CONSTRAINTS
# =============================================================================

OMEGA_MIN_HZ: float = 10.0
OMEGA_MAX_HZ: float = 1000.0
OMEGA_ANCHOR_HZ: float = 50.0

OMEGA_MIN: float = 2.0 * math.pi * OMEGA_MIN_HZ
OMEGA_MAX: float = 2.0 * math.pi * OMEGA_MAX_HZ
OMEGA_ANCHOR: float = 2.0 * math.pi * OMEGA_ANCHOR_HZ

DT_PSL: float = 1.0e-4
"""Canonical PSL timestep [s]."""

Q_BASE: float = 5.0
"""Canonical quality factor for oscillators."""

R_FLOOR: float = math.sqrt(2.0) * C_CONSTANT
"""Minimum triangle coherence threshold: √2 · C."""

GATE_SIGMA: float = C_CONSTANT * C_CONSTANT
"""Gate sharpness scale: C²."""

# =============================================================================
# DERIVED CONSTANTS
# =============================================================================

ELECTRON_MASS: float = 9.10938371e-31
"""Electron rest mass [kg] (CODATA 2018)."""

SQRT_2: float = math.sqrt(2.0)
"""√2 ≈ 1.41421356, precomputed for impedance matching."""

PHI: float = (1.0 + math.sqrt(5.0)) / 2.0
"""Golden ratio φ = (1+√5)/2 ≈ 1.618033988749895."""

RHO_INFO: float = PHI
"""Information stacking ratio (alias for PHI)."""

# =============================================================================
# CAUSAL SPHERE CONSTANTS
# =============================================================================

SQRT2_C: float = SQRT_2 * C_CONSTANT
"""Closure threshold for phase tracking: √2 × C ≈ 0.200."""

COGNITIVE_TURN: float = math.pi
"""Target phase for closure detection (half-turn)."""

KAPPA: float = 0.3
"""Linear coupling coefficient in K_M fixed-point operator."""

ETA: float = math.sqrt(1.0 - KAPPA**2) - KAPPA
"""Nonlinear coupling coefficient: η = √(1−κ²) − κ ≈ 0.654."""

L_CONTRACTION: float = KAPPA + ETA
"""Contraction rate for K_M operator (Lipschitz constant < 1)."""

DEFAULT_HISTORY_DEPTH: int = 16
"""Default ring buffer depth for closure history."""

FIXEDPOINT_TOL: float = 1e-7
"""Convergence tolerance for fixed-point iteration."""

FIXEDPOINT_MAX_ITER: int = 50
"""Maximum iterations for fixed-point computation."""

# =============================================================================
# DERIVED PHYSICS FUNCTIONS
# =============================================================================


def compute_damping(omega: float | np.ndarray) -> float | np.ndarray:
    """Compute TSR-derived damping: ζ = √(D'/ℏω).

    Accepts scalar or numpy array (broadcasting).

    Parameters
    ----------
    omega : float or array_like
        Natural frequency [rad/s].

    Returns
    -------
    float or ndarray
        Damping ratio (dimensionless).
    """
    return np.sqrt(D_PRIME / (HBAR * np.asarray(omega, dtype=np.float64)))


def compute_coupling(
    n: float | np.ndarray,
    d: float | np.ndarray,
    lam: float,
    K0: float = 1.0,
) -> float | np.ndarray:
    """Compute TSR impedance-matched coupling: K = K₀·√2·Cⁿ·exp(−d/λ).

    Parameters
    ----------
    n : float or array_like
        Coupling order (frequency hierarchy level).
    d : float or array_like
        Spatial distance [m].
    lam : float
        Characteristic decay length [m].
    K0 : float
        Base coupling strength (default 1.0).

    Returns
    -------
    float or ndarray
    """
    n = np.asarray(n, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    return K0 * SQRT_2 * (C_CONSTANT**n) * np.exp(-d / lam)


def impedance_matching_exponent(omega_i: float, omega_j: float) -> float:
    """Coupling order n between two frequencies.

    n = log(max(ωᵢ,ωⱼ)/min(ωᵢ,ωⱼ)) / log(1/C).
    """
    omega_max = max(omega_i, omega_j)
    omega_min = min(omega_i, omega_j)
    return math.log(omega_max / omega_min) / math.log(1.0 / C_CONSTANT)


def boltzmann_weight(E: float | np.ndarray) -> float | np.ndarray:
    """Gibbs-Boltzmann weight: w(E) = exp(−E/D').

    Accepts scalar or numpy array.
    """
    return np.exp(-np.asarray(E, dtype=np.float64) / D_PRIME)


def effective_temperature() -> float:
    """Effective temperature T_eff = D'/k_B  ≈ 1.93×10⁷ K."""
    return D_PRIME / K_BOLTZMANN


def K_M(x: float | np.ndarray) -> float | np.ndarray:
    """K_M fixed-point operator: K_M(x) = κx + η·sin(x).

    Contraction map with Lipschitz constant L = κ+η ≈ 0.954 < 1.
    Accepts scalar or numpy array.
    """
    arr = np.asarray(x, dtype=np.float64)
    out = KAPPA * arr + ETA * np.sin(arr)
    if np.ndim(out) == 0:
        return float(out)
    return np.asarray(out, dtype=np.float64)


# =============================================================================
# VALIDATION
# =============================================================================


def validate_tsr_constants() -> bool:
    """Validate that TSR constants satisfy known relationships.

    Checks:
        1. C ∈ (0, 1)
        2. D' > 0 and finite
        3. ℏ matches CODATA (relative error < 1e-10)
        4. α matches experiment (relative error < 1e-9)
        5. φ² = φ + 1 (golden ratio)
        6. L_CONTRACTION < 1
        7. SQRT2_C ≈ √2 × C
        8. κ + η = L_CONTRACTION

    Returns True if all pass, False otherwise (with warnings).
    """
    valid = True

    if not (0 < C_CONSTANT < 1):
        warnings.warn(f"C_CONSTANT out of range: {C_CONSTANT}", stacklevel=2)
        valid = False

    if not (D_PRIME > 0 and math.isfinite(D_PRIME)):
        warnings.warn(f"D_PRIME invalid: {D_PRIME}", stacklevel=2)
        valid = False

    hbar_codata = 1.054571817e-34
    if abs(HBAR - hbar_codata) / hbar_codata > 1e-10:
        warnings.warn(f"HBAR deviates from CODATA: {HBAR}", stacklevel=2)
        valid = False

    alpha_exp = 1.0 / 137.035999084
    if abs(ALPHA - alpha_exp) / alpha_exp > 1e-9:
        warnings.warn(f"ALPHA deviates from experimental: {ALPHA}", stacklevel=2)
        valid = False

    if abs(PHI**2 - PHI - 1) > 1e-14:
        warnings.warn(f"PHI does not satisfy golden ratio property: {PHI}", stacklevel=2)
        valid = False

    if not (L_CONTRACTION < 1):
        warnings.warn(f"L_CONTRACTION >= 1: {L_CONTRACTION}", stacklevel=2)
        valid = False

    sqrt2_c_expected = math.sqrt(2.0) * C_CONSTANT
    if abs(SQRT2_C - sqrt2_c_expected) / sqrt2_c_expected > 1e-14:
        warnings.warn(
            f"SQRT2_C derivation mismatch: {SQRT2_C} vs {sqrt2_c_expected}",
            stacklevel=2,
        )
        valid = False

    if abs(KAPPA + ETA - L_CONTRACTION) > 1e-14:
        warnings.warn(
            f"KAPPA+ETA != L_CONTRACTION: {KAPPA}+{ETA} vs {L_CONTRACTION}",
            stacklevel=2,
        )
        valid = False

    return valid
