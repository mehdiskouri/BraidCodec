"""Deterministic K_M fixed-point iteration utilities.

Phase E foundation: Julia-aligned contraction checks and convergence tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from braidcodec._exceptions import ContractionError
from braidcodec.algebra.tsr_constants import (
    ETA,
    FIXEDPOINT_MAX_ITER,
    FIXEDPOINT_TOL,
    KAPPA,
    L_CONTRACTION,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class FixedPointDiagnostics:
    """Aggregate diagnostics for a fixed-point solve."""

    residual_max: float
    residual_mean: float
    iterations_mean: float
    converged_ratio: float


@dataclass(frozen=True, slots=True)
class FixedPointResult:
    """Result payload for vectorized K_M fixed-point iteration."""

    values: np.ndarray
    residuals: np.ndarray
    iteration_counts: np.ndarray
    converged: np.ndarray
    diagnostics: FixedPointDiagnostics


def km_step(values: np.ndarray, *, kappa: float = KAPPA, eta: float = ETA) -> np.ndarray:
    """Apply one K_M update: kappa*x + eta*sin(x)."""
    arr = np.asarray(values, dtype=np.float64)
    return kappa * arr + eta * np.sin(arr)


def ensure_contraction(*, kappa: float = KAPPA, eta: float = ETA) -> None:
    """Raise if K_M parameters violate contraction condition."""
    if (kappa + eta) >= 1.0:
        raise ContractionError(
            "K_M contraction condition violated: kappa + eta must be < 1",
            kappa=kappa,
            eta=eta,
            lipschitz=(kappa + eta),
        )


def iterate_fixedpoint(
    values: Iterable[float] | np.ndarray,
    *,
    tol: float = FIXEDPOINT_TOL,
    max_iter: int = FIXEDPOINT_MAX_ITER,
    kappa: float = KAPPA,
    eta: float = ETA,
) -> FixedPointResult:
    """Run deterministic vectorized fixed-point iteration for K_M."""
    ensure_contraction(kappa=kappa, eta=eta)

    raw_values = list(values) if not isinstance(values, np.ndarray) else values
    current = np.asarray(raw_values, dtype=np.float64)
    if current.ndim != 1:
        raise ValueError("iterate_fixedpoint expects a 1D vector")

    n = current.size
    residuals = np.full(n, np.inf, dtype=np.float64)
    iteration_counts = np.zeros(n, dtype=np.int32)
    converged = np.zeros(n, dtype=bool)

    for _ in range(int(max_iter)):
        nxt = km_step(current, kappa=kappa, eta=eta)
        residuals = np.abs(nxt - current)

        newly_converged = residuals < float(tol)
        iteration_counts += (~converged).astype(np.int32)
        converged |= newly_converged

        current = nxt
        if np.all(converged):
            break

    diagnostics = FixedPointDiagnostics(
        residual_max=float(np.max(residuals) if n else 0.0),
        residual_mean=float(np.mean(residuals) if n else 0.0),
        iterations_mean=float(np.mean(iteration_counts) if n else 0.0),
        converged_ratio=float(np.mean(converged) if n else 1.0),
    )

    return FixedPointResult(
        values=current,
        residuals=residuals,
        iteration_counts=iteration_counts,
        converged=converged,
        diagnostics=diagnostics,
    )


def verify_fixedpoint(
    values: Iterable[float] | np.ndarray,
    *,
    tol: float = FIXEDPOINT_TOL,
    kappa: float = KAPPA,
    eta: float = ETA,
) -> np.ndarray:
    """Return boolean mask for |K_M(x)-x| < tol on each coordinate."""
    raw_values = list(values) if not isinstance(values, np.ndarray) else values
    arr = np.asarray(raw_values, dtype=np.float64)
    residuals = np.abs(km_step(arr, kappa=kappa, eta=eta) - arr)
    return residuals < float(tol)


def contraction_rate(*, kappa: float = KAPPA, eta: float = ETA) -> float:
    """Return Lipschitz contraction rate L = kappa + eta."""
    return float(kappa + eta)


__all__ = [
    "L_CONTRACTION",
    "FixedPointDiagnostics",
    "FixedPointResult",
    "contraction_rate",
    "ensure_contraction",
    "iterate_fixedpoint",
    "km_step",
    "verify_fixedpoint",
]
