"""Tests for K_M fixed-point utilities (Phase E foundation)."""

from __future__ import annotations

import numpy as np
import pytest

from braidcodec._exceptions import ContractionError
from braidcodec.algebra.fixedpoint import (
    contraction_rate,
    ensure_contraction,
    iterate_fixedpoint,
    km_step,
    verify_fixedpoint,
)
from braidcodec.algebra.tsr_constants import FIXEDPOINT_MAX_ITER, L_CONTRACTION


def test_contraction_rate_matches_constant() -> None:
    assert abs(contraction_rate() - L_CONTRACTION) < 1e-14


def test_ensure_contraction_rejects_invalid_params() -> None:
    with pytest.raises(ContractionError):
        ensure_contraction(kappa=0.7, eta=0.4)


def test_km_step_vector_shape() -> None:
    x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    y = km_step(x)
    assert y.shape == x.shape


def test_iterate_fixedpoint_converges() -> None:
    # Use near-attractor seeds; with L≈0.954 and max_iter=50,
    # far initial values are not expected to hit 1e-7 tolerance.
    x0 = np.array([0.0, 1e-4, -2e-4, 3e-4], dtype=np.float64)
    tol = 1e-5
    result = iterate_fixedpoint(x0, tol=tol, max_iter=FIXEDPOINT_MAX_ITER)

    assert result.values.shape == x0.shape
    assert result.converged.shape == x0.shape
    assert np.all(result.converged)
    assert result.diagnostics.converged_ratio == 1.0
    assert result.diagnostics.iterations_mean > 0.0
    assert result.diagnostics.residual_max < tol


def test_verify_fixedpoint_mask() -> None:
    x0 = np.array([1e-4, -2e-4, 3e-4], dtype=np.float64)
    tol = 1e-5
    solved = iterate_fixedpoint(x0, tol=tol)
    mask = verify_fixedpoint(solved.values, tol=tol)
    assert mask.dtype == np.bool_
    assert np.all(mask)
