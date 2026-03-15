"""Reconstructive K_M solver payload validation utilities.

This module validates reconstructive solver contracts emitted in metadata payloads.
"""

from __future__ import annotations

from braidcodec._exceptions import FormatError
from braidcodec.algebra.fixedpoint import ensure_contraction, iterate_fixedpoint


def _parse_float(payload: dict[str, str], key: str) -> float:
    try:
        return float(payload[key])
    except (KeyError, ValueError) as exc:
        raise FormatError(f"Invalid reconstructive payload float '{key}'") from exc


def _parse_int(payload: dict[str, str], key: str) -> int:
    try:
        return int(payload[key])
    except (KeyError, ValueError) as exc:
        raise FormatError(f"Invalid reconstructive payload int '{key}'") from exc


def _parse_seed_vector(payload: dict[str, str]) -> list[float]:
    raw = payload.get("km_seed_vector", "")
    if not raw:
        raise FormatError("Missing reconstructive payload key: km_seed_vector")

    parts = [p for p in raw.split(",") if p]
    if not parts:
        raise FormatError("km_seed_vector must contain at least one coordinate")

    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise FormatError("km_seed_vector contains non-numeric values") from exc


def validate_reconstructive_solver_payload(payload: dict[str, str]) -> None:
    """Validate K_M contraction/convergence gates from reconstructive payload.

    Enforced gates:
    - contraction: kappa + eta < 1
    - convergence: converged_ratio >= threshold
    - residual: residual_max <= threshold
    """
    seed = _parse_seed_vector(payload)
    kappa = _parse_float(payload, "km_kappa")
    eta = _parse_float(payload, "km_eta")
    tol = _parse_float(payload, "km_tol")
    max_iter = _parse_int(payload, "km_max_iter")

    residual_max_threshold = _parse_float(payload, "km_threshold_residual_max")
    valid_ratio_threshold = _parse_float(payload, "km_threshold_valid_ratio")

    if max_iter < 1:
        raise FormatError("km_max_iter must be >= 1")
    if not (0.0 <= valid_ratio_threshold <= 1.0):
        raise FormatError("km_threshold_valid_ratio must be within [0,1]")
    if residual_max_threshold < 0.0 or tol <= 0.0:
        raise FormatError("K_M thresholds/tolerance must be positive")

    try:
        ensure_contraction(kappa=kappa, eta=eta)
    except Exception as exc:
        raise FormatError("K_M contraction gate failed") from exc

    result = iterate_fixedpoint(seed, kappa=kappa, eta=eta, tol=tol, max_iter=max_iter)

    if result.diagnostics.converged_ratio < valid_ratio_threshold:
        raise FormatError(
            "K_M convergence gate failed",
            converged_ratio=result.diagnostics.converged_ratio,
            required_ratio=valid_ratio_threshold,
        )
    if result.diagnostics.residual_max > residual_max_threshold:
        raise FormatError(
            "K_M residual gate failed",
            residual_max=result.diagnostics.residual_max,
            threshold=residual_max_threshold,
        )
