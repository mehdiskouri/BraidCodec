"""Deterministic equation discovery helpers for reconstructive mode.

This module uses a profile-pinned scientific stack (PySINDy, scikit-learn,
SciPy) to assist candidate search, while preserving strict deterministic and
exact replay contracts for accepted discovered programs.
"""

from __future__ import annotations

import importlib
import json
import zlib
from base64 import b85encode
from dataclasses import dataclass
from typing import Any, Literal, cast

import blake3
import numpy as np
import sympy as sp

from braidcodec.codec.braid_program_codec import compile_discovered_braid_program

DomainKind = Literal["text", "json", "logs"]


@dataclass(frozen=True, slots=True)
class DiscoveredEquation:
    """A single deterministic governing equation candidate."""

    equation_family: str
    symbolic_form: str
    symbolic_hash: str
    coefficients: tuple[int, ...]
    initial_state: tuple[int, ...]
    rollout_length: int


@dataclass(frozen=True, slots=True)
class DiscoveryDiagnostics:
    """Compact diagnostics emitted alongside discovery outputs."""

    method: str
    profile_id: str
    backend_chain: str
    candidate_count: int
    selected_family: str
    exact_replay: bool


@dataclass(frozen=True, slots=True)
class DiscoveredEquationSet:
    """Discovery result bundle."""

    equation: DiscoveredEquation | None
    diagnostics: DiscoveryDiagnostics


@dataclass(frozen=True, slots=True)
class DiscoveryProfile:
    """Profile-pinned discovery hyperparameters."""

    profile_id: str
    sindy_threshold: float
    lasso_alpha: float
    affine_radius: int


_DEFAULT_PROFILE = DiscoveryProfile(
    profile_id="default-v1",
    sindy_threshold=1e-6,
    lasso_alpha=1e-6,
    affine_radius=8,
)

_PACK_PREFIX = "~mp85:"


def _resolve_profile(profile_id: str) -> DiscoveryProfile:
    normalized = profile_id.strip().lower()
    if normalized in {"", "default", "default-v1", "strict-v1"}:
        # strict-v1 keeps the same discovery search; strictness is handled by K_M gates.
        return _DEFAULT_PROFILE
    return _DEFAULT_PROFILE


def _symbolic_hash(symbolic_form: str, coefficients: tuple[int, ...], initial: tuple[int, ...]) -> str:
    payload = f"{symbolic_form}|{','.join(str(v) for v in coefficients)}|{','.join(str(v) for v in initial)}"
    return blake3.blake3(payload.encode("utf-8")).hexdigest()


def _discover_constant(values: list[int]) -> DiscoveredEquation | None:
    if not values:
        return None
    first = values[0]
    if any(v != first for v in values):
        return None

    n = sp.Symbol("n", integer=True, nonnegative=True)
    c = sp.Integer(first)
    symbolic = str(sp.Eq(sp.Function("x")(n), c))
    return DiscoveredEquation(
        equation_family="byte-constant-v1",
        symbolic_form=symbolic,
        symbolic_hash=_symbolic_hash(symbolic, (first,), (first,)),
        coefficients=(first,),
        initial_state=(first,),
        rollout_length=len(values),
    )


def _discover_linear_mod(values: list[int]) -> DiscoveredEquation | None:
    if len(values) < 2:
        return None

    step = (values[1] - values[0]) % 256
    for i in range(2, len(values)):
        if (values[i] - values[i - 1]) % 256 != step:
            return None

    n = sp.Symbol("n", integer=True, nonnegative=True)
    x0 = sp.Integer(values[0])
    d = sp.Integer(step)
    symbolic = str(sp.Eq(sp.Function("x")(n), sp.Mod(x0 + d * n, 256)))
    return DiscoveredEquation(
        equation_family="byte-linear-mod-v1",
        symbolic_form=symbolic,
        symbolic_hash=_symbolic_hash(symbolic, (step,), (values[0],)),
        coefficients=(step,),
        initial_state=(values[0],),
        rollout_length=len(values),
    )


def _discover_xor_step(values: list[int]) -> DiscoveredEquation | None:
    if len(values) < 2:
        return None

    step = values[0] ^ values[1]
    for i in range(1, len(values)):
        if (values[i - 1] ^ step) != values[i]:
            return None

    n = sp.Symbol("n", integer=True, nonnegative=True)
    k = sp.Integer(step)
    symbolic = str(sp.Eq(sp.Function("x")(n + 1), sp.Xor(sp.Function("x")(n), k)))
    return DiscoveredEquation(
        equation_family="byte-xor-step-v1",
        symbolic_form=symbolic,
        symbolic_hash=_symbolic_hash(symbolic, (step,), (values[0],)),
        coefficients=(step,),
        initial_state=(values[0],),
        rollout_length=len(values),
    )


def _discover_affine_mod(values: list[int]) -> DiscoveredEquation | None:
    if len(values) < 3:
        return None

    x0 = values[0]
    x1 = values[1]

    found: tuple[int, int] | None = None
    for a in range(256):
        b = (x1 - (a * x0)) % 256
        ok = True
        prev = x1
        for i in range(2, len(values)):
            next_val = (a * prev + b) % 256
            if next_val != values[i]:
                ok = False
                break
            prev = next_val
        if ok:
            found = (a, b)
            break

    if found is None:
        return None

    a, b = found
    symbolic = f"x[n+1]=({a}*x[n]+{b}) mod 256"
    return DiscoveredEquation(
        equation_family="byte-affine-recursion-v1",
        symbolic_form=symbolic,
        symbolic_hash=_symbolic_hash(symbolic, (a, b), (x0,)),
        coefficients=(a, b),
        initial_state=(x0,),
        rollout_length=len(values),
    )


def _affine_exact_match(values: list[int], a: int, b: int) -> bool:
    if not values:
        return False
    prev = values[0]
    for i in range(1, len(values)):
        nxt = (a * prev + b) % 256
        if nxt != values[i]:
            return False
        prev = nxt
    return True


def _estimate_affine_candidates(values: list[int], profile: DiscoveryProfile) -> list[tuple[int, int]]:
    """Estimate affine map candidates using sklearn, scipy, and PySINDy.

    Returned candidates are ordered deterministically for exact verification.
    """
    if len(values) < 3:
        return []

    x = np.asarray(values[:-1], dtype=np.float64).reshape(-1, 1)
    y = np.asarray(values[1:], dtype=np.float64)

    sklearn_preprocessing = importlib.import_module("sklearn.preprocessing")
    sklearn_linear_model = importlib.import_module("sklearn.linear_model")
    scipy_optimize = importlib.import_module("scipy.optimize")

    scaler = sklearn_preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)

    linear = sklearn_linear_model.LinearRegression()
    linear.fit(x_scaled, y)
    w_scaled = float(linear.coef_[0])
    b_scaled = float(linear.intercept_)
    scale = cast("np.ndarray[Any, np.dtype[np.float64]]", scaler.scale_)
    mean = cast("np.ndarray[Any, np.dtype[np.float64]]", scaler.mean_)
    w = w_scaled / float(scale[0])
    b = b_scaled - w * float(mean[0])

    def _mod_residual(params: np.ndarray) -> np.ndarray:
        aa = float(params[0])
        bb = float(params[1])
        pred = np.mod(aa * x[:, 0] + bb, 256.0)
        err = np.mod(pred - y + 128.0, 256.0) - 128.0
        return err

    least_squares = scipy_optimize.least_squares
    refined = least_squares(
        _mod_residual,
        x0=np.asarray([w, b], dtype=np.float64),
        method="trf",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=500,
    )

    candidates: list[tuple[int, int]] = []

    def _append_neighborhood(center_a: int, center_b: int) -> None:
        radius = profile.affine_radius
        for da in range(-radius, radius + 1):
            for db in range(-radius, radius + 1):
                aa = (center_a + da) % 256
                bb = (center_b + db) % 256
                candidates.append((aa, bb))

    refined_x = np.asarray(getattr(refined, "x", np.asarray([w, b], dtype=np.float64)))
    _append_neighborhood(int(round(float(refined_x[0]))) % 256, int(round(float(refined_x[1]))) % 256)

    # PySINDy-assisted discrete map estimate (best-effort; deterministic settings).
    try:
        ps_mod = importlib.import_module("pysindy")
        sindy_x = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        feature_lib = ps_mod.PolynomialLibrary(
            degree=1,
            include_interaction=False,
            include_bias=True,
        )
        optimizer = ps_mod.STLSQ(threshold=profile.sindy_threshold, alpha=profile.lasso_alpha)
        model = ps_mod.SINDy(feature_library=feature_lib, optimizer=optimizer)
        model.fit(sindy_x, t=1, feature_names=["x"])
        coef = np.asarray(model.coefficients(), dtype=np.float64)
        if coef.ndim == 2 and coef.shape[0] >= 1 and coef.shape[1] >= 2:
            # Discrete map with degree-1 polynomial library -> [1, x] coefficients.
            c0 = float(coef[0, 0])
            c1 = float(coef[0, 1])
            _append_neighborhood(int(round(c1)) % 256, int(round(c0)) % 256)
    except Exception:
        # Keep deterministic fallback behavior even if PySINDy fit fails.
        pass

    # Deterministic de-dup + ordering.
    unique = sorted(set(candidates), key=lambda t: (t[0], t[1]))
    return unique


def _discover_affine_mod_scientific(
    values: list[int],
    profile: DiscoveryProfile,
) -> DiscoveredEquation | None:
    if len(values) < 3:
        return None

    for a, b in _estimate_affine_candidates(values, profile):
        if _affine_exact_match(values, a, b):
            symbolic = f"x[n+1]=({a}*x[n]+{b}) mod 256"
            return DiscoveredEquation(
                equation_family="byte-affine-recursion-v1",
                symbolic_form=symbolic,
                symbolic_hash=_symbolic_hash(symbolic, (a, b), (values[0],)),
                coefficients=(a, b),
                initial_state=(values[0],),
                rollout_length=len(values),
            )

    # Deterministic exhaustive fallback to preserve completeness.
    return _discover_affine_mod(values)


def compile_discovered_equation_program(equation: DiscoveredEquation) -> dict[str, str]:
    """Compile equation model to reconstructive compact program metadata fields."""
    payload: dict[str, Any] = {
        "equation_family": equation.equation_family,
        "symbolic_form": equation.symbolic_form,
        "symbolic_hash": equation.symbolic_hash,
        "coefficients": list(equation.coefficients),
        "initial_state": list(equation.initial_state),
        "rollout_length": equation.rollout_length,
        "projection": "identity-bytes-v1",
    }
    import json

    return {
        "reconstructive_program_type": "discovered-equation-v1",
        "reconstructive_program_payload": json.dumps(payload, sort_keys=True, separators=(",", ":")),
    }


def compile_discovered_braid_equation_program(equation: DiscoveredEquation) -> dict[str, str]:
    """Compile equation model into braid-backed reconstructive metadata fields."""
    payload = compile_discovered_braid_program(
        equation_family=equation.equation_family,
        coefficients=equation.coefficients,
        initial_state=equation.initial_state,
        rollout_length=equation.rollout_length,
        symbolic_hash=equation.symbolic_hash,
        symbolic_form=equation.symbolic_form,
    )

    json_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    msgpack_mod = importlib.import_module("msgpack")
    packed_obj = msgpack_mod.packb(payload, use_bin_type=True)
    packed = bytes(packed_obj)
    packed_payload = _PACK_PREFIX + b85encode(zlib.compress(packed, level=9)).decode("ascii")

    return {
        "reconstructive_program_type": "discovered-braid-equation-v1",
        "reconstructive_program_payload": packed_payload if len(packed_payload) < len(json_payload) else json_payload,
    }


def discover_equations(
    source_bytes: bytes,
    *,
    domain_kind: DomainKind,
    profile_id: str,
) -> DiscoveredEquationSet:
    """Discover deterministic equations for a byte sequence.

    Parameters are intentionally narrow to keep behavior reproducible.
    """
    _ = domain_kind
    profile = _resolve_profile(profile_id)

    values = [int(b) for b in source_bytes]
    candidates = [
        _discover_constant(values),
        _discover_linear_mod(values),
        _discover_xor_step(values),
        _discover_affine_mod_scientific(values, profile),
    ]
    filtered = [c for c in candidates if c is not None]
    selected = filtered[0] if filtered else None

    diagnostics = DiscoveryDiagnostics(
        method="scientific-assisted-byte-family-v2",
        profile_id=profile.profile_id,
        backend_chain="pysindy+sklearn+scipy+deterministic-fallback",
        candidate_count=len(filtered),
        selected_family=selected.equation_family if selected is not None else "none",
        exact_replay=selected is not None,
    )
    return DiscoveredEquationSet(equation=selected, diagnostics=diagnostics)
