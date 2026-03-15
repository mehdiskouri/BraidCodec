"""Tests for reconstructive compact fitting helpers."""

from __future__ import annotations

import json

from braidcodec.codec.reconstructive_compact import (
    _build_spectral_predictor,
    fit_reconstructive_program,
)


def test_build_spectral_predictor_changes_with_nnz() -> None:
    p_low, params_low = _build_spectral_predictor(
        128,
        coupling_density=0.25,
        coupling_spectral_radius=120.0,
        coupling_nnz=16,
        morton_key=0,
    )
    p_high, params_high = _build_spectral_predictor(
        128,
        coupling_density=0.25,
        coupling_spectral_radius=120.0,
        coupling_nnz=32000,
        morton_key=0,
    )

    assert p_low != p_high
    assert params_low != params_high


def test_build_spectral_predictor_changes_with_morton_index() -> None:
    p0, params0 = _build_spectral_predictor(
        128,
        coupling_density=0.25,
        coupling_spectral_radius=120.0,
        coupling_nnz=1024,
        morton_key=0,
    )
    p1, params1 = _build_spectral_predictor(
        128,
        coupling_density=0.25,
        coupling_spectral_radius=120.0,
        coupling_nnz=1024,
        morton_key=17,
    )

    assert p0 != p1
    assert params0 != params1


def test_fit_reconstructive_program_accepts_nnz_signal() -> None:
    text = "A compact fitter path with explicit nnz coupling signal."
    fitted = fit_reconstructive_program(
        text,
        domain_kind="text",
        coupling_density=0.3,
        coupling_spectral_radius=250.0,
        coupling_nnz=32000,
    )
    assert fitted["reconstructive_program_type"] in {
        "latent-residual-v2",
        "latent-residual-v3",
    }
    payload = json.loads(fitted["reconstructive_program_payload"])
    assert isinstance(payload, dict)


def test_fit_reconstructive_program_emits_base85_residual_payload() -> None:
    text = "Non-periodic text that should use latent residual storage path." * 4
    fitted = fit_reconstructive_program(
        text,
        domain_kind="text",
        coupling_density=0.4,
        coupling_spectral_radius=300.0,
        coupling_nnz=1024,
    )
    payload = json.loads(fitted["reconstructive_program_payload"])

    if fitted["reconstructive_program_type"] == "latent-residual-v2":
        residual = payload.get("residual_b85", payload.get("r85"))
        assert isinstance(residual, str)
        assert residual
    elif fitted["reconstructive_program_type"] == "latent-residual-v3":
        segments = payload.get("segments", payload.get("s", []))
        assert isinstance(segments, list)
        assert segments
        residual = segments[0].get("residual_b85", segments[0].get("r85"))
        assert isinstance(residual, str)
        assert residual
