"""Deterministic oscillator-constrained normalization for reconstructive mode.

Phase C provides a canonical projection from frequency tokens into solver-ready
oscillator state while preserving reproducibility across platforms.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import blake3

from braidcodec._exceptions import FormatError
from braidcodec.algebra.tsr_constants import OMEGA_ANCHOR, OMEGA_MAX, OMEGA_MIN

if TYPE_CHECKING:
    from collections.abc import Sequence

    from braidcodec.codec.tokenizer_frequency import FrequencyToken

NORMALIZATION_PROFILE_ID = "osc-norm-v1"

# Explicit constants for a reproducible profile.
_AMP_MIN = 0.25
_AMP_MAX = 1.0
_PHASE_MIN = -math.pi
_PHASE_MAX = math.pi
_SCALE_MIN = 0.1
_SCALE_MAX = 2.5


@dataclass(frozen=True, slots=True)
class OscillatorPoint:
    """One normalized oscillator sample derived from a frequency token."""

    omega: float
    amplitude: float
    phase: float
    scale: float


@dataclass(frozen=True, slots=True)
class OscillatorNormalizationResult:
    """Normalization output plus contract metadata."""

    points: tuple[OscillatorPoint, ...]
    metadata: dict[str, str]


def _profile_payload() -> dict[str, float | str]:
    return {
        "profile_id": NORMALIZATION_PROFILE_ID,
        "omega_min": OMEGA_MIN,
        "omega_max": OMEGA_MAX,
        "omega_anchor": OMEGA_ANCHOR,
        "amp_min": _AMP_MIN,
        "amp_max": _AMP_MAX,
        "phase_min": _PHASE_MIN,
        "phase_max": _PHASE_MAX,
        "scale_min": _SCALE_MIN,
        "scale_max": _SCALE_MAX,
        "mapping": "log-hz-to-omega-bounded",
    }


def normalization_profile_hash() -> str:
    """Stable hash for oscillator normalization constants and mapping."""
    blob = json.dumps(_profile_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return blake3.blake3(blob).hexdigest()


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _phase_from_token(token_class: str, value: str) -> float:
    payload = f"{token_class}|{value}".encode()
    digest = blake3.blake3(payload).digest(length=8)
    u = int.from_bytes(digest, byteorder="big", signed=False) / ((1 << 64) - 1)
    phase = _PHASE_MIN + u * (_PHASE_MAX - _PHASE_MIN)
    return _clamp(phase, _PHASE_MIN, _PHASE_MAX)


class OscillatorNormalizationV1:
    """Canonical normalization from frequency tokens to oscillator points."""

    profile_id = NORMALIZATION_PROFILE_ID

    def normalize_tokens(self, tokens: Sequence[FrequencyToken]) -> OscillatorNormalizationResult:
        if not tokens:
            raise FormatError("Oscillator normalization requires at least one token")

        # Stable log-frequency projection into [OMEGA_MIN, OMEGA_MAX].
        hz_values = [max(float(t.frequency_hz), 1e-9) for t in tokens]
        log_lo = math.log(min(hz_values))
        log_hi = math.log(max(hz_values))
        denom = max(log_hi - log_lo, 1e-12)

        points: list[OscillatorPoint] = []
        for token in tokens:
            hz = max(float(token.frequency_hz), 1e-9)
            u = (math.log(hz) - log_lo) / denom
            u = _clamp(u, 0.0, 1.0)

            omega = OMEGA_MIN + u * (OMEGA_MAX - OMEGA_MIN)
            omega = _clamp(omega, OMEGA_MIN, OMEGA_MAX)

            # Bin position drives amplitude deterministically.
            bin_norm = _clamp(float(token.bin_index) / 127.0, 0.0, 1.0)
            amplitude = _AMP_MAX - (_AMP_MAX - _AMP_MIN) * bin_norm
            amplitude = _clamp(amplitude, _AMP_MIN, _AMP_MAX)

            phase = _phase_from_token(token.token_class, token.value)

            # Anchor-scaled factor keeps solver inputs in a bounded regime.
            scale = amplitude * math.sqrt(OMEGA_ANCHOR / max(omega, 1e-12))
            scale = _clamp(scale, _SCALE_MIN, _SCALE_MAX)

            points.append(
                OscillatorPoint(
                    omega=omega,
                    amplitude=amplitude,
                    phase=phase,
                    scale=scale,
                )
            )

        profile_hash = normalization_profile_hash()
        metadata = {
            "normalization_profile_id": NORMALIZATION_PROFILE_ID,
            "normalization_profile_hash": profile_hash,
            "omega_min": f"{OMEGA_MIN:.12g}",
            "omega_max": f"{OMEGA_MAX:.12g}",
            "omega_anchor": f"{OMEGA_ANCHOR:.12g}",
        }

        return OscillatorNormalizationResult(points=tuple(points), metadata=metadata)
