"""Tests for deterministic oscillator normalization (Phase C)."""

from __future__ import annotations

from braidcodec.codec.oscillator_normalization import (
    OscillatorNormalizationV1,
    normalization_profile_hash,
)
from braidcodec.codec.tokenizer_frequency import FrequencyTokenizerV1


def test_normalization_is_deterministic() -> None:
    tokenizer = FrequencyTokenizerV1()
    normalizer = OscillatorNormalizationV1()

    tokens = tokenizer.tokenize("alpha beta gamma", domain="text").tokens
    out_a = normalizer.normalize_tokens(tokens)
    out_b = normalizer.normalize_tokens(tokens)

    assert out_a.points == out_b.points
    assert out_a.metadata == out_b.metadata


def test_profile_hash_stable() -> None:
    h1 = normalization_profile_hash()
    h2 = normalization_profile_hash()
    assert h1 == h2


def test_normalized_ranges_are_bounded() -> None:
    tokenizer = FrequencyTokenizerV1()
    normalizer = OscillatorNormalizationV1()

    tokens = tokenizer.tokenize('{"x":1,"y":2}', domain="json").tokens
    out = normalizer.normalize_tokens(tokens)

    assert len(out.points) > 0
    for p in out.points:
        assert p.amplitude >= 0.25
        assert p.amplitude <= 1.0
        assert p.phase >= -3.141592653589793
        assert p.phase <= 3.141592653589793
        assert p.scale >= 0.1
        assert p.scale <= 2.5


def test_metadata_contains_profile_hash() -> None:
    tokenizer = FrequencyTokenizerV1()
    normalizer = OscillatorNormalizationV1()

    tokens = tokenizer.tokenize("2026-03-15T12:00:00Z INFO ok", domain="logs").tokens
    out = normalizer.normalize_tokens(tokens)
    assert out.metadata["normalization_profile_id"] == "osc-norm-v1"
    assert out.metadata["normalization_profile_hash"] == normalization_profile_hash()
