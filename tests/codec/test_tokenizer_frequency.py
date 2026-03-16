"""Tests for Phase A/B frequency tokenizer contracts."""

from __future__ import annotations

import pytest

from braidcodec._exceptions import FormatError
from braidcodec.codec.schema import validate_reconstructive_metadata
from braidcodec.codec.tokenizer_frequency import FrequencyTokenizerV1, reconstructive_metadata_keys


def test_text_canonicalization_is_deterministic() -> None:
    tokenizer = FrequencyTokenizerV1()
    # "Cafe" with composed and decomposed e-acute + different newline forms.
    a = "Cafe\u0301\r\nLine2"
    b = "Caf\u00e9\nLine2"

    out_a = tokenizer.tokenize(a, domain="text")
    out_b = tokenizer.tokenize(b, domain="text")

    assert out_a.canonical_text == out_b.canonical_text
    assert [t.value for t in out_a.tokens] == [t.value for t in out_b.tokens]
    assert out_a.metadata["vocab_hash"] == out_b.metadata["vocab_hash"]


def test_json_canonicalization_stable_key_order() -> None:
    tokenizer = FrequencyTokenizerV1()

    out_a = tokenizer.tokenize('{"b":2,"a":1}', domain="json")
    out_b = tokenizer.tokenize('{"a":1,"b":2}', domain="json")

    assert out_a.canonical_text == '{"a":1,"b":2}'
    assert out_a.canonical_text == out_b.canonical_text
    assert out_a.metadata["vocab_hash"] == out_b.metadata["vocab_hash"]


def test_logs_tokenization_produces_expected_classes() -> None:
    tokenizer = FrequencyTokenizerV1()
    out = tokenizer.tokenize("2026-03-15T12:34:56Z INFO core started", domain="logs")

    classes = {t.token_class for t in out.tokens}
    assert "log-timestamp" in classes
    assert "log-ident" in classes


def test_frequency_bins_within_range() -> None:
    tokenizer = FrequencyTokenizerV1()
    out = tokenizer.tokenize("alpha beta gamma", domain="text")
    assert len(out.tokens) > 0
    for token in out.tokens:
        assert 0 <= token.bin_index < tokenizer.bin_count
        assert token.frequency_hz > 0.0


def test_metadata_contains_required_keys() -> None:
    tokenizer = FrequencyTokenizerV1()
    out = tokenizer.tokenize("hello", domain="text")
    required = reconstructive_metadata_keys()
    assert required.issubset(out.metadata.keys())


def test_invalid_utf8_text_rejected() -> None:
    tokenizer = FrequencyTokenizerV1()
    with pytest.raises(FormatError, match="UTF-8"):
        tokenizer.tokenize(b"\xff\xfe", domain="text")


def test_validate_reconstructive_metadata_success() -> None:
    tokenizer = FrequencyTokenizerV1()
    out = tokenizer.tokenize("hello world", domain="text")
    metadata = {"preprocessing_mode": "reconstructive", **out.metadata}
    validate_reconstructive_metadata(metadata)


def test_validate_reconstructive_metadata_missing_key_rejected() -> None:
    tokenizer = FrequencyTokenizerV1()
    out = tokenizer.tokenize("hello world", domain="text")
    metadata = {"preprocessing_mode": "reconstructive", **out.metadata}
    metadata.pop("vocab_hash")

    with pytest.raises(FormatError, match="Missing reconstructive metadata keys"):
        validate_reconstructive_metadata(metadata)


def test_validate_reconstructive_metadata_requires_mode() -> None:
    tokenizer = FrequencyTokenizerV1()
    out = tokenizer.tokenize("hello world", domain="text")
    metadata = dict(out.metadata)

    with pytest.raises(FormatError, match="preprocessing_mode='reconstructive'"):
        validate_reconstructive_metadata(metadata)


def test_validate_reconstructive_metadata_rejects_domain() -> None:
    tokenizer = FrequencyTokenizerV1()
    out = tokenizer.tokenize("hello world", domain="text")
    metadata = {"preprocessing_mode": "reconstructive", **out.metadata}
    metadata["domain_kind"] = "binary"

    with pytest.raises(FormatError, match="domain_kind"):
        validate_reconstructive_metadata(metadata)
