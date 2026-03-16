"""Tests for deterministic hypergraph manifold helpers (Phase D)."""

from __future__ import annotations

import numpy as np

from braidcodec.codec.manifold import (
    build_coupling_matrix_summary,
    build_deterministic_hypergraph,
    coupling_matrix_metadata,
    fit_compact_manifold_state,
    manifold_metadata,
    manifold_seed_vector,
)
from braidcodec.codec.oscillator_normalization import OscillatorNormalizationV1
from braidcodec.codec.tokenizer_frequency import FrequencyTokenizerV1


def test_hypergraph_determinism() -> None:
    tokenizer = FrequencyTokenizerV1()
    tokens = tokenizer.tokenize("alpha beta gamma", domain="text").tokens

    g1 = build_deterministic_hypergraph(tokens, layer_count=8)
    g2 = build_deterministic_hypergraph(tokens, layer_count=8)

    assert g1 == g2
    assert g1.graph_hash == g2.graph_hash


def test_manifold_state_determinism() -> None:
    tokenizer = FrequencyTokenizerV1()
    normalizer = OscillatorNormalizationV1()

    tokens = tokenizer.tokenize('{"b":2,"a":1}', domain="json").tokens
    points = normalizer.normalize_tokens(tokens).points
    graph = build_deterministic_hypergraph(tokens, layer_count=8)

    s1 = fit_compact_manifold_state(points, graph)
    s2 = fit_compact_manifold_state(points, graph)

    assert s1 == s2
    assert s1.state_hash == s2.state_hash


def test_seed_vector_shape_and_type() -> None:
    tokenizer = FrequencyTokenizerV1()
    normalizer = OscillatorNormalizationV1()

    tokens = tokenizer.tokenize("2026-03-15T12:00:00Z INFO core started", domain="logs").tokens
    points = normalizer.normalize_tokens(tokens).points
    graph = build_deterministic_hypergraph(tokens, layer_count=4)
    state = fit_compact_manifold_state(points, graph)

    seed = manifold_seed_vector(state)
    assert seed.dtype == np.float64
    assert seed.ndim == 1
    assert seed.size == (len(state.layer_vectors) * 5 + 5)


def test_metadata_fields_present() -> None:
    tokenizer = FrequencyTokenizerV1()
    normalizer = OscillatorNormalizationV1()

    tokens = tokenizer.tokenize("hello world", domain="text").tokens
    points = normalizer.normalize_tokens(tokens).points
    graph = build_deterministic_hypergraph(tokens)
    state = fit_compact_manifold_state(points, graph)

    metadata = manifold_metadata(state)
    assert metadata["manifold_profile_id"] == "hypergraph-manifold-v1"
    assert int(metadata["manifold_layer_count"]) == state.layer_count
    assert metadata["manifold_state_hash"] == state.state_hash


def test_coupling_matrix_summary_determinism() -> None:
    tokenizer = FrequencyTokenizerV1()
    tokens = tokenizer.tokenize("alpha beta gamma alpha beta", domain="text").tokens

    c1 = build_coupling_matrix_summary(tokens, window=4)
    c2 = build_coupling_matrix_summary(tokens, window=4)

    assert c1 == c2
    assert c1.nnz > 0
    assert 0.0 <= c1.density <= 1.0
    assert c1.spectral_radius >= 0.0


def test_coupling_matrix_metadata_fields_present() -> None:
    tokenizer = FrequencyTokenizerV1()
    tokens = tokenizer.tokenize("hello world hello", domain="text").tokens
    summary = build_coupling_matrix_summary(tokens)
    metadata = coupling_matrix_metadata(summary)

    assert int(metadata["coupling_matrix_dim"]) == summary.dimension
    assert int(metadata["coupling_matrix_nnz"]) == summary.nnz
    assert float(metadata["coupling_matrix_density"]) == summary.density
    assert float(metadata["coupling_matrix_spectral_radius"]) == summary.spectral_radius
    assert metadata["coupling_matrix_hash"] == summary.matrix_hash
