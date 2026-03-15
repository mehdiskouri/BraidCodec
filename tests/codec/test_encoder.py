"""Tests for the encode pipeline."""

from __future__ import annotations

import blake3
import numpy as np
import pytest

from braidcodec.algebra.braid_equations import (
    BraidEquation,
    contract_braid_tensor,
    jones_polynomial,
    writhe,
)
from braidcodec.codec.chunker import bytes_to_generators, compute_block_size
from braidcodec.codec.encoder import _build_layer_aware_batches, encode
from braidcodec.codec.reconstructive_compact import parse_reconstructive_program_payload
from braidcodec.codec.schema import (
    EncodedStream,
    parse_reconstructive_payload_metadata,
    validate_reconstructive_metadata,
)
from braidcodec.codec.tokenizer_frequency import FrequencyTokenizerV1
from braidcodec.crypto.keys import BraidKey, keygen

# ── Helpers ───────────────────────────────────────────────────────────────

# Use generators_per_block=8 for fast tier-2 tests (2^8 = 256 state iterations).
_K_SMALL: int = 8
# Default generators_per_block=32 triggers tier 3 (trace only, fast).
_K_DEFAULT: int = 32


def _make_key(sector: str = "TSR", theta_offset: float = 1.0, n_strands: int = 4) -> BraidKey:
    return keygen(sector=sector, n_strands=n_strands, theta_offset=theta_offset)


# ── Basic encoding ────────────────────────────────────────────────────────


class TestEncodeBasic:
    def test_encode_returns_stream(self) -> None:
        key = _make_key()
        stream = encode(b"Hello", key, generators_per_block=_K_SMALL)
        assert isinstance(stream, EncodedStream)

    def test_stream_fields(self) -> None:
        key = _make_key()
        stream = encode(b"Hello", key, generators_per_block=_K_SMALL)
        assert stream.total_bytes == 5
        assert stream.n_strands == key.n_strands
        assert stream.sector == key.sector
        assert stream.version == 2

    def test_checksum_matches(self) -> None:
        data = b"Hello, World!"
        key = _make_key()
        stream = encode(data, key, generators_per_block=_K_SMALL)
        assert stream.checksum == blake3.blake3(data).digest()

    def test_generators_valid(self) -> None:
        key = _make_key()
        stream = encode(b"\x01\x02", key, generators_per_block=_K_SMALL)
        for block in stream.blocks:
            for g in block.generators:
                assert g != 0
                assert abs(g) < key.n_strands

    def test_writhe_correctness(self) -> None:
        """Writhe is computed on the *original* generators (pre-simplification)."""
        key = _make_key()
        data = b"\x01\x02"
        stream = encode(
            data,
            key,
            generators_per_block=_K_SMALL,
            preprocessing_mode="legacy",
        )
        block = stream.blocks[0]

        # Independently compute writhe on original generators.
        bs = compute_block_size(key.n_strands, _K_SMALL)
        padded = data.ljust(bs, b"\x00")
        original_gens = bytes_to_generators(padded, key.n_strands, _K_SMALL)
        expected_writhe = writhe(BraidEquation(key.n_strands, original_gens, sector=key.sector))
        assert block.writhe == expected_writhe


# ── Jones correctness (tier 2) ────────────────────────────────────────────


class TestJonesCorrectness:
    def test_jones_matches_independent_computation(self) -> None:
        key = _make_key()
        data = b"\x01\x02"
        stream = encode(
            data,
            key,
            generators_per_block=_K_SMALL,
            preprocessing_mode="legacy",
        )
        block = stream.blocks[0]
        assert block.invariant_tier == 2

        # Independently replicate the encoder pipeline.
        bs = compute_block_size(key.n_strands, _K_SMALL)
        padded = data.ljust(bs, b"\x00")
        gens = bytes_to_generators(padded, key.n_strands, _K_SMALL)
        braid = BraidEquation(
            key.n_strands,
            gens,
            sector=key.sector,
            _sector_params=key.sector_params,
        )
        expected = jones_polynomial(braid)

        assert block.jones is not None
        assert abs(block.jones - expected) < 1e-10


# ── Trace correctness (tier 3) ────────────────────────────────────────────


class TestTraceCorrectness:
    def test_trace_matches_independent_computation(self) -> None:
        key = _make_key()
        data = b"Hello"
        stream = encode(
            data,
            key,
            generators_per_block=_K_DEFAULT,
            preprocessing_mode="legacy",
        )
        block = stream.blocks[0]
        assert block.invariant_tier == 3

        # Independently replicate.
        bs = compute_block_size(key.n_strands, _K_DEFAULT)
        padded = data.ljust(bs, b"\x00")
        gens = bytes_to_generators(padded, key.n_strands, _K_DEFAULT)
        braid = BraidEquation(
            key.n_strands,
            gens,
            sector=key.sector,
            _sector_params=key.sector_params,
        )
        matrix = contract_braid_tensor(braid)
        expected = complex(np.trace(matrix))

        assert block.trace_invariant is not None
        assert abs(block.trace_invariant - expected) < 1e-10


# ── Tier selection ────────────────────────────────────────────────────────


class TestTierSelection:
    def test_tier_2_for_small_k(self) -> None:
        key = _make_key()
        stream = encode(b"\x01\x02", key, generators_per_block=_K_SMALL)
        for block in stream.blocks:
            assert block.invariant_tier == 2
            assert block.jones is not None
            assert block.trace_invariant is None

    def test_tier_3_for_large_k(self) -> None:
        key = _make_key()
        stream = encode(b"Hello", key, generators_per_block=_K_DEFAULT)
        for block in stream.blocks:
            assert block.invariant_tier == 3
            assert block.jones is None
            assert block.trace_invariant is not None


# ── Sectors ───────────────────────────────────────────────────────────────


class TestEncodeSectors:
    @pytest.mark.parametrize("sector", ["Identity", "TSR", "Ising", "Fibonacci", "SU2k2"])
    def test_sector_produces_valid_stream(self, sector: str) -> None:
        key = _make_key(sector=sector)
        stream = encode(b"\x01\x02", key, generators_per_block=_K_SMALL)
        assert stream.sector == sector
        assert len(stream.blocks) >= 1

    def test_different_keys_different_invariants(self) -> None:
        key1 = _make_key(theta_offset=0.5)
        key2 = _make_key(theta_offset=3.0)
        data = b"\x01\x02"
        s1 = encode(data, key1, generators_per_block=_K_SMALL)
        s2 = encode(data, key2, generators_per_block=_K_SMALL)
        # Same data, different keys → different Jones polynomials.
        assert s1.blocks[0].jones != s2.blocks[0].jones


# ── Multi-block ───────────────────────────────────────────────────────────


class TestEncodeMultiBlock:
    def test_sequential_block_indices(self) -> None:
        key = _make_key()
        stream = encode(b"A" * 100, key, generators_per_block=_K_DEFAULT)
        indices = [b.block_index for b in stream.blocks]
        assert indices == list(range(len(indices)))
        assert len(indices) > 1

    def test_empty_data(self) -> None:
        key = _make_key()
        stream = encode(b"", key, generators_per_block=_K_SMALL)
        assert stream.total_bytes == 0
        assert len(stream.blocks) == 1
        assert stream.blocks[0].original_length == 0

    def test_block_count(self) -> None:
        key = _make_key()
        bs = compute_block_size(key.n_strands, _K_DEFAULT)
        data = b"X" * (bs * 3)  # exactly 3 full blocks
        stream = encode(data, key, generators_per_block=_K_DEFAULT)
        assert len(stream.blocks) == 3


# ── Parallelism ───────────────────────────────────────────────────────────


class TestEncodeParallelism:
    def test_serial_parallel_blocks_match(self) -> None:
        data = b"A" * 100
        key = _make_key()
        serial = encode(data, key, generators_per_block=_K_DEFAULT, max_workers=1)
        parallel = encode(data, key, generators_per_block=_K_DEFAULT, max_workers=2)
        assert len(serial.blocks) == len(parallel.blocks)
        for sb, pb in zip(serial.blocks, parallel.blocks, strict=True):
            assert sb.generators == pb.generators
            assert sb.writhe == pb.writhe
            assert sb.invariant_tier == pb.invariant_tier
            assert sb.block_index == pb.block_index

    def test_max_workers_1(self) -> None:
        key = _make_key()
        stream = encode(b"Hello", key, generators_per_block=_K_SMALL, max_workers=1)
        assert isinstance(stream, EncodedStream)


# ── Preprocessing modes ──────────────────────────────────────────────────


class TestEncodePreprocessingModes:
    def test_default_mode_matches_explicit_topology(self) -> None:
        key = _make_key()
        data = b"Topology default mode parity"

        implicit = encode(data, key, generators_per_block=_K_DEFAULT)
        explicit = encode(
            data,
            key,
            generators_per_block=_K_DEFAULT,
            preprocessing_mode="topology",
        )

        assert len(implicit.blocks) == len(explicit.blocks)
        for a, b in zip(implicit.blocks, explicit.blocks, strict=True):
            assert a.block_index == b.block_index
            assert a.generators == b.generators
            assert a.writhe == b.writhe
            assert a.invariant_tier == b.invariant_tier

    def test_legacy_mode_compatibility(self) -> None:
        key = _make_key()
        data = b"Legacy fallback compatibility" * 4

        topology = encode(
            data,
            key,
            generators_per_block=_K_SMALL,
            preprocessing_mode="topology",
        )
        legacy = encode(
            data,
            key,
            generators_per_block=_K_SMALL,
            preprocessing_mode="legacy",
        )

        assert len(topology.blocks) == len(legacy.blocks)
        assert [b.block_index for b in topology.blocks] == [
            b.block_index for b in legacy.blocks
        ]
        for tb, lb in zip(topology.blocks, legacy.blocks, strict=True):
            assert tb.generators != lb.generators
            assert tb.decode_generators is None
            assert tb.invariant_tier == lb.invariant_tier

    def test_invalid_preprocessing_mode_raises(self) -> None:
        key = _make_key()
        with pytest.raises(ValueError, match="preprocessing_mode"):
            encode(b"bad", key, preprocessing_mode="unknown")

    def test_reconstructive_mode_emits_contract_metadata(self) -> None:
        key = _make_key()
        data = b'{"msg":"hello","x":1}'
        stream = encode(
            data,
            key,
            generators_per_block=_K_SMALL,
            preprocessing_mode="reconstructive",
        )
        assert stream.metadata["preprocessing_mode"] == "reconstructive"
        validate_reconstructive_metadata(stream.metadata)
        payload = parse_reconstructive_payload_metadata(stream.metadata)
        assert payload["model_id"] == "frequency-manifold"
        assert payload["model_version"] == "1"
        assert "reconstructive_commitment" in stream.metadata
        assert float(stream.metadata["km_residual_max"]) >= 0.0
        assert float(stream.metadata["km_valid_ratio"]) >= 0.0

    def test_reconstructive_mode_roundtrip_compatibility(self) -> None:
        key = _make_key()
        data = "Cafe\u0301\nlog line".encode()
        stream = encode(
            data,
            key,
            generators_per_block=_K_SMALL,
            preprocessing_mode="reconstructive",
            reconstructive_domain="text",
        )
        # Reconstructive route now applies an invertible payload-seeded transform.
        from braidcodec.codec.decoder import decode

        assert decode(stream, key, verify=False) == data

    def test_reconstructive_mode_uses_compact_no_block_payload(self) -> None:
        key = _make_key()
        data = ("Cafe\u0301\nlog line\n" * 10).encode("utf-8")
        reconstructive = encode(
            data,
            key,
            generators_per_block=_K_SMALL,
            preprocessing_mode="reconstructive",
            reconstructive_domain="text",
        )

        assert len(reconstructive.blocks) == 0
        assert reconstructive.metadata["execution_mode"] == "reconstructive-compact"

    def test_reconstructive_mode_rejects_non_utf8(self) -> None:
        key = _make_key()
        with pytest.raises(ValueError, match="UTF-8"):
            encode(
                b"\xff\xfe\xfd",
                key,
                generators_per_block=_K_SMALL,
                preprocessing_mode="reconstructive",
            )

    def test_reconstructive_mode_accepts_pretokenized_input(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        key = _make_key()
        data = ("Cafe\u0301\nlog line\n" * 5).encode("utf-8")

        tokenizer = FrequencyTokenizerV1()
        tokenized = tokenizer.tokenize(data, domain="text")

        def _fail_tokenize(
            _self: FrequencyTokenizerV1,
            _raw: object,
            *,
            _domain: str,
        ) -> object:
            raise AssertionError("tokenize should not be called when tokenization is provided")

        monkeypatch.setattr(FrequencyTokenizerV1, "tokenize", _fail_tokenize)

        stream = encode(
            data,
            key,
            generators_per_block=_K_SMALL,
            preprocessing_mode="reconstructive",
            reconstructive_domain="text",
            reconstructive_tokenization=tokenized,
        )

        assert stream.metadata["execution_mode"] == "reconstructive-compact"
        from braidcodec.codec.decoder import decode

        assert decode(stream, key, verify=False) == data

    def test_reconstructive_mode_domain_mismatch_raises(self) -> None:
        key = _make_key()
        data = b'{"items":[{"x":0,"y":0}]}'
        tokenized = FrequencyTokenizerV1().tokenize(data, domain="json")

        with pytest.raises(ValueError, match="does not match"):
            encode(
                data,
                key,
                generators_per_block=_K_SMALL,
                preprocessing_mode="reconstructive",
                reconstructive_domain="text",
                reconstructive_tokenization=tokenized,
            )

    def test_reconstructive_mode_emits_fidelity_bundle(self) -> None:
        key = _make_key()
        data = ("Cafe\u0301\nlog line\n" * 3).encode("utf-8")
        stream = encode(
            data,
            key,
            generators_per_block=_K_SMALL,
            preprocessing_mode="reconstructive",
            reconstructive_domain="text",
        )

        assert "fidelity_energy" in stream.metadata
        assert "fidelity_topology" in stream.metadata
        assert "fidelity_coherence" in stream.metadata
        assert "fidelity_bundle_v1" in stream.metadata
        assert "coupling_matrix_nnz" in stream.metadata
        assert "coupling_matrix_density" in stream.metadata
        assert "coupling_matrix_spectral_radius" in stream.metadata
        assert "coupling_matrix_hash" in stream.metadata

    def test_reconstructive_mode_uses_latent_residual_for_nonrepeating_text(self) -> None:
        key = _make_key()
        data = b"This text is not a strict periodic repeat block for compact replay."
        stream = encode(
            data,
            key,
            generators_per_block=_K_SMALL,
            preprocessing_mode="reconstructive",
            reconstructive_domain="text",
        )
        payload = parse_reconstructive_payload_metadata(stream.metadata)
        assert payload["reconstructive_program_type"] in {
            "latent-residual-v2",
            "latent-residual-v3",
        }
        program_payload = parse_reconstructive_program_payload(
            payload["reconstructive_program_payload"]
        )
        if payload["reconstructive_program_type"] == "latent-residual-v2":
            predictor = program_payload.get("predictor", program_payload.get("p"))
            codec = program_payload.get("codec", program_payload.get("c"))
            assert predictor in {
                "zero-v1",
                "prev-byte-v1",
                "spectral-byte-v1",
                "z",
                "p",
                "s",
            }
            assert codec in {"zlib-xor-v1", "bz2-xor-v1", "lzma-xor-v1", "z", "b", "l"}
        else:
            segments = program_payload.get("segments", program_payload.get("s"))
            original_length = program_payload.get("original_length", program_payload.get("n"))
            assert isinstance(segments, list)
            assert int(original_length) == len(data)

    def test_reconstructive_mode_keeps_repeat_program_for_periodic_text(self) -> None:
        key = _make_key()
        data = ("abc\n" * 20).encode("utf-8")
        stream = encode(
            data,
            key,
            generators_per_block=_K_SMALL,
            preprocessing_mode="reconstructive",
            reconstructive_domain="text",
        )
        payload = parse_reconstructive_payload_metadata(stream.metadata)
        assert payload["reconstructive_program_type"] == "repeat-text-v1"

    def test_metadata_contains_phase5_telemetry(self) -> None:
        key = _make_key()
        stream = encode(b"A" * 64, key, generators_per_block=_K_DEFAULT)
        assert stream.metadata["preprocessing_mode"] == "topology"
        assert stream.metadata["execution_mode"] in {"serial", "thread", "process"}
        assert int(stream.metadata["task_count"]) >= 1
        assert int(stream.metadata["batch_count"]) >= 1
        assert int(stream.metadata["batch_size"]) >= 1
        assert float(stream.metadata["estimated_cost_mean"]) >= 0.0
        assert float(stream.metadata["timing_preprocess_s"]) >= 0.0
        assert float(stream.metadata["timing_layer_order_s"]) >= 0.0
        assert float(stream.metadata["timing_encode_core_s"]) >= 0.0
        assert float(stream.metadata["timing_total_s"]) >= 0.0

    def test_topology_mode_avoids_decode_generator_duplication(self) -> None:
        key = _make_key()
        stream = encode(b"topology payload" * 2, key, generators_per_block=_K_SMALL)
        assert all(b.decode_generators is None for b in stream.blocks)
        assert all(b.topology_morton_key is not None for b in stream.blocks)
        assert all(b.topology_commitment is not None for b in stream.blocks)


class TestLayerAwareBatching:
    def test_batches_preserve_layer_boundaries(self) -> None:
        dummy = (
            b"x",
            4,
            "TSR",
            None,
            8,
            0,
            1,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        tasks = [
            (0, dummy),
            (0, dummy),
            (1, dummy),
            (1, dummy),
            (2, dummy),
        ]
        batches = _build_layer_aware_batches(tasks, batch_size=2)
        # Expect layer-aligned grouping with no cross-layer mixed batch.
        assert len(batches) == 3
        assert all(len(batch) <= 2 for batch in batches)
