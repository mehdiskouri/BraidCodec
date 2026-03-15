"""Tests for braidcodec._exceptions and braidcodec.codec.schema.

Covers:
  - Exception hierarchy: inheritance chains, context dict
  - EncodedBlock: construction per tier, properties, to_dict/from_dict, validation
  - EncodedStream: to_bytes/from_bytes round-trip, multi-block, metadata
  - Corruption: bad magic, bad version, truncation, digest tampering, bit-flips
  - Hypothesis: round-trip + random corruption detection
"""

from __future__ import annotations

import struct

import blake3
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from braidcodec._exceptions import (
    BraidCodecError,
    BraidKeyError,
    ChecksumError,
    ChunkError,
    CompressionError,
    ContractionError,
    DigestError,
    EncodingError,
    FermionError,
    FormatError,
    IntegrityError,
    JonesError,
    KeyMismatchError,
    KeyValidationError,
    MagicMismatchError,
    RewriteVerificationError,
    TraceError,
    VersionError,
    WritheError,
)
from braidcodec.codec.schema import (
    EncodedBlock,
    EncodedStream,
    build_reconstructive_payload_metadata,
    compute_reconstructive_commitment,
    parse_reconstructive_payload_metadata,
    validate_reconstructive_commitment_metadata,
    validate_reconstructive_payload_metadata,
)

# ═══════════════════════════════════════════════════════════════════════════
# Exception hierarchy tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExceptionHierarchy:
    """Verify inheritance chains and structured context."""

    def test_base_stores_context(self) -> None:
        err = BraidCodecError("boom", key="val", num=42)
        assert str(err) == "boom"
        assert err.context == {"key": "val", "num": 42}

    def test_base_empty_context(self) -> None:
        err = BraidCodecError("simple")
        assert err.context == {}

    @pytest.mark.parametrize(
        ("cls", "parent"),
        [
            (FormatError, BraidCodecError),
            (MagicMismatchError, FormatError),
            (VersionError, FormatError),
            (DigestError, FormatError),
            (BraidKeyError, BraidCodecError),
            (KeyMismatchError, BraidKeyError),
            (KeyValidationError, BraidKeyError),
            (IntegrityError, BraidCodecError),
            (WritheError, IntegrityError),
            (JonesError, IntegrityError),
            (TraceError, IntegrityError),
            (FermionError, IntegrityError),
            (ChecksumError, IntegrityError),
            (EncodingError, BraidCodecError),
            (ChunkError, EncodingError),
            (ContractionError, EncodingError),
            (CompressionError, BraidCodecError),
            (RewriteVerificationError, CompressionError),
        ],
    )
    def test_inheritance(self, cls: type, parent: type) -> None:
        assert issubclass(cls, parent)

    def test_all_catchable_by_base(self) -> None:
        """Every leaf exception is catchable via BraidCodecError."""
        leaves = [
            MagicMismatchError,
            VersionError,
            DigestError,
            KeyMismatchError,
            KeyValidationError,
            WritheError,
            JonesError,
            TraceError,
            FermionError,
            ChecksumError,
            ChunkError,
            ContractionError,
            RewriteVerificationError,
        ]
        for cls in leaves:
            with pytest.raises(BraidCodecError):
                raise cls("test")

    def test_integrity_subtypes(self) -> None:
        """All 5 integrity error subtypes are IntegrityError."""
        for cls in [WritheError, JonesError, TraceError, FermionError, ChecksumError]:
            assert issubclass(cls, IntegrityError)
            err = cls("fail", channel="test")
            assert err.context["channel"] == "test"


# ═══════════════════════════════════════════════════════════════════════════
# EncodedBlock tests
# ═══════════════════════════════════════════════════════════════════════════

# -- Tier fixtures --

_TIER1 = EncodedBlock(
    generators=[1, -2, 3],
    n_strands=4,
    sector="TSR",
    writhe=1,
    block_index=0,
    original_length=5,
    invariant_tier=1,
)

_TIER2 = EncodedBlock(
    generators=[1, 2],
    n_strands=3,
    sector="Ising",
    writhe=2,
    block_index=1,
    original_length=4,
    invariant_tier=2,
    jones_real=0.5,
    jones_imag=-0.3,
)

_TIER3 = EncodedBlock(
    generators=[1, -1, 2],
    n_strands=4,
    sector="Fibonacci",
    writhe=0,
    block_index=2,
    original_length=8,
    invariant_tier=3,
    trace_real=2.1,
    trace_imag=-1.4,
)


class TestEncodedBlockConstruction:
    """Verify dataclass construction enforces tier constraints."""

    def test_tier1_no_invariants(self) -> None:
        assert _TIER1.jones is None
        assert _TIER1.trace_invariant is None

    def test_tier2_jones_present(self) -> None:
        assert _TIER2.jones == complex(0.5, -0.3)
        assert _TIER2.trace_invariant is None

    def test_tier3_trace_present(self) -> None:
        assert _TIER3.trace_invariant == complex(2.1, -1.4)
        assert _TIER3.jones is None

    def test_invalid_tier_value(self) -> None:
        with pytest.raises(FormatError, match="invariant_tier must be 1, 2, or 3"):
            EncodedBlock(
                generators=[1],
                n_strands=2,
                sector="TSR",
                writhe=1,
                block_index=0,
                original_length=1,
                invariant_tier=4,
            )

    def test_tier1_with_jones_rejected(self) -> None:
        with pytest.raises(FormatError, match=r"Tier 1.*Jones"):
            EncodedBlock(
                generators=[1],
                n_strands=2,
                sector="TSR",
                writhe=1,
                block_index=0,
                original_length=1,
                invariant_tier=1,
                jones_real=1.0,
                jones_imag=0.0,
            )

    def test_tier1_with_trace_rejected(self) -> None:
        with pytest.raises(FormatError, match=r"Tier 1.*trace"):
            EncodedBlock(
                generators=[1],
                n_strands=2,
                sector="TSR",
                writhe=1,
                block_index=0,
                original_length=1,
                invariant_tier=1,
                trace_real=1.0,
                trace_imag=0.0,
            )

    def test_tier2_missing_jones_rejected(self) -> None:
        with pytest.raises(FormatError, match=r"Tier 2.*jones"):
            EncodedBlock(
                generators=[1],
                n_strands=2,
                sector="TSR",
                writhe=1,
                block_index=0,
                original_length=1,
                invariant_tier=2,
            )

    def test_tier2_with_trace_rejected(self) -> None:
        with pytest.raises(FormatError, match=r"Tier 2.*trace"):
            EncodedBlock(
                generators=[1],
                n_strands=2,
                sector="TSR",
                writhe=1,
                block_index=0,
                original_length=1,
                invariant_tier=2,
                jones_real=1.0,
                jones_imag=2.0,
                trace_real=0.5,
                trace_imag=0.5,
            )

    def test_tier3_missing_trace_rejected(self) -> None:
        with pytest.raises(FormatError, match=r"Tier 3.*trace"):
            EncodedBlock(
                generators=[1],
                n_strands=2,
                sector="TSR",
                writhe=1,
                block_index=0,
                original_length=1,
                invariant_tier=3,
            )

    def test_tier3_with_jones_rejected(self) -> None:
        with pytest.raises(FormatError, match=r"Tier 3.*Jones"):
            EncodedBlock(
                generators=[1],
                n_strands=2,
                sector="TSR",
                writhe=1,
                block_index=0,
                original_length=1,
                invariant_tier=3,
                trace_real=1.0,
                trace_imag=0.0,
                jones_real=1.0,
                jones_imag=0.0,
            )

    def test_frozen_enforcement(self) -> None:
        with pytest.raises(AttributeError):
            _TIER1.writhe = 99  # type: ignore[misc]


class TestEncodedBlockDictRoundTrip:
    """Verify to_dict / from_dict serialization."""

    @pytest.mark.parametrize("block", [_TIER1, _TIER2, _TIER3], ids=["t1", "t2", "t3"])
    def test_roundtrip(self, block: EncodedBlock) -> None:
        d = block.to_dict()
        recovered = EncodedBlock.from_dict(d)
        assert recovered == block

    def test_missing_required_key(self) -> None:
        d = _TIER1.to_dict()
        del d["generators"]
        with pytest.raises(FormatError, match="missing required keys"):
            EncodedBlock.from_dict(d)  # type: ignore[arg-type]

    def test_bad_type_coercion(self) -> None:
        d = _TIER1.to_dict()
        d["generators"] = "not_a_list"
        # from_dict calls list("not_a_list") which makes a list of chars — valid
        # But n_strands="bogus" would fail at int()
        d["n_strands"] = "bogus"
        with pytest.raises(FormatError, match="Invalid EncodedBlock data"):
            EncodedBlock.from_dict(d)  # type: ignore[arg-type]

    def test_empty_generators(self) -> None:
        block = EncodedBlock(
            generators=[],
            n_strands=2,
            sector="TSR",
            writhe=0,
            block_index=0,
            original_length=0,
            invariant_tier=1,
        )
        d = block.to_dict()
        recovered = EncodedBlock.from_dict(d)  # type: ignore[arg-type]
        assert recovered.generators == []

    def test_decode_generators_roundtrip(self) -> None:
        """decode_generators survives to_dict / from_dict."""
        block = EncodedBlock(
            generators=[2, 1],
            n_strands=4,
            sector="TSR",
            writhe=2,
            block_index=0,
            original_length=5,
            invariant_tier=1,
            decode_generators=[1, 3, -1, 2],
        )
        d = block.to_dict()
        assert d["decode_generators"] == [1, 3, -1, 2]
        recovered = EncodedBlock.from_dict(d)  # type: ignore[arg-type]
        assert recovered.decode_generators == [1, 3, -1, 2]
        assert recovered.effective_decode_generators == [1, 3, -1, 2]

    def test_decode_generators_none_roundtrip(self) -> None:
        """decode_generators=None round-trips correctly."""
        d = _TIER1.to_dict()
        assert d["decode_generators"] is None
        recovered = EncodedBlock.from_dict(d)  # type: ignore[arg-type]
        assert recovered.decode_generators is None
        assert recovered.effective_decode_generators == recovered.generators

    def test_decode_generators_wire_format(self) -> None:
        """Compressed block with decode_generators survives wire serialisation."""
        block = EncodedBlock(
            generators=[2, 1],
            n_strands=4,
            sector="TSR",
            writhe=2,
            block_index=0,
            original_length=5,
            invariant_tier=1,
            decode_generators=[1, 3, -1, 2],
        )
        stream = EncodedStream(
            blocks=(block,),
            n_strands=4,
            sector="TSR",
            total_bytes=5,
            checksum=blake3.blake3(b"dummy").digest(),
            timestamp=1_000_000_000,
        )
        wire = stream.to_bytes()
        recovered = EncodedStream.from_bytes(wire)
        rb = recovered.blocks[0]
        assert rb.decode_generators == [1, 3, -1, 2]
        assert rb.effective_decode_generators == [1, 3, -1, 2]

    def test_topology_fields_roundtrip(self) -> None:
        block = EncodedBlock(
            generators=[1, -2, 1],
            n_strands=4,
            sector="TSR",
            writhe=0,
            block_index=3,
            original_length=6,
            invariant_tier=3,
            trace_real=1.5,
            trace_imag=-0.25,
            topology_layer_index=2,
            topology_layer_n_chunks=7,
            topology_nnz_bits=23,
            topology_dt_scale=1.125,
            topology_hash32=123456,
            topology_density_fp=1200,
            topology_centroid_fp=42000,
            topology_variance_fp=900,
            topology_morton_key=987654,
            topology_commitment=123456789,
        )
        d = block.to_dict()
        recovered = EncodedBlock.from_dict(d)  # type: ignore[arg-type]
        assert recovered.topology_layer_index == 2
        assert recovered.topology_layer_n_chunks == 7
        assert recovered.topology_nnz_bits == 23
        assert recovered.topology_dt_scale == 1.125
        assert recovered.topology_hash32 == 123456
        assert recovered.topology_density_fp == 1200
        assert recovered.topology_centroid_fp == 42000
        assert recovered.topology_variance_fp == 900
        assert recovered.topology_morton_key == 987654
        assert recovered.topology_commitment == 123456789


class TestReconstructivePayloadMetadata:
    def _sample_metadata(self) -> dict[str, str]:
        return {
            "preprocessing_mode": "reconstructive",
            "tokenizer_id": "frequency-tokenizer",
            "tokenizer_version": "v1",
            "domain_kind": "text",
            "bin_count": "128",
            "bin_table_hash": "a" * 64,
            "vocab_hash": "b" * 64,
            "normalization_profile_id": "osc-norm-v1",
            "normalization_profile_hash": "c" * 64,
            "manifold_profile_id": "hypergraph-manifold-v1",
            "manifold_state_hash": "d" * 64,
            "reconstructive_graph_hash": "e" * 64,
            "km_residual_max": "0.001",
            "km_residual_mean": "0.0001",
            "km_iters_mean": "5",
            "km_valid_ratio": "1",
            "km_seed_vector": "0.1,0.2,0.3",
            "km_kappa": "0.3",
            "km_eta": "0.654",
            "km_tol": "1e-5",
            "km_max_iter": "100",
            "km_threshold_residual_max": "1e-5",
            "km_threshold_valid_ratio": "1",
            "reconstructive_program_type": "json-literal-v1",
            "reconstructive_program_payload": '{"raw_json":"{}"}',
        }

    def test_build_parse_validate_roundtrip(self) -> None:
        meta = self._sample_metadata()
        payload = build_reconstructive_payload_metadata(meta)
        meta["reconstructive_payload_v1"] = payload

        parsed = parse_reconstructive_payload_metadata(meta)
        assert parsed["model_id"] == "frequency-manifold"
        assert parsed["model_version"] == "1"
        validated = validate_reconstructive_payload_metadata(meta)
        assert validated["domain_kind"] == "text"

    def test_validate_rejects_payload_mismatch(self) -> None:
        meta = self._sample_metadata()
        payload = build_reconstructive_payload_metadata(meta)
        meta["reconstructive_payload_v1"] = payload
        meta["vocab_hash"] = "f" * 64

        with pytest.raises(FormatError, match="payload mismatch"):
            validate_reconstructive_payload_metadata(meta)

    def test_commitment_validation_rejects_mismatch(self) -> None:
        meta = self._sample_metadata()
        payload = build_reconstructive_payload_metadata(meta)
        meta["reconstructive_payload_v1"] = payload

        block = EncodedBlock(
            generators=[1, -2, 1],
            n_strands=4,
            sector="TSR",
            writhe=0,
            block_index=0,
            original_length=3,
            invariant_tier=1,
        )
        meta["reconstructive_commitment"] = compute_reconstructive_commitment(payload, (block,))
        validate_reconstructive_commitment_metadata(meta, (block,))

        bad_meta = dict(meta)
        bad_meta["reconstructive_commitment"] = "0" * 64
        with pytest.raises(FormatError, match="commitment mismatch"):
            validate_reconstructive_commitment_metadata(bad_meta, (block,))


# ═══════════════════════════════════════════════════════════════════════════
# EncodedStream round-trip tests
# ═══════════════════════════════════════════════════════════════════════════


def _make_checksum(data: bytes = b"Hello") -> bytes:
    return blake3.blake3(data).digest()


def _make_stream(
    blocks: tuple[EncodedBlock, ...] | None = None,
    **kwargs: object,
) -> EncodedStream:
    """Convenience factory with sensible defaults."""
    defaults: dict[str, object] = {
        "blocks": (_TIER1,) if blocks is None else blocks,
        "n_strands": 4,
        "sector": "TSR",
        "total_bytes": 5,
        "checksum": _make_checksum(),
        "timestamp": 1_000_000_000,
        "metadata": {},
    }
    defaults.update(kwargs)
    return EncodedStream(**defaults)  # type: ignore[arg-type]


class TestEncodedStreamRoundTrip:
    """Verify to_bytes / from_bytes produce identical streams."""

    def test_single_block(self) -> None:
        stream = _make_stream()
        wire = stream.to_bytes()
        recovered = EncodedStream.from_bytes(wire)
        assert recovered.version == stream.version
        assert recovered.n_strands == stream.n_strands
        assert recovered.sector == stream.sector
        assert recovered.total_bytes == stream.total_bytes
        assert recovered.checksum == stream.checksum
        assert recovered.timestamp == stream.timestamp
        assert recovered.metadata == stream.metadata
        assert len(recovered.blocks) == 1
        assert recovered.blocks[0] == _TIER1

    def test_multiple_blocks_mixed_tiers(self) -> None:
        stream = _make_stream(blocks=(_TIER1, _TIER2, _TIER3))
        wire = stream.to_bytes()
        recovered = EncodedStream.from_bytes(wire)
        assert len(recovered.blocks) == 3
        assert recovered.blocks[0] == _TIER1
        assert recovered.blocks[1] == _TIER2
        assert recovered.blocks[2] == _TIER3

    def test_metadata_preserved(self) -> None:
        meta = {"encoder_version": "0.1.0", "note": "test"}
        stream = _make_stream(metadata=meta)
        wire = stream.to_bytes()
        recovered = EncodedStream.from_bytes(wire)
        assert recovered.metadata == meta

    def test_large_metadata(self) -> None:
        meta = {f"key_{i}": f"value_{i}" for i in range(100)}
        stream = _make_stream(metadata=meta)
        wire = stream.to_bytes()
        recovered = EncodedStream.from_bytes(wire)
        assert recovered.metadata == meta

    def test_binary_checksum_preserved(self) -> None:
        cksum = bytes(range(32))
        stream = _make_stream(checksum=cksum)
        wire = stream.to_bytes()
        recovered = EncodedStream.from_bytes(wire)
        assert recovered.checksum == cksum

    def test_version_field(self) -> None:
        stream = _make_stream()
        wire = stream.to_bytes()
        recovered = EncodedStream.from_bytes(wire)
        assert recovered.version == 2

    def test_version_1_backward_compatibility(self) -> None:
        stream = _make_stream(version=1)
        wire = stream.to_bytes()
        recovered = EncodedStream.from_bytes(wire)
        assert recovered.version == 1

    def test_empty_blocks_list(self) -> None:
        stream = _make_stream(blocks=())
        wire = stream.to_bytes()
        recovered = EncodedStream.from_bytes(wire)
        assert len(recovered.blocks) == 0

    def test_wire_starts_with_magic(self) -> None:
        wire = _make_stream().to_bytes()
        assert wire[:4] == b"BRDC"

    def test_wire_ends_with_32_byte_digest(self) -> None:
        wire = _make_stream().to_bytes()
        payload = wire[:-32]
        expected = blake3.blake3(payload).digest()
        assert wire[-32:] == expected

    def test_from_bytes_accepts_bytearray(self) -> None:
        wire = _make_stream().to_bytes()
        recovered = EncodedStream.from_bytes(bytearray(wire))
        assert recovered.n_strands == 4

    def test_from_bytes_accepts_memoryview(self) -> None:
        wire = _make_stream().to_bytes()
        recovered = EncodedStream.from_bytes(memoryview(wire))
        assert recovered.n_strands == 4

    def test_hdf5_roundtrip(self) -> None:
        h5py = pytest.importorskip("h5py")
        assert h5py is not None
        stream = _make_stream(blocks=(_TIER1, _TIER2, _TIER3))
        payload = stream.to_hdf5_bytes()
        recovered = EncodedStream.from_hdf5_bytes(payload)
        assert recovered.version == stream.version
        assert recovered.n_strands == stream.n_strands
        assert recovered.sector == stream.sector
        assert recovered.total_bytes == stream.total_bytes
        assert recovered.checksum == stream.checksum
        assert recovered.metadata == stream.metadata
        assert recovered.blocks == stream.blocks

    def test_hdf5_is_more_compact_for_long_generators(self) -> None:
        h5py = pytest.importorskip("h5py")
        assert h5py is not None
        long_block = EncodedBlock(
            generators=[1 if i % 2 == 0 else -2 for i in range(120_000)],
            n_strands=4,
            sector="TSR",
            writhe=0,
            block_index=0,
            original_length=8192,
            invariant_tier=1,
        )
        stream = _make_stream(
            blocks=(long_block,),
            n_strands=4,
            total_bytes=8192,
            checksum=_make_checksum(b"x" * 8192),
        )
        wire = stream.to_bytes()
        h5 = stream.to_hdf5_bytes()
        assert len(h5) < len(wire)


# ═══════════════════════════════════════════════════════════════════════════
# Corruption / error detection tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCorruptionDetection:
    """Verify that corrupted wire data raises the correct exception."""

    def test_bad_magic(self) -> None:
        wire = bytearray(_make_stream().to_bytes())
        wire[0:4] = b"XXXX"
        with pytest.raises(MagicMismatchError, match="BRDC"):
            EncodedStream.from_bytes(bytes(wire))

    def test_unsupported_version(self) -> None:
        wire = bytearray(_make_stream().to_bytes())
        # Set version to 99 and recompute digest
        struct.pack_into(">H", wire, 4, 99)
        _recompute_digest(wire)
        with pytest.raises(VersionError, match="99"):
            EncodedStream.from_bytes(bytes(wire))

    def test_truncated_too_short(self) -> None:
        with pytest.raises(FormatError, match="too short"):
            EncodedStream.from_bytes(b"BRDC" + b"\x00" * 10)

    def test_corrupted_digest_flip_bit(self) -> None:
        wire = bytearray(_make_stream().to_bytes())
        wire[-1] ^= 0xFF  # flip last byte of digest
        with pytest.raises(DigestError, match="digest mismatch"):
            EncodedStream.from_bytes(bytes(wire))

    def test_corrupted_header_content(self) -> None:
        wire = bytearray(_make_stream().to_bytes())
        # Corrupt a byte in the header region (after magic+version+header_len = offset 10)
        wire[12] ^= 0xFF
        _recompute_digest(wire)
        # Should fail during msgpack unpack or dict parsing
        with pytest.raises((FormatError, Exception)):
            EncodedStream.from_bytes(bytes(wire))

    def test_corrupted_block_content(self) -> None:
        wire = bytearray(_make_stream().to_bytes())
        # Corrupt a byte near the middle (likely inside block data)
        mid = len(wire) // 2
        wire[mid] ^= 0xFF
        _recompute_digest(wire)
        with pytest.raises((FormatError, Exception)):
            EncodedStream.from_bytes(bytes(wire))

    def test_truncated_header_length(self) -> None:
        wire = bytearray(_make_stream().to_bytes())
        # Set header_len to something huge
        struct.pack_into(">I", wire, 6, 999999)
        _recompute_digest(wire)
        with pytest.raises(FormatError, match="exceeds"):
            EncodedStream.from_bytes(bytes(wire))

    def test_data_under_46_bytes(self) -> None:
        with pytest.raises(FormatError, match="too short"):
            EncodedStream.from_bytes(b"\x00" * 45)


# ═══════════════════════════════════════════════════════════════════════════
# Hypothesis property tests
# ═══════════════════════════════════════════════════════════════════════════


def _block_strategy(tier: int) -> st.SearchStrategy[EncodedBlock]:
    """Strategy for a valid EncodedBlock of the given tier."""
    base = {
        "generators": st.lists(
            st.integers(min_value=-5, max_value=5).filter(lambda x: x != 0),
            min_size=1,
            max_size=32,
        ),
        "n_strands": st.integers(min_value=2, max_value=6),
        "sector": st.sampled_from(["TSR", "Ising", "Fibonacci", "Identity", "SU2k2"]),
        "writhe": st.integers(min_value=-100, max_value=100),
        "block_index": st.integers(min_value=0, max_value=1000),
        "original_length": st.integers(min_value=0, max_value=64),
        "invariant_tier": st.just(tier),
    }
    if tier == 1:
        pass
    elif tier == 2:
        base["jones_real"] = st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        )
        base["jones_imag"] = st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        )
    elif tier == 3:
        base["trace_real"] = st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        )
        base["trace_imag"] = st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        )
    return st.builds(EncodedBlock, **base)


_any_block_st = st.one_of(_block_strategy(1), _block_strategy(2), _block_strategy(3))


def _stream_strategy() -> st.SearchStrategy[EncodedStream]:
    return st.builds(
        EncodedStream,
        blocks=st.lists(_any_block_st, min_size=0, max_size=5).map(tuple),
        n_strands=st.integers(min_value=2, max_value=6),
        sector=st.sampled_from(["TSR", "Ising", "Fibonacci"]),
        total_bytes=st.integers(min_value=0, max_value=100000),
        checksum=st.binary(min_size=32, max_size=32),
        timestamp=st.integers(min_value=0, max_value=2**63 - 1),
        metadata=st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.text(min_size=0, max_size=50),
            max_size=5,
        ),
    )


class TestHypothesisStreamRoundTrip:
    """Property-based round-trip and corruption detection."""

    @given(stream=_stream_strategy())
    @settings(max_examples=1000, deadline=None)
    def test_roundtrip_identity(self, stream: EncodedStream) -> None:
        wire = stream.to_bytes()
        recovered = EncodedStream.from_bytes(wire)
        assert recovered.version == stream.version
        assert recovered.n_strands == stream.n_strands
        assert recovered.sector == stream.sector
        assert recovered.total_bytes == stream.total_bytes
        assert recovered.checksum == stream.checksum
        assert recovered.timestamp == stream.timestamp
        assert recovered.metadata == stream.metadata
        assert len(recovered.blocks) == len(stream.blocks)
        for orig, rec in zip(stream.blocks, recovered.blocks, strict=True):
            assert rec == orig

    @given(stream=_stream_strategy(), flip_pos=st.data())
    @settings(max_examples=1000, deadline=None)
    def test_random_bit_flip_detected(
        self, stream: EncodedStream, flip_pos: st.DataObject
    ) -> None:
        """A single random bit-flip in the wire bytes is always detected."""
        wire = bytearray(stream.to_bytes())
        idx = flip_pos.draw(st.integers(min_value=0, max_value=len(wire) - 1))
        bit = flip_pos.draw(st.integers(min_value=0, max_value=7))
        wire[idx] ^= 1 << bit
        corrupted = bytes(wire)
        with pytest.raises(
            (MagicMismatchError, VersionError, DigestError, FormatError, Exception)
        ):
            EncodedStream.from_bytes(corrupted)

    @given(block=_any_block_st)
    @settings(max_examples=1000, deadline=None)
    def test_block_dict_roundtrip(self, block: EncodedBlock) -> None:
        d = block.to_dict()
        recovered = EncodedBlock.from_dict(d)  # type: ignore[arg-type]
        assert recovered == block


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _recompute_digest(wire: bytearray) -> None:
    """Recompute and overwrite the trailing BLAKE3 digest in-place."""
    payload = bytes(wire[:-32])
    digest = blake3.blake3(payload).digest()
    wire[-32:] = digest
