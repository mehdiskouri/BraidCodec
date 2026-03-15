"""Wire-format schema for ``EncodedBlock`` and ``EncodedStream``.

Binary layout (``EncodedStream.to_bytes``):

    ┌──────────┬──────────┬───────────────┬────────────┐
    │ MAGIC 4B │ VER 2B   │ HDR_LEN 4B    │ HEADER …   │
    ├──────────┴──────────┴───────────────┴────────────┤
    │ BLK_COUNT 4B │ (LEN 4B + BLOCK …) × N           │
    ├──────────────┴───────────────────────────────────┤
    │ BLAKE3 DIGEST 32B                                │
    └──────────────────────────────────────────────────┘

The trailing BLAKE3 digest covers everything *before* it, and is verified
**before** msgpack deserialization as a defense-in-depth measure.
"""

from __future__ import annotations

import io
import json
import struct
import time
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import blake3
import msgpack
import numpy as np

from braidcodec._exceptions import (
    DigestError,
    FormatError,
    MagicMismatchError,
    VersionError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

# ── Constants ─────────────────────────────────────────────────────────────

_MAGIC: bytes = b"BRDC"
_CURRENT_VERSION: int = 2
_DIGEST_LEN: int = 32
# magic(4) + version(2) + header_len(4) + block_count(4) + digest(32) = 46
_MIN_STREAM_LEN: int = 46
_RECONSTRUCTIVE_REQUIRED_METADATA: frozenset[str] = frozenset(
    {
        "tokenizer_id",
        "tokenizer_version",
        "domain_kind",
        "bin_count",
        "bin_table_hash",
        "vocab_hash",
        "normalization_profile_id",
        "normalization_profile_hash",
    }
)
_RECONSTRUCTIVE_PAYLOAD_KEY = "reconstructive_payload_v1"
_RECONSTRUCTIVE_PAYLOAD_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "model_id",
        "model_version",
        "domain_kind",
        "tokenizer_id",
        "tokenizer_version",
        "bin_count",
        "bin_table_hash",
        "vocab_hash",
        "normalization_profile_id",
        "normalization_profile_hash",
        "manifold_profile_id",
        "manifold_state_hash",
        "reconstructive_graph_hash",
        "km_residual_max",
        "km_residual_mean",
        "km_iters_mean",
        "km_valid_ratio",
        "km_seed_vector",
        "km_kappa",
        "km_eta",
        "km_tol",
        "km_max_iter",
        "km_threshold_residual_max",
        "km_threshold_valid_ratio",
        "reconstructive_program_type",
        "reconstructive_program_payload",
    }
)


# ── EncodedBlock ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EncodedBlock:
    """A single encoded block — generator sequence plus invariants.

    Invariant-tier semantics:

    * **Tier 1** — writhe only; ``jones`` and ``trace_invariant`` are ``None``.
    * **Tier 2** — Jones polynomial computed; ``trace_invariant`` is ``None``.
    * **Tier 3** — matrix-trace invariant; ``jones`` is ``None``.
    """

    generators: list[int]
    n_strands: int
    sector: str
    writhe: int
    block_index: int
    original_length: int
    invariant_tier: int
    jones_real: float | None = None
    jones_imag: float | None = None
    trace_real: float | None = None
    trace_imag: float | None = None
    decode_generators: list[int] | None = None
    topology_layer_index: int | None = None
    topology_layer_n_chunks: int | None = None
    topology_nnz_bits: int | None = None
    topology_dt_scale: float | None = None
    topology_hash32: int | None = None
    topology_density_fp: int | None = None
    topology_centroid_fp: int | None = None
    topology_variance_fp: int | None = None
    topology_morton_key: int | None = None
    topology_commitment: int | None = None

    # Fields that must appear in the dict representation.
    _REQUIRED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "generators",
            "n_strands",
            "sector",
            "writhe",
            "block_index",
            "original_length",
            "invariant_tier",
        }
    )

    def __post_init__(self) -> None:
        # Validate decode_generators if present.
        dg = self.decode_generators
        if dg is not None and not all(g != 0 and abs(g) < self.n_strands for g in dg):
            raise FormatError(
                "decode_generators values must be non-zero with |g| < n_strands",
                n_strands=self.n_strands,
            )

        if self.topology_layer_index is not None and self.topology_layer_index < 0:
            raise FormatError("topology_layer_index must be >= 0")
        if self.topology_layer_n_chunks is not None and self.topology_layer_n_chunks < 1:
            raise FormatError("topology_layer_n_chunks must be >= 1")
        if self.topology_nnz_bits is not None and self.topology_nnz_bits < 0:
            raise FormatError("topology_nnz_bits must be >= 0")
        if self.topology_dt_scale is not None and self.topology_dt_scale <= 0:
            raise FormatError("topology_dt_scale must be > 0")

        for name, value in (
            ("topology_density_fp", self.topology_density_fp),
            ("topology_centroid_fp", self.topology_centroid_fp),
            ("topology_variance_fp", self.topology_variance_fp),
        ):
            if value is not None and not (0 <= value <= 65535):
                raise FormatError(f"{name} must be in [0, 65535]")

        if self.topology_morton_key is not None and self.topology_morton_key < 0:
            raise FormatError("topology_morton_key must be >= 0")
        if self.topology_commitment is not None and self.topology_commitment < 0:
            raise FormatError("topology_commitment must be >= 0")

        tier = self.invariant_tier
        if tier not in {1, 2, 3}:
            raise FormatError(
                f"invariant_tier must be 1, 2, or 3, got {tier}",
                invariant_tier=tier,
            )
        if tier == 1:
            if self.jones_real is not None or self.jones_imag is not None:
                raise FormatError(
                    "Tier 1 blocks must not carry Jones values",
                    invariant_tier=tier,
                )
            if self.trace_real is not None or self.trace_imag is not None:
                raise FormatError(
                    "Tier 1 blocks must not carry trace values",
                    invariant_tier=tier,
                )
        elif tier == 2:
            if self.jones_real is None or self.jones_imag is None:
                raise FormatError(
                    "Tier 2 blocks require jones_real and jones_imag",
                    invariant_tier=tier,
                )
            if self.trace_real is not None or self.trace_imag is not None:
                raise FormatError(
                    "Tier 2 blocks must not carry trace values",
                    invariant_tier=tier,
                )
        elif tier == 3:
            if self.trace_real is not None and self.trace_imag is None:
                raise FormatError(
                    "Tier 3 blocks require both trace_real and trace_imag",
                    invariant_tier=tier,
                )
            if self.trace_real is None or self.trace_imag is None:
                raise FormatError(
                    "Tier 3 blocks require trace_real and trace_imag",
                    invariant_tier=tier,
                )
            if self.jones_real is not None or self.jones_imag is not None:
                raise FormatError(
                    "Tier 3 blocks must not carry Jones values",
                    invariant_tier=tier,
                )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def effective_decode_generators(self) -> list[int]:
        """Generators to use for byte recovery.

        Returns ``decode_generators`` if present (compressed block),
        otherwise falls back to ``generators``.
        """
        return self.decode_generators if self.decode_generators is not None else self.generators

    @property
    def jones(self) -> complex | None:
        """Jones polynomial as a complex number, or ``None``."""
        if self.jones_real is not None and self.jones_imag is not None:
            return complex(self.jones_real, self.jones_imag)
        return None

    @property
    def trace_invariant(self) -> complex | None:
        """Matrix-trace invariant as a complex number, or ``None``."""
        if self.trace_real is not None and self.trace_imag is not None:
            return complex(self.trace_real, self.trace_imag)
        return None

    # ── Dict serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, object]:
        """Produce a plain-dict suitable for msgpack serialization."""
        d: dict[str, object] = {
            "generators": self.generators,
            "n_strands": self.n_strands,
            "sector": self.sector,
            "writhe": self.writhe,
            "block_index": self.block_index,
            "original_length": self.original_length,
            "invariant_tier": self.invariant_tier,
            "jones_real": self.jones_real,
            "jones_imag": self.jones_imag,
            "trace_real": self.trace_real,
            "trace_imag": self.trace_imag,
            "decode_generators": self.decode_generators,
            "topology_layer_index": self.topology_layer_index,
            "topology_layer_n_chunks": self.topology_layer_n_chunks,
            "topology_nnz_bits": self.topology_nnz_bits,
            "topology_dt_scale": self.topology_dt_scale,
            "topology_hash32": self.topology_hash32,
            "topology_density_fp": self.topology_density_fp,
            "topology_centroid_fp": self.topology_centroid_fp,
            "topology_variance_fp": self.topology_variance_fp,
            "topology_morton_key": self.topology_morton_key,
            "topology_commitment": self.topology_commitment,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EncodedBlock:
        """Reconstruct from a plain dict.  Raises ``FormatError`` on bad input."""
        missing = cls._REQUIRED_KEYS - d.keys()
        if missing:
            raise FormatError(
                f"EncodedBlock missing required keys: {sorted(missing)}",
                missing_keys=sorted(missing),
            )
        try:
            return cls(
                generators=list(d["generators"]),
                n_strands=int(d["n_strands"]),
                sector=str(d["sector"]),
                writhe=int(d["writhe"]),
                block_index=int(d["block_index"]),
                original_length=int(d["original_length"]),
                invariant_tier=int(d["invariant_tier"]),
                jones_real=_opt_float(d.get("jones_real")),
                jones_imag=_opt_float(d.get("jones_imag")),
                trace_real=_opt_float(d.get("trace_real")),
                trace_imag=_opt_float(d.get("trace_imag")),
                decode_generators=_opt_int_list(d.get("decode_generators")),
                topology_layer_index=_opt_int(d.get("topology_layer_index")),
                topology_layer_n_chunks=_opt_int(d.get("topology_layer_n_chunks")),
                topology_nnz_bits=_opt_int(d.get("topology_nnz_bits")),
                topology_dt_scale=_opt_float(d.get("topology_dt_scale")),
                topology_hash32=_opt_int(d.get("topology_hash32")),
                topology_density_fp=_opt_int(d.get("topology_density_fp")),
                topology_centroid_fp=_opt_int(d.get("topology_centroid_fp")),
                topology_variance_fp=_opt_int(d.get("topology_variance_fp")),
                topology_morton_key=_opt_int(d.get("topology_morton_key")),
                topology_commitment=_opt_int(d.get("topology_commitment")),
            )
        except FormatError:
            raise
        except Exception as exc:
            raise FormatError(
                f"Invalid EncodedBlock data: {exc}",
                original_error=str(exc),
            ) from exc


# ── EncodedStream ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EncodedStream:
    """Complete encoded payload with header, blocks, and integrity digest.

    ``to_bytes`` / ``from_bytes`` implement the binary wire format described
    in the module docstring.
    """

    blocks: tuple[EncodedBlock, ...]
    n_strands: int
    sector: str
    total_bytes: int
    checksum: bytes  # 32-byte BLAKE3 digest of original data
    version: int = _CURRENT_VERSION
    timestamp: int = field(default_factory=lambda: time.time_ns())
    metadata: dict[str, str] = field(default_factory=dict)

    # ── Wire format serialization ─────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """Serialize to the BRDC wire format."""
        buf = bytearray()

        # Magic + version
        buf.extend(_MAGIC)
        buf.extend(struct.pack(">H", self.version))

        # Header (msgpack)
        header: dict[str, object] = {
            "n_strands": self.n_strands,
            "sector": self.sector,
            "total_bytes": self.total_bytes,
            "checksum": self.checksum,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        header_bytes = msgpack.packb(header, use_bin_type=True)
        buf.extend(struct.pack(">I", len(header_bytes)))
        buf.extend(header_bytes)

        # Blocks
        buf.extend(struct.pack(">I", len(self.blocks)))
        for block in self.blocks:
            block_bytes = msgpack.packb(block.to_dict(), use_bin_type=True)
            buf.extend(struct.pack(">I", len(block_bytes)))
            buf.extend(block_bytes)

        # Trailing BLAKE3 digest over everything so far
        digest = blake3.blake3(bytes(buf)).digest()
        buf.extend(digest)

        return bytes(buf)

    def to_hdf5_bytes(self) -> bytes:
        """Serialize to a compact HDF5 container.

        This format stores packed generator bitstreams plus per-block attrs,
        reducing overhead versus msgpack list encoding.
        """
        try:
            import h5py  # type: ignore[import-untyped]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise FormatError("HDF5 serialization requires h5py") from exc

        bio = io.BytesIO()
        with h5py.File(bio, "w") as h5:
            h5.attrs["format"] = "BRDH5"
            h5.attrs["version"] = int(self.version)
            h5.attrs["n_strands"] = int(self.n_strands)
            h5.attrs["sector"] = self.sector
            h5.attrs["total_bytes"] = int(self.total_bytes)
            h5.attrs["timestamp"] = int(self.timestamp)

            h5.create_dataset("checksum", data=np.frombuffer(self.checksum, dtype=np.uint8))

            metadata_blob = json.dumps(self.metadata, separators=(",", ":")).encode("utf-8")
            h5.create_dataset("metadata_json", data=np.frombuffer(metadata_blob, dtype=np.uint8))

            n_blocks = len(self.blocks)
            blocks_group = h5.create_group("blocks")

            block_index = np.empty(n_blocks, dtype=np.int64)
            block_n_strands = np.empty(n_blocks, dtype=np.int16)
            block_sector: list[str] = [""] * n_blocks
            original_length = np.empty(n_blocks, dtype=np.int64)
            writhe = np.empty(n_blocks, dtype=np.int64)
            invariant_tier = np.empty(n_blocks, dtype=np.int8)

            jones_real = np.full(n_blocks, np.nan, dtype=np.float64)
            jones_imag = np.full(n_blocks, np.nan, dtype=np.float64)
            trace_real = np.full(n_blocks, np.nan, dtype=np.float64)
            trace_imag = np.full(n_blocks, np.nan, dtype=np.float64)

            topology_layer_index = np.full(n_blocks, -1, dtype=np.int64)
            topology_layer_n_chunks = np.full(n_blocks, -1, dtype=np.int64)
            topology_nnz_bits = np.full(n_blocks, -1, dtype=np.int64)
            topology_dt_scale = np.full(n_blocks, np.nan, dtype=np.float64)
            topology_hash32 = np.full(n_blocks, -1, dtype=np.int64)
            topology_density_fp = np.full(n_blocks, -1, dtype=np.int64)
            topology_centroid_fp = np.full(n_blocks, -1, dtype=np.int64)
            topology_variance_fp = np.full(n_blocks, -1, dtype=np.int64)
            topology_morton_key = np.zeros(n_blocks, dtype=np.uint64)
            topology_commitment = np.zeros(n_blocks, dtype=np.uint64)
            topology_morton_key_present = np.zeros(n_blocks, dtype=np.uint8)
            topology_commitment_present = np.zeros(n_blocks, dtype=np.uint8)

            bits_per_mag = np.empty(n_blocks, dtype=np.uint8)
            generator_count = np.empty(n_blocks, dtype=np.int32)
            decode_generator_count = np.full(n_blocks, -1, dtype=np.int32)

            gen_offsets = np.empty(n_blocks, dtype=np.int64)
            gen_sizes = np.empty(n_blocks, dtype=np.int32)
            dec_offsets = np.empty(n_blocks, dtype=np.int64)
            dec_sizes = np.full(n_blocks, 0, dtype=np.int32)

            gen_blob = bytearray()
            dec_blob = bytearray()

            for i, block in enumerate(self.blocks):
                block_index[i] = int(block.block_index)
                block_n_strands[i] = int(block.n_strands)
                block_sector[i] = block.sector
                original_length[i] = int(block.original_length)
                writhe[i] = int(block.writhe)
                invariant_tier[i] = int(block.invariant_tier)

                jones_real[i] = float(block.jones_real) if block.jones_real is not None else np.nan
                jones_imag[i] = float(block.jones_imag) if block.jones_imag is not None else np.nan
                trace_real[i] = float(block.trace_real) if block.trace_real is not None else np.nan
                trace_imag[i] = float(block.trace_imag) if block.trace_imag is not None else np.nan

                if block.topology_layer_index is not None:
                    topology_layer_index[i] = int(block.topology_layer_index)
                if block.topology_layer_n_chunks is not None:
                    topology_layer_n_chunks[i] = int(block.topology_layer_n_chunks)
                if block.topology_nnz_bits is not None:
                    topology_nnz_bits[i] = int(block.topology_nnz_bits)
                if block.topology_dt_scale is not None:
                    topology_dt_scale[i] = float(block.topology_dt_scale)
                if block.topology_hash32 is not None:
                    topology_hash32[i] = int(block.topology_hash32)
                if block.topology_density_fp is not None:
                    topology_density_fp[i] = int(block.topology_density_fp)
                if block.topology_centroid_fp is not None:
                    topology_centroid_fp[i] = int(block.topology_centroid_fp)
                if block.topology_variance_fp is not None:
                    topology_variance_fp[i] = int(block.topology_variance_fp)
                if block.topology_morton_key is not None:
                    topology_morton_key[i] = np.uint64(int(block.topology_morton_key))
                    topology_morton_key_present[i] = 1
                if block.topology_commitment is not None:
                    topology_commitment[i] = np.uint64(int(block.topology_commitment))
                    topology_commitment_present[i] = 1

                bmag = _bits_for_magnitude(block.n_strands)
                bits_per_mag[i] = int(bmag)
                packed = _pack_generators(block.generators, bmag)
                generator_count[i] = len(block.generators)
                gen_offsets[i] = len(gen_blob)
                gen_sizes[i] = len(packed)
                gen_blob.extend(packed)

                if block.decode_generators is not None:
                    packed_d = _pack_generators(block.decode_generators, bmag)
                    decode_generator_count[i] = len(block.decode_generators)
                    dec_offsets[i] = len(dec_blob)
                    dec_sizes[i] = len(packed_d)
                    dec_blob.extend(packed_d)
                else:
                    dec_offsets[i] = -1

            blocks_group.create_dataset("block_index", data=block_index)
            blocks_group.create_dataset("block_n_strands", data=block_n_strands)
            str_dtype = h5py.string_dtype(encoding="utf-8")
            blocks_group.create_dataset(
                "block_sector",
                data=np.asarray(block_sector, dtype=str_dtype),
                dtype=str_dtype,
            )
            blocks_group.create_dataset("original_length", data=original_length)
            blocks_group.create_dataset("writhe", data=writhe)
            blocks_group.create_dataset("invariant_tier", data=invariant_tier)
            blocks_group.create_dataset("jones_real", data=jones_real)
            blocks_group.create_dataset("jones_imag", data=jones_imag)
            blocks_group.create_dataset("trace_real", data=trace_real)
            blocks_group.create_dataset("trace_imag", data=trace_imag)

            blocks_group.create_dataset("topology_layer_index", data=topology_layer_index)
            blocks_group.create_dataset("topology_layer_n_chunks", data=topology_layer_n_chunks)
            blocks_group.create_dataset("topology_nnz_bits", data=topology_nnz_bits)
            blocks_group.create_dataset("topology_dt_scale", data=topology_dt_scale)
            blocks_group.create_dataset("topology_hash32", data=topology_hash32)
            blocks_group.create_dataset("topology_density_fp", data=topology_density_fp)
            blocks_group.create_dataset("topology_centroid_fp", data=topology_centroid_fp)
            blocks_group.create_dataset("topology_variance_fp", data=topology_variance_fp)
            blocks_group.create_dataset("topology_morton_key", data=topology_morton_key)
            blocks_group.create_dataset("topology_commitment", data=topology_commitment)
            blocks_group.create_dataset(
                "topology_morton_key_present",
                data=topology_morton_key_present,
            )
            blocks_group.create_dataset(
                "topology_commitment_present",
                data=topology_commitment_present,
            )

            blocks_group.create_dataset("bits_per_mag", data=bits_per_mag)
            blocks_group.create_dataset("generator_count", data=generator_count)
            blocks_group.create_dataset("decode_generator_count", data=decode_generator_count)
            blocks_group.create_dataset("gen_offset", data=gen_offsets)
            blocks_group.create_dataset("gen_size", data=gen_sizes)
            blocks_group.create_dataset("dec_offset", data=dec_offsets)
            blocks_group.create_dataset("dec_size", data=dec_sizes)

            blocks_group.create_dataset(
                "gen_blob",
                data=np.frombuffer(bytes(gen_blob), dtype=np.uint8),
                compression="gzip",
                compression_opts=9,
                shuffle=True,
            )
            blocks_group.create_dataset(
                "dec_blob",
                data=np.frombuffer(bytes(dec_blob), dtype=np.uint8),
                compression="gzip",
                compression_opts=9,
                shuffle=True,
            )

        return bio.getvalue()

    def to_hdf5_file(self, path: str) -> None:
        """Write stream to an HDF5 file path."""
        payload = self.to_hdf5_bytes()
        with Path(path).open("wb") as f:
            f.write(payload)

    @classmethod
    def from_hdf5_bytes(cls, data: bytes | bytearray | memoryview) -> EncodedStream:
        """Deserialize from the compact HDF5 container."""
        try:
            import h5py
        except Exception as exc:  # pragma: no cover - optional dependency
            raise FormatError("HDF5 deserialization requires h5py") from exc

        raw = bytes(data)
        with h5py.File(io.BytesIO(raw), "r") as h5:
            if str(h5.attrs.get("format", "")) != "BRDH5":
                raise FormatError("Not a BRDH5 stream")

            version = int(h5.attrs["version"])
            if version > _CURRENT_VERSION:
                raise VersionError(
                    f"Unsupported version {version} (max {_CURRENT_VERSION})",
                    version=version,
                )

            checksum = bytes(np.asarray(h5["checksum"], dtype=np.uint8).tobytes())
            metadata_blob = bytes(np.asarray(h5["metadata_json"], dtype=np.uint8).tobytes())
            metadata_obj = json.loads(metadata_blob.decode("utf-8")) if metadata_blob else {}
            metadata = {str(k): str(v) for k, v in metadata_obj.items()}

            blocks_group = h5["blocks"]
            block_index = np.asarray(blocks_group["block_index"], dtype=np.int64)
            block_n_strands = np.asarray(blocks_group["block_n_strands"], dtype=np.int16)
            block_sector = np.asarray(blocks_group["block_sector"], dtype=str)
            original_length = np.asarray(blocks_group["original_length"], dtype=np.int64)
            writhe = np.asarray(blocks_group["writhe"], dtype=np.int64)
            invariant_tier = np.asarray(blocks_group["invariant_tier"], dtype=np.int8)
            jones_real = np.asarray(blocks_group["jones_real"], dtype=np.float64)
            jones_imag = np.asarray(blocks_group["jones_imag"], dtype=np.float64)
            trace_real = np.asarray(blocks_group["trace_real"], dtype=np.float64)
            trace_imag = np.asarray(blocks_group["trace_imag"], dtype=np.float64)

            topology_layer_index = np.asarray(blocks_group["topology_layer_index"], dtype=np.int64)
            topology_layer_n_chunks = np.asarray(
                blocks_group["topology_layer_n_chunks"],
                dtype=np.int64,
            )
            topology_nnz_bits = np.asarray(blocks_group["topology_nnz_bits"], dtype=np.int64)
            topology_dt_scale = np.asarray(blocks_group["topology_dt_scale"], dtype=np.float64)
            topology_hash32 = np.asarray(blocks_group["topology_hash32"], dtype=np.int64)
            topology_density_fp = np.asarray(blocks_group["topology_density_fp"], dtype=np.int64)
            topology_centroid_fp = np.asarray(blocks_group["topology_centroid_fp"], dtype=np.int64)
            topology_variance_fp = np.asarray(blocks_group["topology_variance_fp"], dtype=np.int64)
            topology_morton_key = np.asarray(blocks_group["topology_morton_key"], dtype=np.uint64)
            topology_commitment = np.asarray(blocks_group["topology_commitment"], dtype=np.uint64)
            topology_morton_key_present = np.asarray(
                blocks_group["topology_morton_key_present"],
                dtype=np.uint8,
            )
            topology_commitment_present = np.asarray(
                blocks_group["topology_commitment_present"],
                dtype=np.uint8,
            )

            bits_per_mag = np.asarray(blocks_group["bits_per_mag"], dtype=np.uint8)
            generator_count = np.asarray(blocks_group["generator_count"], dtype=np.int32)
            decode_generator_count = np.asarray(
                blocks_group["decode_generator_count"],
                dtype=np.int32,
            )
            gen_offset = np.asarray(blocks_group["gen_offset"], dtype=np.int64)
            gen_size = np.asarray(blocks_group["gen_size"], dtype=np.int32)
            dec_offset = np.asarray(blocks_group["dec_offset"], dtype=np.int64)
            dec_size = np.asarray(blocks_group["dec_size"], dtype=np.int32)

            gen_blob = bytes(np.asarray(blocks_group["gen_blob"], dtype=np.uint8).tobytes())
            dec_blob = bytes(np.asarray(blocks_group["dec_blob"], dtype=np.uint8).tobytes())

            n_blocks = len(block_index)
            blocks: list[EncodedBlock] = []
            for i in range(n_blocks):
                bmag = int(bits_per_mag[i])
                g_count = int(generator_count[i])
                g_off = int(gen_offset[i])
                g_size = int(gen_size[i])
                generators = _unpack_generators(gen_blob[g_off : g_off + g_size], bmag, g_count)

                decode_count = int(decode_generator_count[i])
                decode_generators: list[int] | None = None
                if decode_count >= 0:
                    d_off = int(dec_offset[i])
                    d_size = int(dec_size[i])
                    decode_generators = _unpack_generators(
                        dec_blob[d_off : d_off + d_size],
                        bmag,
                        decode_count,
                    )

                block = EncodedBlock(
                    generators=generators,
                    n_strands=int(block_n_strands[i]),
                    sector=str(block_sector[i]),
                    writhe=int(writhe[i]),
                    block_index=int(block_index[i]),
                    original_length=int(original_length[i]),
                    invariant_tier=int(invariant_tier[i]),
                    jones_real=_nan_to_none(float(jones_real[i])),
                    jones_imag=_nan_to_none(float(jones_imag[i])),
                    trace_real=_nan_to_none(float(trace_real[i])),
                    trace_imag=_nan_to_none(float(trace_imag[i])),
                    decode_generators=decode_generators,
                    topology_layer_index=_opt_h5_int(topology_layer_index[i]),
                    topology_layer_n_chunks=_opt_h5_int(topology_layer_n_chunks[i]),
                    topology_nnz_bits=_opt_h5_int(topology_nnz_bits[i]),
                    topology_dt_scale=_opt_h5_float(topology_dt_scale[i]),
                    topology_hash32=_opt_h5_int(topology_hash32[i]),
                    topology_density_fp=_opt_h5_int(topology_density_fp[i]),
                    topology_centroid_fp=_opt_h5_int(topology_centroid_fp[i]),
                    topology_variance_fp=_opt_h5_int(topology_variance_fp[i]),
                    topology_morton_key=int(topology_morton_key[i])
                    if int(topology_morton_key_present[i])
                    else None,
                    topology_commitment=int(topology_commitment[i])
                    if int(topology_commitment_present[i])
                    else None,
                )
                blocks.append(block)

            return cls(
                version=version,
                blocks=tuple(blocks),
                n_strands=int(h5.attrs["n_strands"]),
                sector=str(h5.attrs["sector"]),
                total_bytes=int(h5.attrs["total_bytes"]),
                checksum=checksum,
                timestamp=int(h5.attrs["timestamp"]),
                metadata=metadata,
            )

    @classmethod
    def from_hdf5_file(cls, path: str) -> EncodedStream:
        """Read stream from an HDF5 file path."""
        with Path(path).open("rb") as f:
            return cls.from_hdf5_bytes(f.read())

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> EncodedStream:
        """Deserialize from the BRDC wire format.

        The trailing BLAKE3 digest is verified **before** any msgpack
        unpacking, as a defense-in-depth measure.

        Raises
        ------
        MagicMismatchError
            If the first 4 bytes are not ``b"BRDC"``.
        VersionError
            If the version field is unsupported.
        DigestError
            If the trailing digest does not match the payload.
        FormatError
            On any other structural problem.
        """
        raw = bytes(data)
        if len(raw) < _MIN_STREAM_LEN:
            raise FormatError(
                f"Stream too short: {len(raw)} bytes (minimum {_MIN_STREAM_LEN})",
                length=len(raw),
            )

        # ── Magic ─────────────────────────────────────────────────────────
        if raw[:4] != _MAGIC:
            raise MagicMismatchError(
                f"Expected magic b'BRDC', got {raw[:4]!r}",
                magic=raw[:4],
            )

        # ── Version ───────────────────────────────────────────────────────
        (version,) = struct.unpack(">H", raw[4:6])
        if version > _CURRENT_VERSION:
            raise VersionError(
                f"Unsupported version {version} (max {_CURRENT_VERSION})",
                version=version,
            )

        # ── Digest verification (before any msgpack) ─────────────────────
        payload = raw[:-_DIGEST_LEN]
        expected_digest = raw[-_DIGEST_LEN:]
        actual_digest = blake3.blake3(payload).digest()
        if actual_digest != expected_digest:
            raise DigestError("Trailing BLAKE3 digest mismatch")

        # ── Header ────────────────────────────────────────────────────────
        offset = 6
        (header_len,) = struct.unpack(">I", raw[offset : offset + 4])
        offset += 4
        if offset + header_len > len(payload):
            raise FormatError(
                "Header length exceeds available data",
                header_len=header_len,
            )
        header = msgpack.unpackb(raw[offset : offset + header_len], raw=False)
        offset += header_len

        # ── Blocks ────────────────────────────────────────────────────────
        if offset + 4 > len(payload):
            raise FormatError("Truncated block count field")
        (block_count,) = struct.unpack(">I", raw[offset : offset + 4])
        offset += 4

        blocks: list[EncodedBlock] = []
        for i in range(block_count):
            if offset + 4 > len(payload):
                raise FormatError(
                    f"Truncated block length prefix at block {i}",
                    block_index=i,
                )
            (block_len,) = struct.unpack(">I", raw[offset : offset + 4])
            offset += 4
            if offset + block_len > len(payload):
                raise FormatError(
                    f"Truncated block data at block {i}",
                    block_index=i,
                    block_len=block_len,
                )
            block_dict = msgpack.unpackb(raw[offset : offset + block_len], raw=False)
            blocks.append(EncodedBlock.from_dict(block_dict))
            offset += block_len

        return cls(
            version=version,
            blocks=tuple(blocks),
            n_strands=int(header["n_strands"]),
            sector=str(header["sector"]),
            total_bytes=int(header["total_bytes"]),
            checksum=bytes(header["checksum"]),
            timestamp=int(header["timestamp"]),
            metadata={str(k): str(v) for k, v in header.get("metadata", {}).items()},
        )


# ── Helpers ───────────────────────────────────────────────────────────────


def _opt_float(value: object) -> float | None:
    """Coerce to ``float`` or return ``None``."""
    if value is None:
        return None
    return float(value)  # type: ignore[arg-type]


def _opt_int(value: object) -> int | None:
    """Coerce to ``int`` or return ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        return int(value)
    if isinstance(value, Real):
        return int(float(value))
    raise TypeError("Expected int-like value")


def _opt_int_list(value: object) -> list[int] | None:
    """Coerce to ``list[int]`` or return ``None``."""
    if value is None:
        return None
    return [int(v) for v in value]  # type: ignore[attr-defined]


def _bits_for_magnitude(n_strands: int) -> int:
    return max(1, int(max(n_strands - 1, 1)).bit_length())


def _pack_generators(generators: list[int], bits_per_mag: int) -> bytes:
    width = bits_per_mag + 1
    bitbuf = 0
    bitcount = 0
    out = bytearray()
    mag_mask = (1 << bits_per_mag) - 1

    for g in generators:
        sign = 1 if g < 0 else 0
        mag = abs(int(g)) & mag_mask
        value = (sign << bits_per_mag) | mag
        bitbuf = (bitbuf << width) | value
        bitcount += width
        while bitcount >= 8:
            bitcount -= 8
            out.append((bitbuf >> bitcount) & 0xFF)
            bitbuf &= (1 << bitcount) - 1 if bitcount else 0

    if bitcount:
        out.append((bitbuf << (8 - bitcount)) & 0xFF)
    return bytes(out)


def _unpack_generators(data: bytes, bits_per_mag: int, count: int) -> list[int]:
    width = bits_per_mag + 1
    mag_mask = (1 << bits_per_mag) - 1
    bitbuf = 0
    bitcount = 0
    out: list[int] = []

    for b in data:
        bitbuf = (bitbuf << 8) | int(b)
        bitcount += 8
        while bitcount >= width and len(out) < count:
            bitcount -= width
            value = (bitbuf >> bitcount) & ((1 << width) - 1)
            bitbuf &= (1 << bitcount) - 1 if bitcount else 0
            sign = (value >> bits_per_mag) & 1
            mag = value & mag_mask
            if mag == 0:
                mag = 1
            out.append(-mag if sign else mag)

    if len(out) != count:
        raise FormatError("Packed generator payload truncated", expected=count, actual=len(out))
    return out


def _set_opt_int_attr(group: Any, name: str, value: int | None) -> None:
    group.attrs[name] = int(value) if value is not None else -1


def _set_opt_float_attr(group: Any, name: str, value: float | None) -> None:
    group.attrs[name] = float(value) if value is not None else np.nan


def _opt_h5_int(value: object) -> int | None:
    if isinstance(value, str):
        v = int(value)
    elif isinstance(value, Real):
        v = int(float(value))
    else:
        raise TypeError("Expected int-like HDF5 value")
    return None if v < 0 else v


def _opt_h5_float(value: object) -> float | None:
    if not isinstance(value, Real | str):
        raise TypeError("Expected float-like HDF5 value")
    v = float(value)
    return None if np.isnan(v) else v


def _nan_to_none(value: float) -> float | None:
    return None if np.isnan(value) else value


def validate_reconstructive_metadata(metadata: Mapping[str, str]) -> None:
    """Validate required metadata keys for reconstructive mode.

    This helper is intentionally strict so Phase A contract violations are
    caught early before manifold/fixed-point stages are integrated.
    """

    if str(metadata.get("preprocessing_mode", "")) != "reconstructive":
        raise FormatError(
            "reconstructive metadata requires preprocessing_mode='reconstructive'",
        )

    missing = sorted(_RECONSTRUCTIVE_REQUIRED_METADATA - set(metadata.keys()))
    if missing:
        raise FormatError(
            f"Missing reconstructive metadata keys: {missing}",
            missing_keys=missing,
        )

    domain_kind = str(metadata.get("domain_kind", ""))
    if domain_kind not in {"text", "json", "logs"}:
        raise FormatError(
            "domain_kind must be one of: text, json, logs",
            domain_kind=domain_kind,
        )

    try:
        bin_count = int(str(metadata.get("bin_count", "")))
    except ValueError as exc:
        raise FormatError("bin_count must be an integer") from exc
    if bin_count < 2:
        raise FormatError("bin_count must be >= 2", bin_count=bin_count)


def build_reconstructive_payload_metadata(metadata: Mapping[str, str]) -> str:
    """Build compact reconstructive payload JSON for schema metadata.

    The payload is intentionally compact and versioned so dedicated decode
    routing can validate model/hash/diagnostic commitments in one place.
    """
    validate_reconstructive_metadata(metadata)

    missing = sorted(
        {
            "manifold_profile_id",
            "manifold_state_hash",
            "reconstructive_graph_hash",
            "km_residual_max",
            "km_residual_mean",
            "km_iters_mean",
            "km_valid_ratio",
            "km_seed_vector",
            "km_kappa",
            "km_eta",
            "km_tol",
            "km_max_iter",
            "km_threshold_residual_max",
            "km_threshold_valid_ratio",
            "reconstructive_program_type",
            "reconstructive_program_payload",
        }
        - set(metadata.keys())
    )
    if missing:
        raise FormatError(
            f"Missing reconstructive payload source keys: {missing}",
            missing_keys=missing,
        )

    payload: dict[str, str] = {
        "model_id": "frequency-manifold",
        "model_version": "1",
        "domain_kind": str(metadata["domain_kind"]),
        "tokenizer_id": str(metadata["tokenizer_id"]),
        "tokenizer_version": str(metadata["tokenizer_version"]),
        "bin_count": str(metadata["bin_count"]),
        "bin_table_hash": str(metadata["bin_table_hash"]),
        "vocab_hash": str(metadata["vocab_hash"]),
        "normalization_profile_id": str(metadata["normalization_profile_id"]),
        "normalization_profile_hash": str(metadata["normalization_profile_hash"]),
        "manifold_profile_id": str(metadata["manifold_profile_id"]),
        "manifold_state_hash": str(metadata["manifold_state_hash"]),
        "reconstructive_graph_hash": str(metadata["reconstructive_graph_hash"]),
        "km_residual_max": str(metadata["km_residual_max"]),
        "km_residual_mean": str(metadata["km_residual_mean"]),
        "km_iters_mean": str(metadata["km_iters_mean"]),
        "km_valid_ratio": str(metadata["km_valid_ratio"]),
        "km_seed_vector": str(metadata["km_seed_vector"]),
        "km_kappa": str(metadata["km_kappa"]),
        "km_eta": str(metadata["km_eta"]),
        "km_tol": str(metadata["km_tol"]),
        "km_max_iter": str(metadata["km_max_iter"]),
        "km_threshold_residual_max": str(metadata["km_threshold_residual_max"]),
        "km_threshold_valid_ratio": str(metadata["km_threshold_valid_ratio"]),
        "reconstructive_program_type": str(metadata["reconstructive_program_type"]),
        "reconstructive_program_payload": str(metadata["reconstructive_program_payload"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def parse_reconstructive_payload_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    """Parse compact reconstructive payload JSON from stream metadata."""
    payload_blob = str(metadata.get(_RECONSTRUCTIVE_PAYLOAD_KEY, ""))
    if not payload_blob:
        raise FormatError(f"Missing {_RECONSTRUCTIVE_PAYLOAD_KEY} metadata entry")

    try:
        payload_obj = json.loads(payload_blob)
    except json.JSONDecodeError as exc:
        raise FormatError(
            f"Invalid {_RECONSTRUCTIVE_PAYLOAD_KEY} JSON",
            reason=str(exc),
        ) from exc

    if not isinstance(payload_obj, dict):
        raise FormatError(
            f"{_RECONSTRUCTIVE_PAYLOAD_KEY} must decode to an object",
        )

    return {str(k): str(v) for k, v in payload_obj.items()}


def validate_reconstructive_payload_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    """Validate compact reconstructive payload metadata and return parsed payload."""
    payload = parse_reconstructive_payload_metadata(metadata)

    missing = sorted(_RECONSTRUCTIVE_PAYLOAD_REQUIRED_KEYS - set(payload.keys()))
    if missing:
        raise FormatError(
            f"Missing reconstructive payload keys: {missing}",
            missing_keys=missing,
        )

    if payload["model_id"] != "frequency-manifold" or payload["model_version"] != "1":
        raise FormatError(
            "Unsupported reconstructive payload model/version",
            model_id=payload["model_id"],
            model_version=payload["model_version"],
        )

    for key_name in (
        "tokenizer_id",
        "tokenizer_version",
        "domain_kind",
        "bin_count",
        "bin_table_hash",
        "vocab_hash",
        "normalization_profile_id",
        "normalization_profile_hash",
        "km_seed_vector",
        "km_kappa",
        "km_eta",
        "km_tol",
        "km_max_iter",
        "km_threshold_residual_max",
        "km_threshold_valid_ratio",
        "reconstructive_program_type",
        "reconstructive_program_payload",
    ):
        # Some large payload mirrors may be omitted from top-level metadata for
        # compactness; if present they must exactly match the payload contract.
        if key_name not in metadata:
            continue
        if str(metadata.get(key_name, "")) != payload[key_name]:
            raise FormatError(
                f"Reconstructive payload mismatch for '{key_name}'",
                key=key_name,
            )

    try:
        km_valid_ratio = float(payload["km_valid_ratio"])
        km_residual_max = float(payload["km_residual_max"])
        km_residual_mean = float(payload["km_residual_mean"])
        km_iters_mean = float(payload["km_iters_mean"])
    except ValueError as exc:
        raise FormatError("Reconstructive payload diagnostics must be numeric") from exc

    if not (0.0 <= km_valid_ratio <= 1.0):
        raise FormatError("km_valid_ratio must be within [0,1]", km_valid_ratio=km_valid_ratio)
    if km_residual_max < 0.0 or km_residual_mean < 0.0 or km_iters_mean < 0.0:
        raise FormatError("Reconstructive payload diagnostics must be non-negative")

    return payload


def compute_reconstructive_commitment(
    payload_json: str,
    blocks: tuple[EncodedBlock, ...],
) -> str:
    """Compute deterministic reconstructive commitment over payload and blocks."""
    hasher = blake3.blake3()
    hasher.update(payload_json.encode("utf-8"))

    for block in sorted(blocks, key=lambda b: b.block_index):
        header = (
            f"{block.block_index}|{block.original_length}|{block.n_strands}|"
            f"{len(block.generators)}|{block.writhe}|{block.invariant_tier}|"
        ).encode("ascii")
        hasher.update(header)
        hasher.update(",".join(str(g) for g in block.generators).encode("ascii"))

    return hasher.hexdigest()


def validate_reconstructive_commitment_metadata(
    metadata: Mapping[str, str],
    blocks: tuple[EncodedBlock, ...],
) -> None:
    """Validate reconstructive commitment metadata against stream blocks."""
    payload_json = str(metadata.get(_RECONSTRUCTIVE_PAYLOAD_KEY, ""))
    if not payload_json:
        raise FormatError(f"Missing {_RECONSTRUCTIVE_PAYLOAD_KEY} metadata entry")

    stored = str(metadata.get("reconstructive_commitment", ""))
    if not stored:
        raise FormatError("Missing reconstructive_commitment metadata entry")

    expected = compute_reconstructive_commitment(payload_json, blocks)
    if stored != expected:
        raise FormatError(
            "Reconstructive commitment mismatch",
            expected=expected,
            actual=stored,
        )
