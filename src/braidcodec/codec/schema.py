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

import struct
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

import blake3
import msgpack

from braidcodec._exceptions import (
    DigestError,
    FormatError,
    MagicMismatchError,
    VersionError,
)

# ── Constants ─────────────────────────────────────────────────────────────

_MAGIC: bytes = b"BRDC"
_CURRENT_VERSION: int = 1
_DIGEST_LEN: int = 32
# magic(4) + version(2) + header_len(4) + block_count(4) + digest(32) = 46
_MIN_STREAM_LEN: int = 46


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
