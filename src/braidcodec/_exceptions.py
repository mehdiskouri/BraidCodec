"""BraidCodec exception hierarchy.

All public exceptions inherit from :class:`BraidCodecError`, which carries an
optional ``context`` dict for structured diagnostics.  Leaf classes are grouped
by subsystem (format parsing, key management, integrity verification, encoding,
compression) so that callers can catch at the appropriate granularity.
"""

from __future__ import annotations

# ── Base ──────────────────────────────────────────────────────────────────


class BraidCodecError(Exception):
    """Root exception for the entire BraidCodec library.

    Parameters
    ----------
    message:
        Human-readable error description.
    **context:
        Arbitrary key/value pairs attached as ``self.context`` for
        programmatic inspection.
    """

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.context: dict[str, object] = context


# ── Format errors ─────────────────────────────────────────────────────────


class FormatError(BraidCodecError):
    """Wire-format parsing failure."""


class MagicMismatchError(FormatError):
    """Stream magic bytes do not match ``b"BRDC"``."""


class VersionError(FormatError):
    """Unsupported wire-format version."""


class DigestError(FormatError):
    """Trailing BLAKE3 digest does not match payload."""


# ── Key errors ────────────────────────────────────────────────────────────


class BraidKeyError(BraidCodecError):
    """Key management failure."""


class KeyMismatchError(BraidKeyError):
    """Supplied key does not match the key used for encoding."""


class KeyValidationError(BraidKeyError):
    """Key parameters are invalid or out of range."""


# ── Integrity errors ──────────────────────────────────────────────────────


class IntegrityError(BraidCodecError):
    """Integrity verification failure (any channel)."""


class WritheError(IntegrityError):
    """Writhe pre-check mismatch."""


class JonesError(IntegrityError):
    """Jones polynomial recomputation mismatch."""


class TraceError(IntegrityError):
    """Matrix-trace invariant mismatch."""


class FermionError(IntegrityError):
    """Fermion occupation bound violation."""


class ChecksumError(IntegrityError):
    """BLAKE3 data checksum mismatch."""


# ── Encoding errors ───────────────────────────────────────────────────────


class EncodingError(BraidCodecError):
    """Failure during the encode pipeline."""


class ChunkError(EncodingError):
    """Chunker-level encoding failure."""


class ContractionError(EncodingError):
    """Tensor contraction failure."""


# ── Compression errors ────────────────────────────────────────────────────


class CompressionError(BraidCodecError):
    """Failure during topological compression."""


class RewriteVerificationError(CompressionError):
    """Post-rewrite equivalence check failed."""
