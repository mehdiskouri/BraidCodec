"""BraidKey generation and serialization.

A ``BraidKey`` parametrises the R-matrix via a ``theta_offset`` that shifts
the base sector theta, making the braid encoding key-dependent.  The key is
serializable to a compact 50-byte binary format (``key_to_bytes`` /
``key_from_bytes``).
"""

from __future__ import annotations

import math
import secrets
import struct
from dataclasses import dataclass

import blake3
import numpy as np

from braidcodec._exceptions import FormatError, KeyValidationError
from braidcodec.algebra.braid_equations import VALID_SECTORS, get_sector_r_matrix
from braidcodec.algebra.tsr_constants import C_CONSTANT

# ── Sector ↔ enum mappings ────────────────────────────────────────────────

_SECTOR_TO_ENUM: dict[str, int] = {
    "Identity": 0,
    "TSR": 1,
    "Ising": 2,
    "Fibonacci": 3,
    "SU2k2": 4,
}
_ENUM_TO_SECTOR: dict[int, str] = {v: k for k, v in _SECTOR_TO_ENUM.items()}

# ── Key wire format constants ─────────────────────────────────────────────

_KEY_MAGIC: bytes = b"BRDK"
_KEY_VERSION: int = 1
_KEY_LENGTH: int = 50  # 4 magic + 1 version + 1 sector + 4 n_strands + 8 theta + 32 digest
_KEY_HEADER_LEN: int = 18  # everything before the digest
_KEY_STRUCT_FMT: str = ">4sBBId"

_TWO_PI: float = 2.0 * math.pi


# ── Base theta per sector ─────────────────────────────────────────────────


def _base_sector_theta(sector: str) -> float:
    """Return the base theta for a given sector."""
    if sector in ("Identity", "TSR"):
        return math.pi * C_CONSTANT
    if sector == "Ising":
        return math.pi / 4
    if sector == "Fibonacci":
        return 4 * math.pi / 5
    if sector == "SU2k2":
        return math.pi / 2
    msg = f"Unknown sector '{sector}'"
    raise KeyValidationError(msg, sector=sector)


# ── BraidKey ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BraidKey:
    """Immutable braid encoding key.

    Parameters
    ----------
    sector:
        Anyon sector name.
    n_strands:
        Number of braid strands (≥ 2).
    theta_offset:
        Angle offset in ``[0, 2π)`` added to the base sector theta.
    key_id:
        Deterministic 32-hex-char identifier derived from key parameters.
    """

    sector: str
    n_strands: int
    theta_offset: float
    key_id: str

    def __post_init__(self) -> None:
        if self.sector not in VALID_SECTORS:
            raise KeyValidationError(
                f"Unknown sector '{self.sector}'. Valid: {sorted(VALID_SECTORS)}",
                sector=self.sector,
            )
        if self.n_strands < 2:
            raise KeyValidationError(
                f"n_strands must be >= 2, got {self.n_strands}",
                n_strands=self.n_strands,
            )
        if not (0.0 <= self.theta_offset < _TWO_PI):
            raise KeyValidationError(
                f"theta_offset must be in [0, 2\u03c0), got {self.theta_offset}",
                theta_offset=self.theta_offset,
            )
        if len(self.key_id) != 32 or not all(c in "0123456789abcdef" for c in self.key_id):
            raise KeyValidationError(
                f"key_id must be 32 lowercase hex chars, got {self.key_id!r}",
                key_id=self.key_id,
            )

    @property
    def theta_effective(self) -> float:
        """Base sector theta plus the key offset."""
        return _base_sector_theta(self.sector) + self.theta_offset

    @property
    def sector_params(self) -> dict[str, float]:
        """Sector params dict suitable for ``BraidEquation._sector_params``."""
        return {"theta": self.theta_effective}


# ── Key generation ────────────────────────────────────────────────────────


def _compute_key_id(sector: str, n_strands: int, theta_offset: float) -> str:
    """Deterministic 32-hex key_id from parameters."""
    payload = f"{sector}:{n_strands}:{theta_offset!r}".encode()
    return blake3.blake3(payload).hexdigest()[:32]


def _validate_keyed_unitarity(sector: str, theta_eff: float) -> None:
    """Verify the keyed R-matrix is unitary."""
    r_matrix = get_sector_r_matrix(sector, inverse=False, theta_override=theta_eff)
    product = r_matrix @ r_matrix.conj().T
    deviation = float(np.linalg.norm(product - np.eye(4, dtype=np.complex128)))
    if deviation > 1e-10:
        raise KeyValidationError(
            f"Keyed R-matrix unitarity check failed: deviation={deviation}",
            deviation=deviation,
        )


def keygen(
    sector: str = "TSR",
    n_strands: int = 4,
    theta_offset: float | None = None,
) -> BraidKey:
    """Generate a ``BraidKey``.

    Parameters
    ----------
    sector:
        Anyon sector (default ``"TSR"``).
    n_strands:
        Strand count (default 4).
    theta_offset:
        Explicit offset in ``[0, 2\u03c0)``.  If ``None``, generated from
        ``secrets.token_bytes(32)`` with NaN/inf guard.

    Returns
    -------
    BraidKey
        A validated key whose R-matrix is verified unitary.
    """
    if theta_offset is None:
        while True:
            raw = secrets.token_bytes(32)
            (value,) = struct.unpack(">d", raw[:8])
            if math.isfinite(value):
                theta_offset = value % _TWO_PI
                break

    key_id = _compute_key_id(sector, n_strands, theta_offset)
    key = BraidKey(
        sector=sector,
        n_strands=n_strands,
        theta_offset=theta_offset,
        key_id=key_id,
    )

    _validate_keyed_unitarity(sector, key.theta_effective)

    return key


# ── Key serialization ─────────────────────────────────────────────────────


def key_to_bytes(key: BraidKey) -> bytes:
    """Serialize a ``BraidKey`` to 50 bytes.

    Layout::

        BRDK (4) + version uint8 (1) + sector_enum uint8 (1)
        + n_strands uint32 (4) + theta_offset float64 (8) + BLAKE3 (32)
    """
    sector_enum = _SECTOR_TO_ENUM[key.sector]
    header = struct.pack(
        _KEY_STRUCT_FMT,
        _KEY_MAGIC,
        _KEY_VERSION,
        sector_enum,
        key.n_strands,
        key.theta_offset,
    )
    digest = blake3.blake3(header).digest()
    return header + digest


def key_from_bytes(data: bytes) -> BraidKey:
    """Deserialize a ``BraidKey`` from 50 bytes.

    Raises
    ------
    FormatError
        On bad magic, wrong length, or digest mismatch.
    KeyValidationError
        If reconstructed key fails validation.
    """
    if len(data) != _KEY_LENGTH:
        raise FormatError(
            f"Expected {_KEY_LENGTH} bytes, got {len(data)}",
            length=len(data),
        )

    if data[:4] != _KEY_MAGIC:
        raise FormatError(
            f"Expected magic b'BRDK', got {data[:4]!r}",
            magic=data[:4],
        )

    header = data[:_KEY_HEADER_LEN]
    expected_digest = data[_KEY_HEADER_LEN:]
    actual_digest = blake3.blake3(header).digest()
    if actual_digest != expected_digest:
        raise FormatError("Key digest mismatch")

    _, version, sector_enum, n_strands, theta_offset = struct.unpack(_KEY_STRUCT_FMT, header)

    if version != _KEY_VERSION:
        raise FormatError(
            f"Unsupported key version {version}",
            version=version,
        )

    if sector_enum not in _ENUM_TO_SECTOR:
        raise FormatError(
            f"Unknown sector enum {sector_enum}",
            sector_enum=sector_enum,
        )

    sector = _ENUM_TO_SECTOR[sector_enum]
    return keygen(sector=sector, n_strands=int(n_strands), theta_offset=float(theta_offset))
