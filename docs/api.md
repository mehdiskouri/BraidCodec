# API Reference

## Public Symbols

All symbols are importable from the top-level `braidcodec` package:

```python
from braidcodec import (
    BraidKey, EncodedBlock, EncodedStream, VerificationResult,
    __version__, compress, decode, encode,
    key_from_bytes, key_to_bytes, keygen, verify,
)
```

---

## Key Management

### `keygen`

```python
def keygen(
    sector: str = "TSR",
    n_strands: int = 4,
    theta_offset: float | None = None,
) -> BraidKey
```

Generate a new topological key.

| Parameter | Description |
|-----------|-------------|
| `sector` | Anyon model: `"Identity"`, `"TSR"`, `"Ising"`, `"Fibonacci"`, `"SU2k2"` |
| `n_strands` | Number of braid strands (≥ 2). Higher = slower but more secure. |
| `theta_offset` | Secret phase offset. If `None`, sampled uniformly from $[0, 2\pi)$. |

**Returns**: `BraidKey` — frozen dataclass with `sector`, `n_strands`, `theta_offset`, `key_id`.

### `key_to_bytes`

```python
def key_to_bytes(key: BraidKey) -> bytes
```

Serialize a `BraidKey` to 50 bytes (wire format: `BRDK` + version + sector + n_strands + theta + BLAKE3).

### `key_from_bytes`

```python
def key_from_bytes(data: bytes) -> BraidKey
```

Deserialize bytes back to a `BraidKey`. Raises `FormatError` or `KeyValidationError` on invalid input.

---

## Encoding & Decoding

### `encode`

```python
def encode(
    data: bytes,
    key: BraidKey,
    *,
    generators_per_block: int = 32,
    max_workers: int | None = None,
    preprocessing_mode: str = "topology",
    reconstructive_domain: str | None = None,
    reconstructive_compact_transport: str = "enabled",
) -> EncodedStream
```

Encode arbitrary bytes into a topological braid stream.

| Parameter | Description |
|-----------|-------------|
| `data` | Input bytes to encode (any length). |
| `key` | `BraidKey` from `keygen`. |
| `generators_per_block` | Generators per block. Lower = faster (tier 1-2), higher = more compact. |
| `max_workers` | Parallel encoding workers. `None` = auto (capped at 8). |
| `preprocessing_mode` | `"topology"` (default), `"legacy"`, or `"reconstructive"`. |
| `reconstructive_domain` | Optional domain override for reconstructive mode: `"text"`, `"json"`, or `"logs"`. |
| `reconstructive_compact_transport` | Compact policy for reconstructive mode: `"enabled"` (commitment-validated `ps1.*`) or `"lean"` (checksum-authoritative `ps2.*`). |

Notes for `reconstructive` mode:
- Compact transport metadata uses short keys: `rt` (transport code), `rpb` (program sidechannel), optional `rc3` (commitment for `enabled`), and optional `ra1` (audit sidecar).
- `enabled` validates compact commitment v3 (`rc3`) during decode/verify.
- `lean` intentionally omits reconstructive commitment metadata and relies on checksum authority.

**Returns**: `EncodedStream` with blocks, invariants, and checksum.

### `decode`

```python
def decode(
    stream: EncodedStream,
    key: BraidKey,
    *,
    verify: bool = True,
) -> bytes
```

Decode an `EncodedStream` back to the original bytes.

| Parameter | Description |
|-----------|-------------|
| `stream` | Encoded payload from `encode` or `EncodedStream.from_bytes`. |
| `key` | The same `BraidKey` used during encoding. |
| `verify` | If `True`, recompute and verify invariants per block. Writhe is always checked. |

**Returns**: Original bytes.

**Raises**: `WritheError`, `JonesError`, `TraceError`, `KeyMismatchError`, `ChecksumError`.

---

## Compression

### `compress`

```python
def compress(
    stream: EncodedStream,
    key: BraidKey,
    *,
    level: int = 1,
    max_rewrites: int = 100,
) -> EncodedStream
```

Apply topological compression to reduce generator count while preserving the braid's equivalence class.

| Parameter | Description |
|-----------|-------------|
| `level` | 0 = inverse cancel, 1 = + far-commutativity, 2 = + Yang-Baxter BFS |
| `max_rewrites` | Maximum rewrite steps per block (level 2). |

**Returns**: New `EncodedStream` with shortened generators and stored `decode_generators`.

---

## Integrity Verification

### `verify`

```python
def verify(
    stream: EncodedStream,
    key: BraidKey,
    *,
    fermion_check: bool = False,
    topology_check: bool = False,
) -> VerificationResult
```

Run 5+1-channel integrity verification on an encoded stream.

| Parameter | Description |
|-----------|-------------|
| `fermion_check` | Enable fermion occupation constraint checking (slower). |
| `topology_check` | Enable topology metadata recomputation and commitment checks for topology-mode streams. |

**Returns**: `VerificationResult` with per-channel pass/fail.

---

## Data Classes

### `BraidKey`

```python
@dataclass(frozen=True, slots=True)
class BraidKey:
    sector: str          # Anyon model name
    n_strands: int       # Number of braid strands
    theta_offset: float  # Secret phase parameter
    key_id: str          # 32 hex chars, derived from BLAKE3
```

### `EncodedBlock`

```python
@dataclass(frozen=True, slots=True)
class EncodedBlock:
    generators: list[int]              # Signed braid generators
    n_strands: int                     # Number of braid strands
    sector: str                        # Anyon model name
    writhe: int                        # Sum of generator signs
    block_index: int                   # Position in the stream
    original_length: int               # Original byte count for this block
    invariant_tier: int                # 1, 2, or 3
    jones_real: float | None = None    # Re(Jones) — tier 2
    jones_imag: float | None = None    # Im(Jones) — tier 2
    trace_real: float | None = None    # Re(trace) — tier 3
    trace_imag: float | None = None    # Im(trace) — tier 3
    decode_generators: list[int] | None = None  # Original generators (if compressed)
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
```

### `EncodedStream`

```python
@dataclass(frozen=True, slots=True)
class EncodedStream:
    blocks: tuple[EncodedBlock, ...]
    n_strands: int
    sector: str
    total_bytes: int                 # Total original data length
    checksum: bytes                  # 32-byte BLAKE3 digest
    version: int = 2
    timestamp: int = ...             # time.time_ns()
    metadata: dict[str, str] = {}    # User-defined metadata
```

Methods: `to_bytes() -> bytes`, `from_bytes(data: bytes) -> EncodedStream` (class method).

### `VerificationResult`

```python
@dataclass(frozen=True, slots=True)
class VerificationResult:
    valid: bool                # Overall pass/fail
    structural_passed: bool    # Magic, version, schema
    writhe_passed: bool        # Per-block writhe
    invariant_passed: bool     # Jones or trace
    fermion_passed: bool | None  # None if not checked
    topology_passed: bool | None  # None if not checked
    checksum_passed: bool      # BLAKE3
    failed_blocks: tuple[int, ...]  # Indices of failing blocks
    details: tuple[str, ...]   # Human-readable channel summaries
```

---

## Exception Hierarchy

```
BraidCodecError
├── FormatError
│   ├── MagicMismatchError
│   ├── VersionError
│   └── DigestError
├── BraidKeyError
│   ├── KeyMismatchError
│   └── KeyValidationError
├── IntegrityError
│   ├── WritheError
│   ├── JonesError
│   ├── TraceError
│   ├── FermionError
│   └── ChecksumError
├── EncodingError
│   └── ChunkError
└── CompressionError
    └── RewriteVerificationError
```

All exceptions carry a `context: dict[str, object]` attribute with structured debugging info.

---

## CLI Commands

Entry point: `braidcodec` (requires `pip install braidcodec[cli]`).

| Command | Description | Exit codes |
|---------|-------------|------------|
| `braidcodec keygen -o KEY` | Generate key, write to file | 0, 4 |
| `braidcodec encode INPUT -o OUTPUT --key KEY --preprocessing-mode topology|legacy|reconstructive [--reconstructive-domain text|json|logs] [--reconstructive-compact-transport enabled|lean]` | Encode file | 0, 3, 4 |
| `braidcodec decode INPUT -o OUTPUT --key KEY` | Decode file | 0, 1, 2, 3, 4 |
| `braidcodec verify INPUT --key KEY [--fermion-check] [--topology-check] [--diagnostics]` | Verify integrity | 0, 1, 2, 3, 4 |
| `braidcodec inspect INPUT` | Show stream metadata | 0, 3, 4 |
| `braidcodec benchmark` | Run encode/decode/verify timing | 0 |

All commands support `--verbose` / `--quiet` (mutually exclusive).

**Exit codes**: 0 = OK, 1 = integrity failure, 2 = key mismatch, 3 = format error, 4 = I/O error.
