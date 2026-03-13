# BraidCodec — Product Requirements Document

## 1. Executive Summary

BraidCodec is a Python 3.13 topological data codec that encodes arbitrary binary data as braid group equations, producing a representation that is simultaneously encrypted, compressed, and integrity-verified through the mathematics of knot invariants. The system exploits three properties that emerge naturally from braid group topology: confidentiality from sector-parameterized R-matrices acting as trapdoor functions, compression from topological equivalence class collapse, and tamper detection from Yang-Baxter constraint violation.

The project targets PyPI distribution as `braidcodec` with a public GitHub repository containing architecture diagrams, benchmarks, and documentation — but no source code for the underlying algebra engine beyond what the public API exposes. The algebra modules (`tsr_constants`, `braid_equations`, `yang_baxter`, `fermion_bounds`) are verified and complete. This PRD covers everything from the serialization engine through CI/CD pipeline.

---

## 2. Problem Statement

Conventional data pipelines treat encryption, compression, and integrity verification as three separate concerns requiring three separate tools (e.g., AES + zstd + SHA-256). Each adds overhead, each has its own failure modes, and the composition of the three is not guaranteed to be coherent — you can compress corrupted data, encrypt without integrity checks, or verify a checksum on the wrong ciphertext.

BraidCodec unifies all three by encoding data into a mathematical structure (braid groups) where confidentiality, compression, and integrity are not bolted on but are inherent properties of the representation. The Jones polynomial of a braid is a topological invariant: it doesn't change under valid transformations (integrity), it maps many distinct braid words to the same value (compression), and it cannot be computed without knowing the R-matrix parameters (confidentiality).

---

## 3. Target Audience

**Primary — hiring managers and technical reviewers** evaluating Mehdi's portfolio for backend/ML engineering roles. They need to see: clean architecture, real mathematical depth, working benchmarks, professional packaging, and CI discipline.

**Secondary — applied mathematicians and quantum computing researchers** interested in topological data encoding. The README and docs should be accessible to someone who knows braid groups but not TSR, and vice versa.

**Tertiary — developers** who might actually use the library for experimental data encoding, integrity verification, or as a teaching tool for topological quantum computation concepts.

---

## 4. Core Architecture

### 4.1 Package Structure

```
braidcodec/
├── pyproject.toml
├── LICENSE                         (MIT)
├── README.md
├── CHANGELOG.md
├── .github/
│   └── workflows/
│       ├── ci.yml                  (lint + test + coverage on PR/push)
│       ├── release.yml             (PyPI publish on tag)
│       └── benchmark.yml           (weekly regression benchmarks)
├── src/
│   └── braidcodec/
│       ├── __init__.py             (public API surface)
│       ├── py.typed                (PEP 561 marker)
│       ├── _version.py             (single-source version)
│       ├── algebra/
│       │   ├── __init__.py
│       │   ├── tsr_constants.py
│       │   ├── braid_equations.py
│       │   ├── yang_baxter.py
│       │   └── fermion_bounds.py
│       ├── codec/
│       │   ├── __init__.py
│       │   ├── chunker.py
│       │   ├── encoder.py
│       │   ├── decoder.py
│       │   ├── schema.py
│       │   └── compressor.py
│       ├── crypto/
│       │   ├── __init__.py
│       │   ├── keys.py
│       │   └── integrity.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py
│       └── _types.py
├── tests/
│   ├── conftest.py
│   ├── algebra/
│   │   ├── test_tsr_constants.py
│   │   ├── test_braid_equations.py
│   │   ├── test_yang_baxter.py
│   │   └── test_fermion_bounds.py
│   ├── codec/
│   │   ├── test_chunker.py
│   │   ├── test_encoder.py
│   │   ├── test_decoder.py
│   │   ├── test_schema.py
│   │   ├── test_compressor.py
│   │   └── test_roundtrip.py
│   ├── crypto/
│   │   ├── test_keys.py
│   │   └── test_integrity.py
│   └── benchmarks/
│       ├── bench_compression.py
│       ├── bench_throughput.py
│       └── bench_integrity.py
├── docs/
│   ├── index.md
│   ├── theory.md
│   ├── architecture.md
│   ├── api.md
│   ├── benchmarks.md
│   └── changelog.md
└── examples/
    ├── quickstart.py
    ├── encode_file.py
    └── integrity_demo.py
```

The `src/` layout is used (PEP 517 src-layout) so that tests always import the installed package, never the local source tree. This catches packaging errors early.

### 4.2 Dependency Policy

**Core (mandatory)**:
- `numpy >=1.26,<3` — Kronecker products, matrix algebra, complex arithmetic. This is the only hard runtime dependency.

**Optional extras**:
- `braidcodec[quantum]` — adds `qiskit >=1.0` for full Jordan-Wigner stabilizer formalism in `fermion_bounds`. Without it, `fermion_bounds` falls back to a vendored lightweight Pauli class that covers the string construction and parity operator needs without the 200MB qiskit install.
- `braidcodec[cli]` — adds `click >=8.0` for the command-line interface.
- `braidcodec[dev]` — adds `pytest`, `pytest-cov`, `pytest-benchmark`, `mypy`, `ruff`, `pre-commit`.

**Python version**: `>=3.13` (takes advantage of the new REPL, improved error messages, and the `typing` improvements; signals modernity to reviewers).

### 4.3 Module Dependency Graph

```
tsr_constants          (zero internal deps)
       ↓
braid_equations        (imports tsr_constants.C_CONSTANT)
       ↓
yang_baxter            (imports braid_equations.{BraidEquation, contract_braid_tensor, simplify_braid})
       
fermion_bounds         (independent, optional qiskit)

codec/chunker          (imports nothing from algebra)
codec/schema           (imports braid_equations.BraidEquation, algebra types)
codec/encoder          (imports chunker, schema, braid_equations, yang_baxter)
codec/decoder          (imports schema, braid_equations)
codec/compressor       (imports braid_equations.{simplify_braid, jones_polynomial}, yang_baxter)

crypto/keys            (imports tsr_constants, braid_equations.get_sector_r_matrix)
crypto/integrity       (imports yang_baxter, braid_equations.jones_polynomial, fermion_bounds)

cli/main               (imports codec, crypto — top-level only)
```

No circular dependencies. The algebra layer has zero upward imports.

---

## 5. Serialization Engine — Detailed Design

This is the core of BraidCodec and the primary new engineering work. The Julia `serialization.jl` was tightly coupled to the RNN's GPU-resident `DSLayer` types, `CausalSphereProjection`, and CUDA arrays. None of that applies here. The Python serialization engine is designed from scratch as a general-purpose binary codec operating on the verified Python algebra modules.

### 5.1 `_types.py` — Shared Type Definitions

Defines type aliases used across modules:

- `GeneratorSeq = list[int]` — a sequence of signed braid generators.
- `SectorName = Literal["Identity", "TSR", "Ising", "Fibonacci", "SU2k2"]`
- `JonesValue = complex` — a Jones polynomial evaluation.
- `ByteChunk = bytes` — a fixed-size input block.

### 5.2 `codec/schema.py` — Wire Format

The serialized representation of encoded data. This replaces `SerializedKnowledgeDisk` from the Julia side.

**`EncodedBlock` dataclass** — represents one encoded chunk of input data:

| Field | Type | Description |
|---|---|---|
| `generators` | `list[int]` | Braid generator sequence (the lossless encoding) |
| `n_strands` | `int` | Strand count used for this block |
| `sector` | `str` | Anyon sector (part of the key, stored for decoding) |
| `jones` | `complex` | Jones polynomial of the braid (integrity fingerprint) |
| `writhe` | `int` | Writhe value (fast integrity pre-check) |
| `block_index` | `int` | Position of this block in the original byte stream |
| `original_length` | `int` | Byte length of the original chunk (needed for last-block padding removal) |

**`EncodedStream` dataclass** — the full encoded payload:

| Field | Type | Description |
|---|---|---|
| `version` | `int` | Wire format version (starts at 1) |
| `blocks` | `list[EncodedBlock]` | Ordered encoded blocks |
| `n_strands` | `int` | Global strand count |
| `sector` | `str` | Sector used (redundant with blocks, but useful for header-level validation) |
| `total_bytes` | `int` | Original input byte count |
| `checksum` | `bytes` | BLAKE3 hash of the original input (defense-in-depth) |
| `timestamp` | `int` | Nanosecond Unix timestamp of encoding |
| `metadata` | `dict[str, str]` | Extensible key-value metadata |

**Binary wire format** — `EncodedStream` serializes to bytes using `msgpack` (added as core dependency, ~50KB, pure Python fallback available) with the following layout:

```
[4 bytes: magic "BRDC"]
[2 bytes: version, big-endian uint16]
[4 bytes: header length, big-endian uint32]
[header: msgpack-encoded EncodedStream metadata without blocks]
[4 bytes: block count, big-endian uint32]
[for each block:
    [4 bytes: block payload length, big-endian uint32]
    [block: msgpack-encoded EncodedBlock]
]
[32 bytes: BLAKE3 digest of everything above]
```

The trailing digest covers the entire file including headers, so any tampering with metadata is also detected.

**`to_bytes()` and `from_bytes()` class methods** on `EncodedStream` handle serialization/deserialization with explicit versioning for forward compatibility.

### 5.3 `codec/chunker.py` — Byte-to-Block Partitioning

Responsible for splitting an arbitrary byte stream into fixed-size blocks suitable for braid encoding.

**Design decisions**:

- **Block size**: determined by `n_strands`. With `n` strands and dimension `d=2`, the braid matrix is `2^n × 2^n`. Each generator index is in `[1, n-1]` with a sign bit, so each generator carries `ceil(log2(2(n-1)))` bits of information. A block of `k` generators on `n` strands encodes `k × ceil(log2(2(n-1)))` bits. The chunker computes the optimal block byte size for given `(n_strands, generators_per_block)` parameters.

- **Padding**: the last block is right-padded with zeros. `original_length` on `EncodedBlock` records the actual byte count for stripping on decode.

- **Streaming interface**: `chunk_stream(data: bytes | BinaryIO, block_size: int) -> Iterator[tuple[int, bytes]]` yields `(block_index, chunk)` pairs. This supports both in-memory and file-based encoding without materializing the entire input.

**`bytes_to_generators(chunk: bytes, n_strands: int) -> list[int]`** — the core mapping function:

1. Interpret the chunk as a big-endian integer `N`.
2. Express `N` in a mixed-radix system where each digit position has base `2(n-1)` — there are `n-1` positive generators and `n-1` negative generators.
3. Map each digit `d` to a signed generator: `d < (n-1)` maps to `+(d+1)`, `d >= (n-1)` maps to `-(d - n + 2)`.
4. The generator sequence length `k` is the minimum needed to represent the maximum possible block value: `k = ceil(block_bits / log2(2(n-1)))`.

This is a bijective numeration — every byte sequence maps to exactly one generator sequence and vice versa. No information is lost, no ambiguity exists.

**`generators_to_bytes(generators: list[int], n_strands: int, original_length: int) -> bytes`** — the inverse mapping:

1. Map each generator back to a digit in base `2(n-1)`.
2. Reconstruct the big-endian integer `N`.
3. Convert to bytes, truncate to `original_length`.

### 5.4 `codec/encoder.py` — Encoding Pipeline

The encoder orchestrates the full pipeline: `bytes → chunks → generators → braid → invariants → EncodedStream`.

**`encode(data: bytes, key: BraidKey, *, generators_per_block: int = 32) -> EncodedStream`**:

1. **Chunk**: split `data` into blocks via `chunker.chunk_stream()`.
2. **Map**: for each chunk, call `bytes_to_generators()` to get the generator sequence.
3. **Construct**: create a `BraidEquation(n_strands, generators, sector=key.sector)`.
4. **Contract**: call `contract_braid_tensor()` to get the unitary matrix (validates the braid is well-formed).
5. **Invariants**: compute `jones_polynomial()` and `writhe()` — these become the integrity fingerprints.
6. **Simplify**: call `simplify_braid()` to get the canonical (shortest) generator sequence. This is the compression step — if the original sequence contained cancellable pairs, the simplified form is strictly shorter.
7. **Validate**: call `validate_braid_category()` to verify all generators are in valid range.
8. **Pack**: construct `EncodedBlock` with the simplified generators, Jones value, writhe, and metadata.
9. **Assemble**: collect all blocks into an `EncodedStream` with global metadata and BLAKE3 checksum.

**Compression note**: the simplification in step 6 is the first compression pass. The `compressor` module (section 5.6) provides deeper topological compression as an optional second pass.

**Parallelism**: blocks are independent. The encoder accepts an optional `max_workers: int` parameter and uses `concurrent.futures.ProcessPoolExecutor` for parallel block encoding. Each block's `BraidEquation` construction, contraction, and invariant computation are CPU-bound and embarrassingly parallel.

### 5.5 `codec/decoder.py` — Decoding Pipeline

**`decode(stream: EncodedStream, key: BraidKey) -> bytes`**:

1. **Verify key**: check that `stream.sector == key.sector` and `stream.n_strands == key.n_strands`.
2. **For each block**: extract `generators` from `EncodedBlock`.
3. **Integrity pre-check**: recompute `writhe()` from generators, compare to stored value. Writhe is O(n) to compute vs O(2^m) for Jones, so this is a fast rejection filter.
4. **Integrity full check**: recompute `jones_polynomial()`, compare to stored value within tolerance `|ΔJ| < 1e-8`. If mismatch, raise `IntegrityError` with block index.
5. **Inverse map**: call `generators_to_bytes()` to recover the original chunk.
6. **Strip padding**: use `original_length` to remove trailing padding from the last block.
7. **Reassemble**: concatenate all chunks in block order.
8. **Final verify**: compute BLAKE3 of the reassembled bytes, compare to `stream.checksum`. This catches any systematic error in the decode pipeline itself.

**`IntegrityError`** — custom exception raised when any integrity check fails. Contains `block_index`, `expected_jones`, `actual_jones`, and `expected_writhe`, `actual_writhe` for diagnostics.

### 5.6 `codec/compressor.py` — Topological Compression

The compressor operates on `EncodedBlock` objects after initial encoding, applying deeper topological reductions that go beyond adjacent inverse cancellation.

**Level 0 — Adjacent cancellation** (already done by `simplify_braid`):
Removes `σᵢσᵢ⁻¹` and `σᵢ⁻¹σᵢ` pairs. This is what the encoder does by default.

**Level 1 — Far commutativity**:
Generators on distant strands commute: `σᵢσⱼ = σⱼσᵢ` when `|i-j| ≥ 2`. The compressor sorts the generator sequence by a canonical ordering (e.g., lexicographic on `(|position|, sign)`) within commuting windows, then re-runs adjacent cancellation. This can expose cancellable pairs that were previously separated by commuting generators.

**Level 2 — Yang-Baxter rewriting**:
Apply the braid relation `σᵢσᵢ₊₁σᵢ → σᵢ₊₁σᵢσᵢ₊₁` and its inverse as rewrite rules, searching for shorter representations. This is a bounded search (maximum rewrite depth configurable, default 100) since the space of equivalent braids is infinite. Uses `check_yb_equivalence_catlab()` to verify each rewrite preserves the braid matrix.

**Level 3 — Markov stabilization** (optional, for aggressive compression):
Markov moves relate braids on different strand counts. If a braid on `n` strands can be represented more compactly on `n-1` strands (by removing a stabilization), the compressor does this. This changes `n_strands` on the block and requires the decoder to handle variable strand counts.

**`compress(block: EncodedBlock, *, level: int = 1) -> EncodedBlock`** — returns a new block with potentially shorter generator sequence. The Jones polynomial is preserved (verified) since all transformations are topological equivalences.

**`compression_ratio(original: EncodedBlock, compressed: EncodedBlock) -> float`** — reports `len(compressed.generators) / len(original.generators)`.

---

## 6. Crypto Module

### 6.1 `crypto/keys.py` — Key Management

**`BraidKey` dataclass**:

| Field | Type | Description |
|---|---|---|
| `sector` | `SectorName` | Anyon sector determining R-matrix family |
| `n_strands` | `int` | Strand count (determines matrix dimension) |
| `theta_offset` | `float` | Phase offset added to the sector's base θ |
| `key_id` | `str` | Hex identifier derived from key parameters |

The `theta_offset` is the critical security parameter. The base R-matrix for each sector uses a fixed θ (e.g., `π·C` for TSR). The offset perturbs this, creating a custom R-matrix that still satisfies Yang-Baxter (since the phased-SWAP family satisfies YBE for any phase triple) but produces different Jones polynomials. Without knowing `theta_offset`, an attacker cannot reconstruct the R-matrix from the Jones polynomial.

**`keygen(sector: SectorName = "TSR", n_strands: int = 4, theta_offset: float | None = None) -> BraidKey`**:
- If `theta_offset` is None, generates one from `secrets.token_bytes(32)` mapped to `[0, 2π)`.
- Computes `key_id = blake3(sector || n_strands || theta_offset)[:16].hex()`.
- Validates that the resulting R-matrix passes unitarity and YBE checks before returning.

**`key_to_bytes(key: BraidKey) -> bytes`** and **`key_from_bytes(data: bytes) -> BraidKey`** — serialization for key storage/exchange.

**Security model**: BraidCodec does not claim to be a replacement for AES-256 or any production cryptographic system. The security rests on the computational hardness of inverting the Jones polynomial (a `#P`-hard problem in general), but the specific instantiation here has not been cryptanalyzed. The README and docs will state this explicitly. The value proposition for the portfolio is the mathematical elegance of the approach, not a production security claim.

### 6.2 `crypto/integrity.py` — Verification Pipeline

**`verify(stream: EncodedStream, key: BraidKey) -> VerificationResult`**:

Runs the full validation triad from the RNN's architecture, adapted for standalone use:

1. **Yang-Baxter check**: for each block, reconstruct the `BraidEquation` and call `validate_braid_category()`. For blocks with 3+ generators, spot-check YBE on random adjacent triples by constructing the two sides of the braid relation and calling `verify_yang_baxter_morphism()`.

2. **Jones polynomial check**: recompute `jones_polynomial()` for each block, compare to stored value.

3. **Fermion bounds check** (optional, enabled by flag): map the generator sequence to a fermionic occupation pattern — positive generators occupy sites, negative generators vacate. Verify that the sequence never violates Pauli exclusion (double-occupation) and that total parity is conserved through hopping operations. This provides an independent validation channel that doesn't rely on matrix contraction.

4. **Checksum check**: BLAKE3 of decoded bytes vs stored checksum.

**`VerificationResult` dataclass**:

| Field | Type | Description |
|---|---|---|
| `valid` | `bool` | Overall pass/fail |
| `yb_passed` | `bool` | Yang-Baxter checks passed |
| `jones_passed` | `bool` | All Jones polynomials match |
| `fermion_passed` | `bool | None` | Fermion bounds check (None if not run) |
| `checksum_passed` | `bool` | BLAKE3 match |
| `failed_blocks` | `list[int]` | Block indices that failed any check |
| `details` | `dict[int, str]` | Per-block failure descriptions |

---

## 7. CLI Module

**Entry point**: `braidcodec` (registered via `[project.scripts]` in `pyproject.toml`).

**Commands** (via `click`):

```
braidcodec encode <input-file> -o <output-file> [--sector TSR] [--strands 4] [--key <key-file>] [--compression-level 1] [--workers 4]
braidcodec decode <input-file> -o <output-file> --key <key-file>
braidcodec verify <encoded-file> --key <key-file> [--fermion-check]
braidcodec keygen [--sector TSR] [--strands 4] -o <key-file>
braidcodec inspect <encoded-file>   # Print metadata, block count, compression stats
braidcodec benchmark <input-file>   # Run encode/decode/verify cycle with timing
```

**Output format**: `inspect` prints human-readable YAML-like output. `benchmark` prints a table with timing, compression ratio, and throughput.

**Exit codes**: 0 = success, 1 = integrity failure, 2 = key mismatch, 3 = format error, 4 = I/O error.

---

## 8. Public API Surface

The top-level `braidcodec/__init__.py` exposes a minimal, clean API:

```python
# Core codec
from braidcodec.codec.encoder import encode
from braidcodec.codec.decoder import decode
from braidcodec.codec.compressor import compress

# Key management
from braidcodec.crypto.keys import keygen, BraidKey
from braidcodec.crypto.integrity import verify, VerificationResult

# Schema (for advanced users)
from braidcodec.codec.schema import EncodedStream, EncodedBlock

# Version
from braidcodec._version import __version__
```

Everything else is internal (`_`-prefixed or accessed via subpackage import). The algebra modules are not re-exported at top level — they're implementation details. Advanced users can import `braidcodec.algebra.braid_equations` directly, but it's not the intended interface.

---

## 9. Testing Strategy

### 9.1 Test Categories

**Unit tests** (`tests/algebra/`, `tests/codec/`, `tests/crypto/`):
- Each module gets its own test file mirroring the module structure.
- The existing 150-check verification script is decomposed into proper pytest parametrized tests.
- Target: 95%+ line coverage on all non-CLI modules.

**Property-based tests** (via `hypothesis`):
- `bytes_to_generators(generators_to_bytes(gens)) == gens` for arbitrary generator sequences.
- `decode(encode(data)) == data` for arbitrary byte strings up to 10KB.
- `jones_polynomial(b)` is invariant under `simplify_braid(b)` for random braids.
- Writhe is additive under composition: `writhe(b1 * b2) == writhe(b1) + writhe(b2)`.
- Fermion parity is conserved through arbitrary hopping sequences.

**Integration tests** (`tests/codec/test_roundtrip.py`):
- Full encode → serialize → deserialize → decode → verify pipeline on known test vectors.
- Round-trip on each sector.
- Round-trip on files of various sizes: 0 bytes, 1 byte, 1KB, 1MB.
- Round-trip with each compression level.

**Benchmark tests** (`tests/benchmarks/`):
- Not run in CI by default (gated behind `--benchmark` flag or separate workflow).
- Track encode/decode throughput in bytes/second.
- Track compression ratio vs input entropy.
- Track integrity check time vs block count.

### 9.2 Test Fixtures (`conftest.py`)

- `small_braid` — `BraidEquation(3, [1, 2, 1], sector="TSR")`
- `identity_braid` — `BraidEquation(2, [])`
- `all_sectors` — parametrize over all 5 sectors
- `random_braid(n_strands, length, sector)` — factory fixture for random braids
- `sample_key` — `keygen(sector="TSR", n_strands=4, theta_offset=0.42)`
- `sample_data` — `b"BraidCodec topological data codec"` (32 bytes)
- `fermion_bounds_5` — `create_fermion_bounds(5, initial_occupation=[1, 3])`

### 9.3 Coverage Requirements

| Module | Minimum Coverage |
|---|---|
| `algebra/*` | 98% |
| `codec/*` | 95% |
| `crypto/*` | 95% |
| `cli/*` | 85% |

Coverage is enforced in CI via `pytest-cov` with `--cov-fail-under`.

---

## 10. CI/CD Pipeline

### 10.1 GitHub Actions Workflows

**`ci.yml`** — triggered on every push to `main` and every PR:

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install ruff
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install mypy numpy-stubs
      - run: mypy src/braidcodec --strict

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.13"]
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=braidcodec --cov-report=xml --cov-fail-under=95
      - uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install build
      - run: python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
```

**`release.yml`** — triggered on pushing a version tag (`v*`):

```yaml
jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # trusted publishing
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Uses PyPI trusted publishing (OIDC) — no API tokens stored in secrets. The GitHub environment `pypi` is configured with the PyPI project.

**`benchmark.yml`** — runs weekly on `main` and on-demand:

```yaml
on:
  schedule:
    - cron: "0 6 * * 1"  # Monday 6am UTC
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install -e ".[dev]"
      - run: pytest tests/benchmarks/ --benchmark-json=bench.json
      - uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: pytest
          output-file-path: bench.json
          auto-push: true
          gh-pages-branch: gh-pages
          benchmark-data-dir-path: benchmarks
```

This publishes benchmark results to GitHub Pages, creating a historical performance dashboard.

### 10.2 Pre-commit Hooks

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=500]
```

### 10.3 Branching Strategy

- `main` — always releasable. Protected: requires passing CI and 1 approval (even if self-approved for solo project, the PR discipline shows process).
- `dev` — integration branch for feature work.
- Feature branches: `feat/<name>`, `fix/<name>`, `bench/<name>`.
- Releases: tag `v0.1.0`, `v0.2.0`, etc. on `main` after merging from `dev`.

### 10.4 Versioning

Follows SemVer. Version is stored in `src/braidcodec/_version.py` as:
```python
__version__ = "0.1.0"
```

The `pyproject.toml` reads it dynamically:
```toml
[project]
dynamic = ["version"]

[tool.setuptools.dynamic]
version = {attr = "braidcodec._version.__version__"}
```

---

## 11. `pyproject.toml` Specification

```toml
[build-system]
requires = ["setuptools>=75.0", "setuptools-scm>=8"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "braidcodec"
dynamic = ["version"]
description = "Topological data codec: encryption + compression + integrity via braid group invariants"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.13"
authors = [{name = "Mehdi", email = "..."}]
keywords = ["braid-group", "topology", "encryption", "compression", "jones-polynomial", "yang-baxter", "codec"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.13",
    "Topic :: Security :: Cryptography",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: System :: Archiving :: Compression",
    "Typing :: Typed",
]

dependencies = [
    "numpy",
    "msgpack",
    "blake3>=1.0,<2",
]

[project.optional-dependencies]
quantum = ["qiskit>=1.0"]
cli = ["click>=8.0,<9"]
dev = [
    "pytest>=8.0",
    "pytest-cov>=6.0",
    "pytest-benchmark>=5.0",
    "hypothesis>=6.100",
    "mypy>=1.13",
    "ruff>=0.8",
    "pre-commit>=4.0",
]

[project.scripts]
braidcodec = "braidcodec.cli.main:cli"

[project.urls]
Homepage = "https://github.com/mehdi/braidcodec"
Documentation = "https://mehdi.github.io/braidcodec"
Repository = "https://github.com/mehdi/braidcodec"
Issues = "https://github.com/mehdi/braidcodec/issues"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
target-version = "py313"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "SIM", "TCH", "RUF", "ARG", "PTH"]
ignore = ["E501"]  # line length handled by formatter

[tool.ruff.lint.isort]
known-first-party = ["braidcodec"]

[tool.mypy]
python_version = "3.13"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = ["qiskit.*", "msgpack.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "benchmark: benchmark tests (run with '--benchmark')",
]

[tool.coverage.run]
source = ["braidcodec"]
branch = true

[tool.coverage.report]
fail_under = 95
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "assert_never",
]
```

---

## 12. Documentation

### 12.1 `README.md` Structure

1. **Hero banner** — project name, one-line description, badges (CI, coverage, PyPI version, Python version, license).

2. **10-line quickstart** showing encode → verify → decode.

3. **How it works** — a 3-paragraph explanation with an architecture diagram (Mermaid in the README, rendered by GitHub):
   - Paragraph 1: data → braid generators (bijective numeration).
   - Paragraph 2: generators → R-matrix contraction → Jones polynomial (the invariant).
   - Paragraph 3: sector + phase offset = key; YBE violation = corruption.

4. **Installation**: `pip install braidcodec` and optional extras.

5. **Benchmarks** — table of compression ratios and throughput, link to live dashboard on GitHub Pages.

6. **API reference** — link to full docs.

7. **Theory** — link to `docs/theory.md`.

8. **Security disclaimer** — explicit statement that this is a research/portfolio project, not audited for production cryptographic use.

9. **License** — MIT.

### 12.2 `docs/theory.md`

Covers the mathematical foundation at a level suitable for a hiring manager with a CS degree or a researcher without braid group background:

- Braid groups: generators, relations, the braid word problem.
- R-matrices: the phased-SWAP family, why it satisfies YBE.
- Jones polynomial: Kauffman bracket, writhe, topological invariance.
- TSR connection: how `C = 1/(5√2)` parameterizes the default sector.
- Fermionic bounds: Jordan-Wigner, Pauli exclusion as constraint validation.
- Security argument: Jones polynomial inversion is `#P`-hard (cite Kuperberg, Freedman-Kitaev-Wang).
- Compression argument: topological equivalence classes, Markov theorem.

### 12.3 `docs/architecture.md`

- Module dependency diagram.
- Data flow diagram: bytes → chunks → generators → braids → invariants → wire format.
- Decode flow: wire format → generators → invariants check → bytes.
- Thread model for parallel encoding.

### 12.4 `docs/api.md`

Auto-generated from docstrings using `mkdocs` with `mkdocstrings` plugin. Published to GitHub Pages via the CI benchmark workflow (or a dedicated docs workflow).

---

## 13. Benchmark Suite

### 13.1 Compression Benchmarks (`bench_compression.py`)

Test corpora:
- **Canterbury corpus**: 11 files, standard compression benchmark.
- **Synthetic**: random bytes (incompressible baseline), all-zeros (maximally compressible), English text, JSON, binary protobuf.

Metrics:
- Compression ratio: `encoded_size / original_size`.
- Compare against: raw (no compression), gzip -6, zstd -3.
- Report per-file and aggregate.

Expected outcome: BraidCodec will not beat gzip/zstd on pure compression ratio — that's not the point. The story is "comparable compression with integrated encryption and integrity for free." Even a 0.95 ratio (5% smaller) on structured data would be noteworthy given the other properties.

### 13.2 Throughput Benchmarks (`bench_throughput.py`)

Metrics:
- Encode throughput: MB/s.
- Decode throughput: MB/s.
- Verify throughput: MB/s.
- Scaling: throughput vs. `n_strands` (2, 3, 4, 5, 6) and vs. `generators_per_block` (8, 16, 32, 64).

The 2^n matrix contraction is the bottleneck. For n_strands=4 (16×16 matrices), this should be fast. For n_strands=8 (256×256), it will be slow. The benchmark quantifies this tradeoff.

### 13.3 Integrity Benchmarks (`bench_integrity.py`)

- Encode a file, randomly flip 1, 2, 4, 8, 16 bits in the encoded representation.
- Measure detection rate (should be 100% for any flipped generator).
- Measure false positive rate (should be 0% for unmodified data).
- Measure time to detect corruption (first-failing-block latency).

---

## 14. Implementation Roadmap

### Phase 1 — Codec Core (Week 1)

| Task | Module | Deliverable |
|---|---|---|
| Schema definition | `codec/schema.py` | `EncodedBlock`, `EncodedStream` dataclasses with `to_bytes`/`from_bytes` |
| Chunker | `codec/chunker.py` | `bytes_to_generators`, `generators_to_bytes`, `chunk_stream` |
| Encoder | `codec/encoder.py` | `encode()` function, serial implementation |
| Decoder | `codec/decoder.py` | `decode()` function with integrity checks |
| Round-trip test | `tests/codec/test_roundtrip.py` | Parametrized round-trip on all sectors, sizes 0B–1MB |

**Gate**: `decode(encode(data, key), key) == data` for all test vectors.

### Phase 2 — Crypto + Compression (Week 2)

| Task | Module | Deliverable |
|---|---|---|
| Key management | `crypto/keys.py` | `keygen`, `BraidKey`, serialization |
| Integrity verifier | `crypto/integrity.py` | `verify()` with full triad |
| Compressor L0-L1 | `codec/compressor.py` | Adjacent cancellation + far commutativity |
| Property tests | `tests/` | Hypothesis strategies for braids, byte streams |

**Gate**: `verify(encode(data, key), key).valid == True` for all test vectors. Compression level 1 produces equal or shorter generator sequences than level 0 on all inputs.

### Phase 3 — CLI + Packaging (Week 3)

| Task | Module | Deliverable |
|---|---|---|
| CLI | `cli/main.py` | All 6 commands working |
| pyproject.toml | root | Complete with all extras, scripts, metadata |
| CI pipeline | `.github/workflows/` | ci.yml, release.yml passing |
| Qiskit fallback | `algebra/fermion_bounds.py` | Graceful degradation without qiskit |

**Gate**: `pip install -e ".[dev]"` works clean. `ruff check` and `mypy --strict` pass. CI green.

### Phase 4 — Benchmarks + Docs + Polish (Week 4)

| Task | Module | Deliverable |
|---|---|---|
| Benchmarks | `tests/benchmarks/` | All three benchmark suites |
| README | root | Hero section, quickstart, badges |
| Theory docs | `docs/theory.md` | Full mathematical writeup |
| Architecture docs | `docs/architecture.md` | Diagrams, data flow |
| Parallel encoder | `codec/encoder.py` | `max_workers` parameter |
| L2 compression | `codec/compressor.py` | YB rewriting |
| GitHub Pages | `.github/workflows/` | Benchmark dashboard live |
| PyPI release | `.github/workflows/release.yml` | v0.1.0 published |

**Gate**: `pip install braidcodec` works from PyPI. README renders correctly. Benchmark dashboard is live.

---

## 15. Risk Register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Jones polynomial computation is O(2^m) in generators — blocks with many crossings are slow | High | Medium | Cap `generators_per_block` at 32 default; parallelize blocks; provide timing estimates in CLI |
| Compression ratio may be negative (encoded larger than input) for high-entropy data | Medium | High | Frame benchmarks honestly; the story is unified crypto+compression+integrity, not pure compression. Show that for structured data the ratio is favorable |
| `msgpack` adds a dependency some users may not want | Low | Low | It's 50KB, widely used, and the alternative (custom binary format) is more error-prone |
| Qiskit dependency makes install heavy for `fermion_bounds` | Medium | High | Already mitigated: optional extra, lightweight fallback Pauli class |
| Cryptographic security claims attract scrutiny | High | Medium | Explicit disclaimer in README, docs, and module docstrings. Frame as "topological encoding with confidentiality properties" not "encryption" |
| Python 3.13 requirement excludes users on older versions | Low | Medium | Acceptable for a portfolio project. Signals modernity. Core code would work on 3.10+ with minor syntax changes if demand exists |

---

## 16. Success Criteria

For the portfolio to achieve its goal, the following must all be true at v0.1.0:

1. **`pip install braidcodec`** works from PyPI with zero errors.
2. **`braidcodec encode / decode / verify`** CLI works end-to-end on arbitrary files.
3. **100% round-trip lossless** on all test vectors (0 bytes through 1MB, all sectors).
4. **100% integrity detection** for single-bit corruptions in encoded data.
5. **CI is green** with 95%+ coverage, mypy strict, ruff clean.
6. **README** has working quickstart code block, architecture diagram, and benchmark table.
7. **Theory docs** explain the mathematics to a reader with undergrad linear algebra.
8. **Benchmark dashboard** is live on GitHub Pages with historical tracking.
9. **No security overclaims** — all crypto-adjacent language is qualified.
10. **Clean git history** — conventional commits, PR-based workflow, no force pushes to main.