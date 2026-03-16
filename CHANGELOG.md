# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### In Progress

#### Phase 8 — Topology-First Encoding v2
- Default-on topology preprocessing path with explicit legacy fallback mode.
- Stream schema bumped to `v2` with backward read compatibility for `v1` streams.
- Topology metadata fields added per block:
  - layer descriptors (`topology_layer_index`, `topology_layer_n_chunks`, `topology_nnz_bits`)
  - BFPS terms (`topology_hash32`, `topology_density_fp`, `topology_centroid_fp`, `topology_variance_fp`)
  - Morton/index binding (`topology_morton_key`)
  - additive commitment (`topology_commitment`)
- Encoder now applies deterministic topology-conditioned generator synthesis while preserving decode bijection through `decode_generators`.
- Additive topology integrity verification channel (optional) integrated into `verify(...)` as a sixth channel alongside checksum authority.
- Regime-aware execution policy (`serial` / `thread` / `process`) and layer-aware batching integrated for reduced scheduler/IPC overhead.
- Stage telemetry added to stream metadata (`timing_preprocess_s`, `timing_layer_order_s`, `timing_encode_core_s`, `timing_total_s`).
- New benchmark suites:
  - `bench_core_throughput.py` for bounded CI throughput checks.
  - `bench_stress_regime.py` for pathological combinations in reporting mode.
- Benchmark CI split into core publishing + non-blocking stress reporting with threshold alert script (`scripts/check_stress_thresholds.py`).

#### Remaining for Full Phase 8 Closeout
- Add decode/verify-focused lightweight benchmark suite with explicit topology/legacy deltas.
- Implement hard CI promotion gates (consecutive-pass counters and enforced go/no-go thresholds) once variance stabilizes across repeated runs.
- Expand public docs and migration guidance with finalized performance deltas and tradeoff recommendations from stabilized CI history.
- Final release-note pass for default behavior changes (`preprocessing_mode=topology`, 6-channel optional verify path).

### Added

#### Phase 0 — Scaffold & Algebra Migration
- Project scaffold with PEP 517 src-layout (`src/braidcodec/`)
- Algebra engine migration with strict typing (`mypy --strict`)
- Vendored Pauli fallback for optional qiskit dependency
- CI/CD pipeline: lint (ruff), typecheck (mypy), test (pytest), build

#### Phase 1 — Bijective Chunker
- `chunker.py` — bytes ↔ braid generators via mixed-radix encoding
- Full bijectivity with property-based tests (Hypothesis)

#### Phase 2 — Wire Format Schema
- `schema.py` — `EncodedBlock` / `EncodedStream` dataclasses
- BRDC framing: magic header, versioning, msgpack serialization
- Exception hierarchy: `BraidCodecError` → `FormatError`, `IntegrityError`, etc.

#### Phase 3 — Encoder & Key Management
- `encoder.py` — `encode()` with tiered invariant computation
- `keys.py` — `keygen()`, `key_to_bytes()`, `key_from_bytes()`, `BraidKey`
- Parallel encoding via `ProcessPoolExecutor` (up to 8 workers)
- Five anyon sectors: Identity, TSR, Ising, Fibonacci, SU2k2

#### Phase 4 — Decoder & Integrity Verification
- `decoder.py` — `decode()` with optional per-block invariant verification
- `integrity.py` — `verify()` with 5-channel verification:
  structural, writhe, invariant (Jones/trace), fermion, BLAKE3 checksum
- `VerificationResult` dataclass with per-channel pass/fail and details

#### Phase 5 — Topological Compressor
- `compressor.py` — `compress()` with three levels:
  - Level 0: inverse cancellation
  - Level 1: far-commutativity reordering + re-cancel
  - Level 2: Yang-Baxter BFS search for shorter representative
- `compression_ratio()` utility
- Dual-generator storage for lossless compressed round-trip

#### Phase 6 — CLI & Public API
- `cli/main.py` — 6 click commands: `keygen`, `encode`, `decode`, `verify`, `inspect`, `benchmark`
- Structured exit codes: 0 (OK), 1 (integrity), 2 (key mismatch), 3 (format), 4 (I/O)
- `--verbose` / `--quiet` mutually exclusive flags

#### Phase 7 — Benchmarks, Documentation & Release
- `tests/benchmarks/` — parametrized pytest-benchmark suites (throughput, compression, integrity)
- `docs/` — theory, architecture, API reference, benchmark guide
- `examples/` — quickstart, file encoding, integrity demo
- `.github/workflows/benchmark.yml` — weekly benchmark CI with gh-pages dashboard
- `.github/workflows/release.yml` — PyPI trusted publishing via OIDC
- Shared conftest fixtures (`conftest.py`)
- Enhanced README with badges, Mermaid diagrams, CLI usage, performance table
- Coverage audit: ≥95% overall
