# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

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
