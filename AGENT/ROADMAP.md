

## Plan: BraidCodec Phased Implementation

Build the full `braidcodec` PyPI package on top of the verified algebra engine (4 modules, 459 tests). Restructure into PEP 517 src-layout, enforce strict ruff/mypy from step zero, then build codec → crypto → CLI layers with >95% coverage and full CI/CD. The algebra layer is frozen — only typing/lint fixes, no runtime changes.

---

### Phase 0 — Project Scaffold & Algebra Migration

**Goal**: Installable package, strict linting/typing, all 459 algebra tests green.

1. Create `src/braidcodec/` directory tree per PRD §4.1 (algebra/, codec/, crypto/, cli/)
2. Create pyproject.toml with strict ruff + mypy config per PRD §11 — deps: `numpy>=1.26,<3`, `msgpack>=1.0,<2`, `blake3>=1.0,<2`; extras: `[dev]`, `[cli]`, `[quantum]`
3. Create `_version.py` (`"0.1.0"`), `py.typed` (PEP 561)
4. Copy 4 algebra modules into `src/braidcodec/algebra/`, preserving runtime semantics
5. Add type annotations to all algebra modules for `mypy --strict` — numpy NDArray types, `X | None` unions, missing return types, `_StabilizerPhases` internal typing
6. Vendor `_pauli_compat.py` (~50 lines) as fallback for `qiskit.quantum_info.Pauli` — Z-string construction + parity operator only. Guard import in `fermion_bounds.py` with try/except
7. Fix ruff lint violations (isort, UP rules for py313, etc.)
8. Create `src/braidcodec/_types.py` — `GeneratorSeq`, `SectorName`, `JonesValue`, `ByteChunk`, tolerance constants
9. Migrate 4 test files into `tests/algebra/`, fix imports
10. Create `.github/workflows/ci.yml` (lint + typecheck + test + coverage on 3 OS), `.pre-commit-config.yaml`
11. Create `LICENSE` (MIT), skeleton `README.md`, `CHANGELOG.md`
12. Git init, push, verify CI green

**Key files**: `pyproject.toml`, `src/braidcodec/algebra/*.py` (migrated + typed), `src/braidcodec/_types.py`, `src/braidcodec/algebra/_pauli_compat.py`, `.github/workflows/ci.yml`

**Verify**: `ruff check` zero violations → `mypy --strict` zero errors → `pytest tests/algebra/` all 459 green

---

### Phase 1 — Chunker (Bijective Numeration)

**Goal**: `bytes_to_generators` ↔ `generators_to_bytes` round-trips perfectly. Zero algebra deps.

1. Implement `src/braidcodec/codec/chunker.py` — `compute_block_size()`, `bytes_to_generators()` (mixed-radix, PRD §5.3), `generators_to_bytes()` (inverse), `chunk_stream()` (streaming iterator)
2. Write `tests/codec/test_chunker.py` — known vectors, edge cases (0/1 byte, block boundary, padding), Hypothesis round-trip for arbitrary chunks × n_strands ∈ [3,6]

**Verify**: Hypothesis generates 1000+ round-trip examples, coverage ≥ 95%

---

### Phase 2 — Schema (Wire Format)

**Goal**: `EncodedBlock` / `EncodedStream` with msgpack `to_bytes`/`from_bytes` round-tripping.

1. Implement `src/braidcodec/codec/schema.py` — dataclasses per PRD §5.2, binary framing (`BRDC` magic + version + header + blocks + trailing BLAKE3 per Architecture §8), exception hierarchy (`FormatError`, `IntegrityError` tree per Architecture §10.1)
2. Write `tests/codec/test_schema.py` — construction → to_bytes → from_bytes round-trip, bad magic/version/truncation/digest corruption detection

**Verify**: Corrupted inputs raise correct exception subclass

---

### Phase 3 — Encoder + Keys

**Goal**: `encode(data, key) -> EncodedStream` for all sectors.

1. Implement `src/braidcodec/crypto/keys.py` — `BraidKey` dataclass, `keygen()` (secrets-based), `key_to_bytes`/`key_from_bytes`, `_KeyedRMatrixProvider` (injects theta_offset without modifying algebra layer, Architecture §7.2)
2. Implement `src/braidcodec/codec/encoder.py` — full pipeline: chunk → generators → `BraidEquation` → contract → invariants (tiered: Jones k≤24, trace k>24) → simplify → `EncodedBlock` → `EncodedStream`. `ProcessPoolExecutor` parallelism.
3. Write `tests/crypto/test_keys.py` — keygen validity, key round-trip, deterministic key_id
4. Write `tests/codec/test_encoder.py` — encode "Hello", all sectors, invariant correctness, parallel == serial

**Verify**: Encoded stream deserializes cleanly, invariants match independent recomputation

---

### Phase 4 — Decoder & Round-trip (*depends on Phase 3*)

**Goal**: `decode(encode(data, key), key) == data` everywhere. Full integrity pipeline.

1. Implement `src/braidcodec/codec/decoder.py` — writhe pre-check → Jones/trace recomputation → BLAKE3 final verify. `IntegrityError` subclasses (`WritheError`, `JonesError`, `ChecksumError`). Partial decode with `on_error="skip"`.
2. Implement `src/braidcodec/crypto/integrity.py` — `verify()` with 5-channel validation, `VerificationResult` dataclass
3. Write `tests/codec/test_decoder.py` — tampered generators/checksum/key-mismatch detected
4. Write `tests/codec/test_roundtrip.py` — round-trip: empty, 1B, 32B, 1KB, 10KB; all sectors; **Hypothesis: arbitrary bytes up to 10KB**
5. Write `tests/crypto/test_integrity.py` — clean verify passes, each corruption type caught by correct channel

**Verify**: Hypothesis round-trip 1000+ examples, every corruption class detected

---

### Phase 5 — Compressor (*parallel with Phase 4 after Phase 3*)

**Goal**: Levels 0-2 shorten generators while preserving invariants.

1. Implement `src/braidcodec/codec/compressor.py` — Level 0 (inverse cancellation via `simplify_braid`), Level 1 (far commutativity normalization + re-cancel, Architecture §5), Level 2 (YB rewriting, bounded BFS, max 100 rewrites). Registry pattern.
2. Write `tests/codec/test_compressor.py` — L0 cancellation, L1 distant commute example (`[1,3,-1,2]` → `[3,2]`), L2 YB rewrite validity, **Hypothesis: Jones invariant preserved post-compression**

**Verify**: Compressed blocks decode to same original bytes

---

### Phase 6 — CLI & Public API (*depends on Phases 4, 5*)

**Goal**: `braidcodec encode/decode/verify/keygen/inspect/benchmark` works end-to-end.

1. Finalize `src/braidcodec/__init__.py` — re-export: `encode`, `decode`, `compress`, `keygen`, `BraidKey`, `verify`, `VerificationResult`, `EncodedStream`, `EncodedBlock`, `__version__`
2. Implement `src/braidcodec/cli/main.py` — click-based, 6 commands (PRD §7), exit codes 0-4, `--verbose`/`--quiet`, progress bar
3. Write `tests/cli/test_cli.py` — `CliRunner` tests: keygen → encode → verify → decode pipeline, inspect, error codes

**Verify**: Full file round-trip via CLI commands

---

### Phase 7 — Benchmarks, Docs & Release (*depends on Phase 6*)

**Goal**: Performance tracked, docs published, PyPI release automated.

1. Write `tests/benchmarks/bench_{throughput,compression,integrity}.py` (pytest-benchmark, parametrized by n_strands/size/level)
2. Create `.github/workflows/benchmark.yml` (weekly, gh-pages), `.github/workflows/release.yml` (tag-triggered PyPI trusted publishing)
3. Write docs: `docs/{index,theory,architecture,api,benchmarks}.md` with Mermaid diagrams
4. Write full `README.md` (hero, quickstart, how-it-works, benchmarks, security disclaimer)
5. Write `examples/{quickstart,encode_file,integrity_demo}.py`
6. Final coverage audit: `pytest --cov=braidcodec --cov-fail-under=95`
7. Tag `v0.1.0`, push, verify release pipeline

---

### Relevant files (complete)

- `pyproject.toml` — all config (strict ruff select `E,F,W,I,N,UP,B,A,SIM,TCH,RUF,ARG,PTH`; mypy `strict = true`)
- `src/braidcodec/algebra/*.py` — migrated + typed from src
- `src/braidcodec/algebra/_pauli_compat.py` — vendored qiskit Pauli fallback
- `src/braidcodec/_types.py` — shared aliases + tolerance constants
- `src/braidcodec/codec/{chunker,schema,encoder,decoder,compressor}.py` — core codec
- `src/braidcodec/crypto/{keys,integrity}.py` — security surface
- `src/braidcodec/cli/main.py` — click CLI
- `tests/{algebra,codec,crypto,cli,benchmarks}/` — full test suite
- `.github/workflows/{ci,benchmark,release}.yml` — CI/CD

### Verification (every phase)

1. `ruff check src/ tests/` — zero violations
2. `ruff format --check src/ tests/` — zero diffs
3. `mypy src/braidcodec --strict` — zero errors
4. `pytest tests/ -v --cov=braidcodec --cov-fail-under=95` — all green

### Decisions

- **Algebra frozen**: typing/lint fixes only, zero runtime changes. 459 tests = regression suite.
- **qiskit optional**: Vendored `_pauli_compat` (~50 lines) covers Z-string + parity needs. Full qiskit via `[quantum]` extra.
- **Python ≥3.13 only**: `X | None` union syntax, modern typing throughout.
- **No Level 3 compression (Markov stabilization)**: Excluded from v0.1.0, deferred to v0.2.0.
- **ProcessPoolExecutor** (not Thread): CPU-bound numpy work.
- **Branching**: `main` (protected) ← `dev` ← `feat/*`. One PR per phase.
- **Dead deps dropped**: scipy, matplotlib, sympy, cupy from old `requirements.txt` are unused — not in `pyproject.toml`.
- **Tiered invariant threshold**: Start at k=24; if Jones state-sum exceeds 10s/block during Phase 3, lower to k=20.