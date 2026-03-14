

---

## Plan: Benchmarks, Docs & Release (Phase 7)

Create pytest-benchmark tests, two new CI workflows (benchmark → gh-pages, release → PyPI), markdown documentation with Mermaid diagrams, example scripts, an enhanced README, updated CHANGELOG, shared conftest fixtures, and a coverage audit — then tag v0.1.0.

---

### Phase A: Benchmark Tests

**Step 1** — Create `tests/benchmarks/__init__.py` (empty)

**Step 2** — Create `tests/benchmarks/bench_throughput.py`
- Parametrize: input sizes [1KB, 10KB, 100KB], `n_strands` [3, 4, 5], sectors ["TSR", "Ising", "Fibonacci"]
- `test_encode_throughput` — `benchmark(encode, data, key, generators_per_block=8)`
- `test_decode_throughput` — pre-encode, `benchmark(decode, stream, key, verify=False)`
- `test_decode_verified_throughput` — `benchmark(decode, stream, key, verify=True)`
- Use `generators_per_block=8` (tier 2 feasible) to keep runtime tractable

**Step 3** — Create `tests/benchmarks/bench_compression.py`
- Parametrize: levels [0, 1, 2], sizes [1KB, 10KB], entropy [zeros, random, mixed]
- `test_compress_throughput` — pre-encode, `benchmark(compress, stream, key, level=level)`
- `test_compression_ratio` — measure via `compression_ratio(original, compressed)`
- Entropy strategies: zeros=`b'\x00'*N`, random=`os.urandom(N)`, mixed=`bytes(range(256))*(N//256+1)`

**Step 4** — Create `tests/benchmarks/bench_integrity.py`
- Parametrize: block counts [1, 5, 10, 50], `fermion_check` [False, True]
- `test_verify_throughput` — pre-encode, `benchmark(verify, stream, key, fermion_check=...)`
- `test_writhe_only` — benchmark writhe computation alone

**Step 5** — Update pyproject.toml pytest config
- Add `--benchmark-disable` to `addopts` so normal `pytest tests/` in CI skips benchmarks
- Dedicated benchmark workflow uses `--benchmark-enable` explicitly

---

### Phase B: CI Workflows *(parallel with Phase A)*

**Step 6** — Create `.github/workflows/benchmark.yml`
- Triggers: `schedule: cron: "0 6 * * 1"` (Monday 6am UTC) + `workflow_dispatch`
- Job: checkout → setup python 3.13 → `pip install -e ".[dev]"` → `pytest tests/benchmarks/ --benchmark-json=bench.json --benchmark-enable` → `benchmark-action/github-action-benchmark@v1` (tool=pytest, auto-push gh-pages)
- Per PRD §10.1

**Step 7** — Create `.github/workflows/release.yml`
- Trigger: `push: tags: ["v*"]`
- Job: `environment: pypi`, `permissions: id-token: write` (OIDC trusted publishing)
- Steps: checkout → setup python 3.13 → `pip install build` → `python -m build` → `pypa/gh-action-pypi-publish@release/v1`
- Per PRD §10.1

---

### Phase C: Documentation *(parallel with A+B)*

**Step 8** — Create `docs/index.md`
- Project overview, installation, quickstart code
- Mermaid pipeline diagram (encode/decode data flow from Architecture §1)
- Navigation links to theory, architecture, API, benchmarks

**Step 9** — Create `docs/theory.md`
- Braid group fundamentals (generators, Artin relations)
- R-matrices: phased-SWAP structure, sector parameterization
- Jones polynomial: Kauffman bracket state-sum, writhe normalization
- Yang-Baxter equation and topological invariance
- Fermion bounds: Jordan-Wigner transform, occupation constraints
- Mermaid: invariant computation hierarchy

**Step 10** — Create `docs/architecture.md`
- Condensed public version of ARCHITECTURE.md
- Mermaid layer diagram: L0 (algebra) → L1 (codec) → L2 (crypto) → L3 (CLI)
- Data flow diagrams for encode, decode, compress, verify paths
- Tiered invariant strategy (writhe → Jones/trace → BLAKE3)
- Performance characteristics table from Architecture §11.2

**Step 11** — Create `docs/api.md`
- All 12 public symbols with signatures and param descriptions
- Dataclass fields: `BraidKey`, `EncodedBlock`, `EncodedStream`, `VerificationResult`
- Exception hierarchy tree with when each is raised
- CLI command reference (from PRD §7)
- Code examples for each function

**Step 12** — Create `docs/benchmarks.md`
- How to run benchmarks locally (`pytest tests/benchmarks/ --benchmark-enable`)
- Parametrization dimensions (size × strands × sector × level × entropy)
- Link to gh-pages performance dashboard
- Performance expectations table (Architecture §11.2)
- Compression ratio expectations by level and input entropy

---

### Phase D: Examples *(parallel with everything)*

**Step 13** — Create example scripts
- `examples/quickstart.py` — keygen → encode → decode → assert (~20 lines)
- `examples/encode_file.py` — argparse file encoding: read → encode → write .brdc + save key (~40 lines)
- `examples/integrity_demo.py` — encode, tamper generators, show verify catches it, demonstrate each channel (~50 lines)
- All runnable standalone with `if __name__ == "__main__"`
- Use `generators_per_block=8` for fast execution

---

### Phase E: README & CHANGELOG

**Step 14** — Enhance README.md
- **Hero**: Title + one-liner + badges (CI status, coverage, PyPI version, license)
- **How It Works**: Mermaid pipeline diagram + braid encoding explanation
- **Installation**: `pip install braidcodec` + extras (already exists, keep)
- **Quick Start**: Python API example (already exists, keep)
- **CLI Usage**: All 6 commands with example output snippets
- **Performance**: Table from Architecture §11.2 (n_strands vs throughput)
- **Security Disclaimer**: Already exists, refine
- **Contributing**: Test/lint/CI requirements
- **License**: MIT (keep)

**Step 15** — Update CHANGELOG.md
- Flesh out v0.1.0 with structured entries per phase:
  - Phase 0: Scaffold, algebra migration, strict typing, CI
  - Phase 1: Bijective chunker (bytes ↔ generators)
  - Phase 2: Wire format schema (BRDC framing, msgpack)
  - Phase 3: Encoder + key management (tiered invariants, parallelism)
  - Phase 4: Decoder + integrity (5-channel verify, VerificationResult)
  - Phase 5: Topological compressor (3 levels, dual-generator storage)
  - Phase 6: CLI (6 click commands, exit codes 0-4)
  - Phase 7: Benchmarks, documentation, examples

---

### Phase F: Fixtures & Coverage

**Step 16** — Populate conftest.py with PRD §9.2 fixtures
- `small_braid` → `BraidEquation(3, [1, 2, 1], sector="TSR")`
- `identity_braid` → `BraidEquation(2, [])`
- `all_sectors` → `pytest.fixture(params=["Identity", "TSR", "Ising", "Fibonacci", "SU2k2"])`
- `random_braid(n_strands, length, sector)` → factory fixture
- `sample_key` → `keygen(sector="TSR", n_strands=4, theta_offset=0.42)`
- `sample_data` → `b"BraidCodec topological data codec"` (32 bytes)
- `fermion_bounds_5` → `create_fermion_bounds(5, initial_occupation=[1, 3])`
- Only add fixtures not already defined locally in test files

**Step 17** — Coverage audit
- `pytest --cov=braidcodec --cov-report=html --cov-fail-under=95`
- Per-module: algebra ≥98%, codec ≥95%, crypto ≥95%, cli ≥85%
- Add targeted tests for any uncovered branches

---

### Phase G: Release *(depends on ALL above)*

**Step 18** — Tag v0.1.0
- Verify all CI passes (lint, typecheck, test, coverage)
- Update CHANGELOG date from "Unreleased" to actual date
- `git tag v0.1.0` → `git push --tags`
- Verify release.yml triggers
- Verify benchmark.yml manual dispatch works

---

### Relevant Files

| Action | File | Purpose |
|--------|------|---------|
| Create | `tests/benchmarks/__init__.py` | Empty init |
| Create | `tests/benchmarks/bench_throughput.py` | Encode/decode benchmarks (~120 lines) |
| Create | `tests/benchmarks/bench_compression.py` | Compression benchmarks (~100 lines) |
| Create | `tests/benchmarks/bench_integrity.py` | Verify benchmarks (~80 lines) |
| Create | `.github/workflows/benchmark.yml` | Weekly benchmark CI (~30 lines) |
| Create | `.github/workflows/release.yml` | PyPI publish CI (~25 lines) |
| Create | `docs/index.md` | Docs landing page (~100 lines) |
| Create | `docs/theory.md` | Math foundations (~200 lines) |
| Create | `docs/architecture.md` | Public architecture (~150 lines) |
| Create | `docs/api.md` | API reference (~200 lines) |
| Create | `docs/benchmarks.md` | Benchmark guide (~80 lines) |
| Create | `examples/quickstart.py` | Minimal example (~20 lines) |
| Create | `examples/encode_file.py` | File encoding example (~40 lines) |
| Create | `examples/integrity_demo.py` | Integrity demo (~50 lines) |
| Modify | README.md | Hero, Mermaid, CLI, perf table, badges |
| Modify | CHANGELOG.md | All phase entries |
| Modify | conftest.py | 7 shared fixtures from PRD §9.2 |
| Modify | pyproject.toml | `--benchmark-disable` in addopts |

---

### Verification

1. `pytest tests/benchmarks/ --benchmark-enable -v` — all benchmarks run
2. `pytest tests/ -v --cov=braidcodec --cov-report=html --cov-fail-under=95` — meets threshold
3. `ruff check src/ tests/ examples/` — zero violations
4. `mypy src/braidcodec --strict` — zero errors
5. `python examples/quickstart.py` — runs clean
6. `python examples/encode_file.py` — runs clean
7. `python examples/integrity_demo.py` — runs clean
8. Mermaid diagrams render in GitHub markdown preview
9. README badges have correct URL structure
10. `benchmark.yml` structure matches PRD §10.1 spec
11. `release.yml` uses trusted publishing (OIDC, no API tokens)

---

### Decisions

- **Benchmark gating**: `--benchmark-disable` in pyproject.toml `addopts`; dedicated CI uses `--benchmark-enable`
- **`generators_per_block` for benchmarks**: 8 (tier 2 feasible, keeps CI fast)
- **Docs format**: Plain markdown + Mermaid — no Sphinx/mkdocs build for v0.1.0. GitHub renders natively. Defer mkdocs to v0.2.0.
- **No Markov Level 3**: Excluded per ROADMAP decisions
- **Release**: Tag-triggered via `release.yml`. Requires GitHub environment `pypi` configured for trusted publishing.
- **conftest**: Add only PRD §9.2 fixtures not already defined locally in test files

### Step Dependencies

```
Steps 1-5 (benchmarks) ─┐
Steps 6-7 (CI)         ─┤─ All parallel
Steps 8-12 (docs)      ─┤
Step 13 (examples)     ─┤
Steps 14-15 (README)   ─┘
                         │
          Step 16 (conftest) ── parallel
                         │
          Step 17 (coverage) ── depends on 1-5, 16
                         │
          Step 18 (release) ── depends on ALL
```