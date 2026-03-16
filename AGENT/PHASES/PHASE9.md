## Plan: Equation-Discovery Parity Track

Implement Julia-parity-oriented reconstructive generalization by adding deterministic equation discovery, equation-to-braid compilation, and candidate-library reuse, while preserving exact decode guarantees and backward compatibility.

**Steps**
1. **Foundation (Dependencies + Profiles)**  
Add optional discovery dependencies in pyproject.toml: `pysindy`, `sympy`, `scikit-learn`, `scipy`, `joblib`.  
Introduce a versioned `discovery_profile_id` to pin tolerances, iterations, seeds, and numerical behavior.  
*depends on none*

2. **Discovery Core API**  
Create `src/braidcodec/codec/equation_discovery.py` with deterministic contracts:
- `DiscoveredEquation`
- `DiscoveredEquationSet`
- `DiscoveryDiagnostics`
- `discover_equations(...)`  
Inputs come from existing tokenizer/oscillator/manifold outputs; outputs include symbolic hash, coefficients, validation metrics, and stability flags.  
*depends on 1*

3. **Candidate Library (Retrieval-First)**  
Create `src/braidcodec/codec/equation_library.py`:
- `CandidateLibraryEntry`
- `CandidateLibraryStore`
- exact/near-match retrieval
- promotion rules after successful replay  
Semantic signature includes domain kind + tokenizer hash + normalization hash + manifold/coupling hash.  
*depends on 1; parallel with 2*

4. **Program Type Extension**  
Extend reconstructive_compact.py with `discovered-equation-v1`:
- symbolic basis id
- coefficient payload
- initial conditions
- rollout controls
- canonical projection id  
In synthesis path, attempt deterministic rollout -> byte projection -> exactness pre-check.  
Fallback chain: discovered-equation -> existing latent residual v2/v3.  
*depends on 2 and 3*

5. **Encoder Integration**  
Integrate into `_build_reconstructive_metadata(...)` in encoder.py:
- retrieval-first from library
- discovery on miss
- compile `discovered-equation-v1`
- emit telemetry (hit/miss, failure reason, diagnostics)  
Add knobs:
- `reconstructive_discovery_enabled`
- `reconstructive_discovery_required`
- `reconstructive_library_path`
- `reconstructive_discovery_timeout_ms`
- `reconstructive_strict_gate_profile`  
*depends on 4*

6. **Schema + Commitment V2**  
Update schema.py for discovery metadata while preserving `rp1`/legacy and `rpb` alias compatibility.  
Upgrade commitment semantics to bind:
- canonical payload JSON digest
- canonical side-channel program bytes digest
- topology/dynamics/discovery hashes  
Add `reconstructive_commitment_v2`, while continuing legacy validation.  
*depends on 5*

7. **Decoder + Integrity Enforcement**  
Update decoder.py to validate and execute `discovered-equation-v1` with strict gates.  
Update reconstructive_solver.py for strict profile-driven thresholds.  
Update integrity.py for commitment-v2 verification and discovery-specific failure taxonomy entries.  
*depends on 6*

8. **CLI/API Surface**  
Update main.py with:
- `--reconstructive-discovery enabled|disabled|required`
- `--reconstructive-library PATH`
- `--strict-gates PROFILE`
- `--discovery-timeout-ms`  
Expose config in public API exports (__init__.py) without breaking old usage.  
*depends on 7*

9. **Parity-Critical Tests**  
Add:
- `tests/codec/test_reconstructive_equation_discovery.py`
- `tests/codec/test_reconstructive_commitment_v2.py`
- `tests/codec/test_reconstructive_generalization.py`  
Extend:
- test_encoder.py
- test_decoder.py
- test_integrity.py
- test_cli.py  
Focus on determinism, retrieval-first behavior, strict-gate pass/fail, and tamper rejection.  
*depends on 8; parallel with 10*

10. **Benchmarks + Parity Scorecard**  
Extend bench_reconstructive_path.py to track:
- library hit-rate vs miss-rate
- discovery overhead
- exactness before fallback
- fallback frequency by domain family  
Update parity tracking docs in AGENT with milestone metrics.  
*depends on 8 and 9*

11. **Docs + Rollout**  
Update:
- JULIA_PARITY_DIAGNOSIS.md
- PHASE8_Z.md
- architecture.md
- api.md
- benchmarks.md  
Rollout: feature-flag default off -> canary corpora -> default on after gates are met.  
*depends on all prior steps*

**Relevant files**
- pyproject.toml - add optional heavy scientific dependency group.
- encoder.py - discovery orchestration and metadata emission.
- decoder.py - discovered-equation decode path and strict gating.
- schema.py - schema extensions + backward compatibility.
- reconstructive_compact.py - new program type and deterministic synthesis.
- reconstructive_solver.py - strict profile threshold enforcement.
- integrity.py - commitment-v2 checks and taxonomy expansion.
- main.py - new discovery/strictness flags.
- `/workspace/BraidCodec/src/braidcodec/codec/equation_discovery.py` - new discovery module.
- `/workspace/BraidCodec/src/braidcodec/codec/equation_library.py` - new candidate library module.
- `/workspace/BraidCodec/tests/codec/test_reconstructive_equation_discovery.py` - deterministic discovery tests.
- `/workspace/BraidCodec/tests/codec/test_reconstructive_commitment_v2.py` - side-channel binding tamper tests.
- `/workspace/BraidCodec/tests/codec/test_reconstructive_generalization.py` - broad-domain semantic generalization tests.
- bench_reconstructive_path.py - parity benchmark metrics.

**Verification**
1. `PYTHONPATH=src python -m pytest tests/codec/test_reconstructive_equation_discovery.py -q`
2. `PYTHONPATH=src python -m pytest tests/codec/test_reconstructive_commitment_v2.py -q`
3. `PYTHONPATH=src python -m pytest -k reconstructive -q`
4. `PYTHONPATH=src python -m pytest bench_reconstructive_path.py --benchmark-enable -o 'python_files=bench_*.py' -q`
5. Acceptance gates:
- exact decode remains 100% for accepted outputs
- commitment-v2 rejects side-channel tampering
- discovered-equation usage rises on repeated semantic families
- residual fallback rate declines on structured corpora

**Decisions**
- Chosen stack: Scientific-heavy (`PySINDy + sympy + scikit-learn + scipy`).
- Primary target: semantic equation discovery generalization (not logs-specific optimization).
- Residual path remains correctness fallback.
- Backward compatibility for existing reconstructive aliases is mandatory.
- O(1) remains aspirational and measured via fallback reduction + compact-equation adoption, not claimed upfront.

Saved this plan to `/memories/session/plan.md`. If you want, I can now refine this into a sprint-by-sprint execution schedule (for example: Week 1-2 core plumbing, Week 3-4 integration, Week 5 hardening/bench parity).

## Implementation Progress (2026-03-16)

Completed in code:
- Added deterministic discovery module: `src/braidcodec/codec/equation_discovery.py`.
- Added retrieval/promotion library store: `src/braidcodec/codec/equation_library.py`.
- Integrated retrieval-first discovery flow in `src/braidcodec/codec/encoder.py` with exactness-gated fallback.
- Added compact decode support for `discovered-equation-v1` in `src/braidcodec/codec/reconstructive_compact.py`.
- Upgraded reconstructive commitment to bind side-channel payload bytes with legacy compatibility in `src/braidcodec/codec/schema.py`.
- Exposed CLI flags for discovery/library/profile in `src/braidcodec/cli/main.py`.
- Added optional scientific-heavy extra in `pyproject.toml` as `reconstructive_discovery`.
- Added strict profile plumbing: `default-v1` (observed thresholds) and `strict-v1` (fixed thresholds with encode-time enforcement).

Completed tests:
- `tests/codec/test_reconstructive_equation_discovery.py`
- `tests/codec/test_reconstructive_commitment_v2.py`
- `tests/codec/test_reconstructive_generalization.py`
- Extended benchmark telemetry in `tests/benchmarks/bench_reconstructive_path.py` with discovery/fallback/library markers.

Current status:
- `PYTHONPATH=src /workspace/BraidCodec/.venv/bin/python -m pytest -q -k reconstructive` -> 43 passed.
- `PYTHONPATH=src /workspace/BraidCodec/.venv/bin/python -m pytest -q tests/benchmarks/bench_reconstructive_path.py --benchmark-enable -o 'python_files=bench_*.py'` -> 3 passed.

Next rigorous targets:
1. Replace deterministic byte-family discovery core with PySINDy/scikit-learn/scipy-backed basis search while keeping deterministic profile pinning.
2. Add decoder/integrity strict profile annotations to failure taxonomy (`strict-gate-profile`, `discovery-required-failure`).
3. Add benchmark suites that measure repeated-corpus library hit-rate growth and residual fallback-rate reduction over sequential runs.