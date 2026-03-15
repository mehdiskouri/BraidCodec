
## Plan: Topology-First Encoding v2 Rollout

Implement a default-on `chunk -> hypergraph/topological layers -> braid synthesis` pipeline with immediate wire-format `v2`, additive topology integrity shielding, and benchmark-driven performance mitigation targeting the 1MB bottleneck you observed.

**Baseline and target performance table (from current benchmark run)**

| Case | Current mean time | Current throughput (MB/s) | Target mean time | Target throughput (MB/s) | Goal |
|---|---:|---:|---:|---:|---|
| `encode[16-6-10KB]` | 56.12 s | 0.00018 | <= 8.0 s | >= 0.00128 | Remove catastrophic regime |
| `encode[16-6-100KB]` | 553.74 s | 0.00018 | <= 80.0 s | >= 0.00128 | >= 6.9x faster in stalled profile |
| `encode[16-5-100KB]` | 3.34 s | 0.03070 | <= 3.80 s | >= 0.02695 | No meaningful regression in healthy regime |
| `encode[16-5-1MB]` | 33.17 s | 0.03161 | <= 36.5 s | >= 0.02873 | Preserve 1MB baseline band |
| `encode[8-3-1MB]` | 98.87 s | 0.01061 | <= 90.0 s | >= 0.01165 | Modest uplift in low-gpb-heavy path |

Notes:
- Throughput is computed as `input_bytes / mean_seconds / 1e6`.
- Initial acceptance is based on p50 mean from repeatable runs, then hardened with variance thresholds before default CI gating.

**CI go/no-go acceptance checks (subbranch stage)**

1. `stress-regime` pass gate (must hold for promotion discussion):
	- `encode[16-6-100KB]` mean time `<= 80.0 s`.
	- `encode[16-6-10KB]` mean time `<= 8.0 s`.
	- No run may exceed `2x` target mean in the same benchmark job.
1. `core-throughput` regression guard (must hold every subbranch run):
	- `encode[16-5-100KB]` throughput `>= 0.02695 MB/s`.
	- `encode[16-5-1MB]` throughput `>= 0.02873 MB/s`.
	- `encode[8-3-1MB]` throughput `>= 0.01061 MB/s` until uplift target is achieved.
1. Variance stability requirements before moving checks to default CI gating:
	- Coefficient of variation (`stddev / mean`) `<= 0.12` for core-throughput cases.
	- At least `5` rounds collected for each promoted gating benchmark.
1. Promotion criteria from subbranch CI to default gating:
	- All stress-regime gates pass for `3` consecutive CI runs.
	- Core-throughput regression guard passes for `5` consecutive CI runs.
	- Runtime budget for benchmark job remains below agreed CI wall-clock threshold.

**Steps**
1. Phase 1: Baseline and branch setup  
1. Preserve current long-run benchmark outputs as the baseline matrix for post-change comparison.  
1. Create and use a dedicated implementation subbranch for this full scope.  

2. Phase 2: Encoding architecture extension  
1. Insert a preprocessing stage after chunk generation and before braid construction.  
1. Define deterministic `BinaryFixedPointSignature` derivation from chunk bytes, then deterministic sparsity-first hypergraph construction and topological layering.  
1. Derive generators from `(layer_n_chunks, dt_scale, layer_index, BFPS-derived terms)` with strict bounds to valid braid generator indices.  
1. Keep an explicit compatibility fallback mode available, but ship the topology path as default on this branch.  

3. Phase 3: Wire format and schema  
1. Bump stream format to `v2` and add topology metadata fields (mode/version, layer descriptors, Morton keyed index data, BFPS precision/version, dt-scale params).  
1. Implement dual-read support so `v1` streams still decode via legacy path.  
1. Enforce strict schema validation for malformed `v2` topology metadata.  

4. Phase 4: Integrity shield extension  
1. Add per-layer topology integrity commitments using layer JP/topology values keyed with Morton layer context and index binding.  
1. Aggregate per-layer commitments into stream-level integrity proof material.  
1. Keep this channel additive while preserving checksum/digest authority as the final acceptance gate.  

5. Phase 5: Performance mitigation and parallelism strategy  
1. Shift parallelization strategy from per-small-chunk futures to layer-aware batching to reduce `ProcessPoolExecutor` IPC overhead.  
1. Add regime-aware scheduling with an explicit dispatch policy:
	- `serial` path for small or low-cost batches where pool startup/IPC dominates.
	- `thread` path for shared-memory preprocessing stages that avoid pickling overhead.
	- `process` path for expensive contraction/invariant workloads with sufficiently large task granularity.
1. Implement cost-model based path selection before dispatch:
	- Estimate per-block/per-layer cost using `n_strands`, `gpb`, layer count, estimated matrix dimension, and expected invariant tier.
	- Predict queueing/IPC overhead versus useful work and select execution mode from the regime policy.
	- Persist per-run timing telemetry for model recalibration on subsequent runs.
1. Improve reuse of expensive algebra/invariant computations with stable cache boundaries suited to the new layered path.  

6. Phase 6: Benchmarks and CI progression  
1. Expand benchmark instrumentation into stage timings: preprocess, graph build, layer synthesis, encode core, invariant compute, serialize.  
1. Split benchmark strategy into two suites:
	- `core-throughput` suite for regular CI (stable parameter set, repeatable rounds, bounded runtime).
	- `stress-regime` suite for pathological combinations (including `16-6-100KB`) running in subbranch CI reporting mode.
1. Run legacy-vs-topology comparisons across `10KB/100KB/1MB` and current strands/GPB grid, with explicit regime labels in output artifacts.
1. Keep expanded benchmarks active in subbranch CI first (non-blocking trend reports plus threshold alerts); promote selected checks to default gating only after variance and runtime stabilize and target table is met.

7. Phase 7: Tests, docs, and release preparation  
1. Add deterministic BFPS/layer construction tests, v2 schema roundtrip tests, v1 compatibility tests, and topology-channel corruption tests.  
1. Update architecture/API/benchmark docs and README to reflect v2, default topology pipeline, and 5+1 integrity model.  
1. Publish migration and release notes clarifying defaults, compatibility behavior, and performance tradeoffs.  

**Relevant files**
- encoder.py — insertion point for default topology preprocessing and scheduling logic.  
- chunker.py — chunk-to-preprocess interface boundaries.  
- schema.py — `v2` stream fields, serialization, dual-version read behavior.  
- decoder.py — v1/v2 dispatch and topology-aware verification path.  
- integrity.py — additive topology integrity channel integration.  
- braid_equations.py — JP/invariant compute reuse points.  
- bench_throughput.py — baseline throughput anchor and extended metrics.  
- bench_compression.py — preprocessing/compression synergy validation.  
- bench_integrity.py — topology-channel detection and latency behavior.  
- test_schema.py — v2 and backward-compat validation.  
- test_roundtrip.py — end-to-end v1/v2 fidelity.  
- test_integrity.py — topology integrity scenarios.  
- api.md — updated API contracts and mode semantics.  
- architecture.md — new preprocessing/topology layer in system model.  
- benchmarks.md — methodology and interpretation updates.  
- README.md — high-level pipeline and performance/integrity claim updates.  
- CHANGELOG.md — `v2` and branch-phase release notes.  

**Verification**
1. Full .venv test suite with both `v1` and `v2` fixtures, no legacy regressions.  
1. Determinism checks: same input/key/config produces identical BFPS, layers, generators, and integrity artifacts.  
1. Corruption campaign: layer index/Morton/topology metadata tampering must fail topology channel and report cleanly.  
1. Benchmark delta analysis at `1MB` must show reduction in known bottleneck components, not just shifted cost.  
1. Subbranch CI must emit stable benchmark artifacts suitable for eventual default gating promotion.  

**Decisions Captured**
- Default behavior on this subbranch is topology-first preprocessing.  
- Immediate wire-format bump to `v2` with legacy `v1` read compatibility.  
- Topology integrity channel is additive; checksum/digest remains authoritative.  
- Benchmark integration starts in subbranch CI, then graduates to default gating once outcomes are stable and favorable.  

Plan has been saved to `/memories/session/plan.md` and is ready for execution handoff.