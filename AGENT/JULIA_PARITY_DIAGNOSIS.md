# Julia vs Python Reconstructive Parity Diagnosis

Date: 2026-03-16
Scope: Serialization + reconstruction parity diagnosis focused on the reconstructive path and O(1) target mechanics.

## 1. Inputs Used For This Diagnosis

### Julia/AGENT artifacts reviewed
- `AGENT/reconstruction_pipeline.jl`
- `AGENT/signature_reconstruction.jl`
- `AGENT/PHASES/serialization.jl`
- `AGENT/PHASES/PHASE8_Z.md`

### Python implementation reviewed
- `src/braidcodec/codec/encoder.py`
- `src/braidcodec/codec/decoder.py`
- `src/braidcodec/codec/schema.py`
- `src/braidcodec/codec/reconstructive_compact.py`
- `src/braidcodec/codec/reconstructive_solver.py`
- `src/braidcodec/codec/manifold.py`
- `src/braidcodec/codec/reconstructive_transform.py`
- `src/braidcodec/crypto/integrity.py`

### Executed validation
- `PYTHONPATH=src ... pytest tests/codec/test_encoder.py -k reconstructive -q` -> 9 passed
- `PYTHONPATH=src ... pytest tests/codec/test_decoder.py -k reconstructive -q` -> 7 passed
- `PYTHONPATH=src ... pytest tests/crypto/test_integrity.py -k reconstructive -q` -> 3 passed
- `PYTHONPATH=src ... pytest tests/cli/test_cli.py -k reconstructive -q` -> 5 passed
- `PYTHONPATH=src ... pytest -k reconstructive -q` -> 35 passed
- `PYTHONPATH=src ... pytest tests/benchmarks/bench_reconstructive_path.py --benchmark-enable -o 'python_files=bench_*.py' -q` -> 3 passed

## 2. What Julia O(1) Logic Is Trying To Do

Based on the stated math goal and AGENT Julia reconstruction files, the intended O(1)-class mechanism is:

1. Store global generative state, not per-oscillator dense state.
- Topology seed (braid generators) + global dynamics (`K_M`) + low-order statistics/moments.

2. Reconstruct many oscillator states by deterministic dynamics.
- `K_M(x) = kappa*x + eta*sin(x)` iteration in `AGENT/signature_reconstruction.jl`.
- Fixed-point verification and contraction checks are explicit (`L_CONTRACTION`, `FIXEDPOINT_TOL`).

3. Treat the stored artifact as a physics program, not a literal transcript.
- Reconstruction computes state from compact descriptors and checks fidelity metrics (`km_residual_*`, energy error, topology validity) in `AGENT/reconstruction_pipeline.jl`.

4. Keep fidelity gates explicit.
- Convergence, residual, and energy/topology checks are first-class validation outputs in the Julia reconstruction pipeline.

Important nuance:
- `AGENT/PHASES/serialization.jl` currently includes structures that are O(n) in raw representation (`signatures`, projection vectors). This appears to be a practical layer/transport implementation and not the pure theoretical compact endpoint.
- `AGENT/PHASES/PHASE8_Z.md` already acknowledges parity as incomplete and targets latent/topology/dynamics commitments as next block work.

## 3. Current Python Reconstructive Reality (Parity Baseline)

The Python reconstructive path is deterministic and tested, but presently operates as compact replay plus contract gates:

1. Encode in reconstructive mode emits zero blocks.
- `encoder.py` returns `blocks=tuple()` for reconstructive compact mode.

2. Decode reconstructs bytes from reconstructive program payload.
- `decoder.py` routes to `synthesize_reconstructive_bytes(...)` when reconstructive and zero-block.

3. Program types include templates and latent residual payloads.
- `reconstructive_compact.py` supports `repeat-text-v1`, `json-linear-items-v1`, `logs-seq-v1`, `latent-residual-v2/v3`.

4. Hypergraph/manifold/K_M are emitted as deterministic contract metadata.
- They are used for deterministic profiling/validation gates and payload contract integrity.
- They are not yet the sole latent state used for general reconstruction across arbitrary text/json/logs.

This is consistent with `AGENT/PHASES/PHASE8_Z.md` status notes (known non-parity gaps are already documented there).

## 4. Parity Matrix: Julia O(1) Intent vs Python Implementation

| Capability | Julia intent/status (AGENT) | Python status | Parity |
|---|---|---|---|
| Physics-program reconstruction framing | Present in reconstruction pipeline logic and docs | Partial: replay program + hard gates | Partial |
| Global dynamics gate (`K_M`, contraction) | Explicit and central | Implemented and enforced in payload validation | Good |
| General latent-manifold projection for arbitrary text/json/logs | Declared target, not fully complete | Not implemented yet | Gap |
| No residual transcript dependency | O(1) target implies this | Not met: latent residual stores compressed residual data | Gap |
| Template-free logs generalization | Targeted in roadmap | Not met: strict `logs-seq-v1` template path | Gap |
| Commitment rooted in latent/topology/dynamics object | Declared next alignment step | Not fully: commitment currently rooted in payload JSON + blocks | Gap |

## 5. Concrete O(1)-Critical Gaps In Python

1. Residual payload dependence (primary gap)
- `latent-residual-v2/v3` stores compressed residual bytes (`r85`/`r64`), which is still transcript-like storage.
- This prevents pure topology+dynamics-only representation for general inputs.

2. Logs path constrained to one sequence template
- `reconstructive_compact.py` logs branch raises `FormatError` when pattern is not strict sequential event form.
- There is no latent-residual fallback for logs right now.

3. Commitment scope does not bind raw side-channel payload bytes
- Compact payload marker `@` points to side-channel `rpb` data.
- Commitment is computed over bundled payload JSON and blocks, not raw side-channel bytes.
- Verified mutation experiment: appending whitespace to `rpb` kept `decode` and `verify` valid.

4. K_M thresholds are currently self-derived in emitted metadata
- Encoder sets threshold fields equal to observed diagnostics in the same encode run.
- This is strong for consistency/tamper checks but not yet an externally pinned quality floor.

## 6. Verified Issues To Track Alongside Parity Work

These issues should be tracked in the same workstream because they affect reconstructive trust semantics:

1. `rpb` commitment binding granularity
- Risk: semantically equivalent raw side-channel mutation may bypass commitment mismatch.
- Recommendation: include canonicalized side-channel bytes (or digest) in commitment root.

2. Logs generalization policy
- Risk: valid non-template logs hard-fail in reconstructive mode.
- Recommendation: add deterministic latent residual fallback for logs, then preserve template path as preferred fast path.

3. Strict-gate mode for K_M thresholds
- Risk: quality gate semantics remain relative if thresholds are self-emitted.
- Recommendation: add optional strict policy with fixed floor/ceiling thresholds from config/version profile.

## 7. Why This Is Not A Contradiction With Current Status

This diagnosis does not claim Python "failed" the current phase.
- The current Python state matches declared Phase 8 status in `AGENT/PHASES/PHASE8_Z.md`.
- O(1) is a target trajectory, not a completed claim in current Python implementation.
- Current implementation has strong determinism, hard validation gates, and measurable compactness progress.

## 8. Recommended Parity-Closure Order

1. Commitment-root upgrade
- Bind `rp1` canonical payload + canonical side-channel program bytes + topology/dynamics hash bundle.

2. Logs fallback generalization
- Keep `logs-seq-v1` fast path, add latent residual fallback for non-template logs.

3. Latent projection replacement work
- Reduce dependence on residual transcript by increasing reconstructability from latent/topology/dynamics commitments.

4. Strict gate profile
- Add fixed policy profile for `km_threshold_residual_max` and `km_threshold_valid_ratio`.

5. Bench expansion
- Add non-template logs and adversarial text/json sets to benchmark matrix, not only template-friendly corpora.

## 9. Bottom Line

Julia-side O(1) logic is a physics-program compression thesis: store compact global dynamics/topology constraints, regenerate many states deterministically.

Python reconstructive mode is currently a deterministic compact replay system with strong contracts and gates, plus partial latent/manifold instrumentation.

The largest parity blockers are:
- residual transcript dependence,
- template-constrained logs,
- commitment-root granularity for side-channel payload bytes.

These are implementation parity gaps, not conceptual disagreement on architecture direction.

## 10. Equation-Discovery Track (Core Generalization Path)

This section captures the key direction: the dominant parity blocker is not logs-specific handling, but lack of a general mechanism to discover and encode ruling equations for arbitrary semantic data classes.

### Problem Restatement

Current Python reconstructive mode can replay exact data for many practical inputs, but general latent-manifold projection is blocked because there is no semantic equation-discovery layer that maps arbitrary input structure into compact ruling dynamics and then into braid equations.

### Proposed Mechanism

1. Semantic lifting
- Convert raw input to one or more deterministic observation spaces where governing structure is learnable.
- Examples: token streams, AST/IR traces, compiler action sequences, byte n-gram dynamics, control/data flow motifs.

2. Sparse equation discovery
- Use deterministic sparse regression over a fixed basis library to identify candidate ruling equations.
- Candidate methods: SINDy-style sparse regression with Fourier and polynomial dictionaries, constrained by stability and contraction priors.

3. Equation validation
- Validate discovered equations against held-out segments and reconstruction rollouts.
- Keep only equations satisfying strict residual, stability, and exactness-assisted replay criteria.

4. Equation-to-braid compilation
- Compile accepted ruling equations into a compact braid-program representation (topology + dynamics commitments).
- This is the missing bridge between semantic law discovery and O(1)-class compact representation.

5. Exactness authority
- Use deterministic replay from compiled ruling equations plus checksum match as final authority.
- If exactness fails, fall back to residual channel (as currently implemented) and record the case for future library growth.

### Candidate Library Strategy

To avoid per-input discovery overhead:

1. Build a ruling-equation candidate library keyed by semantic signatures.
- Keys can include domain, tokenizer profile hash, coupling/topology descriptors, and structural fingerprints.

2. Retrieval-first policy
- Attempt nearest/compatible candidate retrieval before running new discovery.

3. Adaptation policy
- If retrieval is close but not exact, run constrained fine-tuning of equation coefficients.

4. Promotion policy
- Promote new equations into the library only when they pass reproducibility and exactness gates across multiple samples.

This supports the expected pattern: semantically similar corpora (same language family, same compiler/toolchain class, same document class) amortize discovery cost and move toward compact generative encoding.

### Practical O(1) Interpretation

For exact reconstruction, true strict O(1) for all possible inputs is impossible in the information-theoretic sense.
However, for structured data classes with reusable ruling equations, effective near-constant descriptor size per semantic family is a realistic target, with residual fallback only for novelty/noise-heavy cases.

### Why This Completes Julia Parity Direction

This mechanism aligns directly with the Julia thesis:
- store governing dynamics/topology,
- regenerate state deterministically,
- keep exactness as a hard gate,
- reserve transcript-like residuals for cases where ruling discovery is insufficient.

In short: equation discovery is the missing generalization layer that transforms reconstructive compact replay into a true semantic physics-program codec.

### Execution Addendum

Recommended implementation track after current parity fixes:

1. Define deterministic basis library and sparse regression contract.
2. Implement discovery module with reproducible solver settings.
3. Add equation-to-braid compiler and commitment schema extension.
4. Add candidate library store/retrieval APIs and promotion tests.
5. Add benchmark slices by semantic family to measure discovery hit-rate, residual fallback rate, and descriptor compactness.

## 11. Latest Measured Serialization Compaction (Phase 9)

Measured on 2026-03-16 after compact braid payload key reduction (`e`,`r`,`g`) with backward-compatible parsing:

- Legacy discovered-equation payload (`discovered-equation-v1`): 247 bytes
- Prior braid payload revision (`discovered-braid-equation-v1`): 37 bytes
- Current braid payload revision (`discovered-braid-equation-v1`): 26 bytes

Computed reductions:

- Legacy -> current braid: 89.4737% smaller
- Prior braid -> current braid: 29.7297% smaller

Benchmark telemetry alignment (discovery-mix case sidechannel bytes):

- Previous measured `reconstructive_program_payload_bytes`: 38
- Current measured `reconstructive_program_payload_bytes`: 27
- Benchmark-side reduction: 28.9474%

Interpretation:

The newest pass materially improves descriptor compactness while preserving deterministic replay and backward decode compatibility, tightening practical parity toward Julia-style compact physics-program serialization.

## 12. Input vs Encoded Size (What You Asked)

Measured on 2026-03-16 using full `EncodedStream` artifact sizes (not only program payload fields):

- discoverable-constant-2048 (`text`, discovery enabled)
	- Input: 2048 bytes
	- Wire encoded (`to_bytes`): 4227 bytes
	- HDF5 encoded (`to_hdf5_bytes`): 22717 bytes
	- Wire reduction: -106.3965% (i.e. wire is larger than input)
	- HDF5 reduction: -1009.2285%

- fallback-freeform (`text`, discovery enabled)
	- Input: 1540 bytes
	- Wire encoded: 5477 bytes
	- HDF5 encoded: 23956 bytes
	- Wire reduction: -255.6494%
	- HDF5 reduction: -1455.5844%

- core-text case
	- Input: 3200 bytes
	- Wire encoded: 4795 bytes
	- Wire reduction: -49.8438%

- core-json case
	- Input: 4246 bytes
	- Wire encoded: 5530 bytes
	- Wire reduction: -30.2402%

- core-logs case
	- Input: 8189 bytes
	- Wire encoded: 5597 bytes
	- Wire reduction: +31.6522%

Interpretation:

- The discovered-braid payload mechanism is very compact, but current full-stream envelope overhead dominates at small/medium inputs.
- This is why we can observe excellent descriptor compaction yet still have negative overall wire reduction in several cases.

## 13. Near-O(1) Parity Delta vs Julia (Post-Mechanism)

### What is now close to O(1)

- Discovered program sidechannel (`rpb`) for constant-family inputs is nearly size-invariant with input length.
- Empirical scaling (`n`, `wire_bytes`, `rpb_bytes`) for `A^n`:
	- 128 -> 4234, 26
	- 256 -> 4233, 26
	- 512 -> 4235, 26
	- 1024 -> 4234, 27
	- 2048 -> 4227, 27
	- 4096 -> 4232, 27
	- 8192 -> 4229, 27

This indicates near-constant descriptor size and near-constant full wire size (with a large constant factor).

### Why parity is still not Julia-like near-O(1) in practice

1. Python stream envelope is metadata-heavy.
- Discoverable 2048-byte case had `54` metadata items.
- `rp1` alone was `1689` bytes; value-byte sum was `3061` before structural framing.
- Several 64-byte hash fields and large fields (`km_seed_vector`, `fidelity_bundle_v1`) inflate constant overhead.

2. Python currently stores a rich audit contract inline in every stream.
- Julia pipeline conceptually emphasizes compact governing state + deterministic reconstruction checks.
- Python includes many diagnostics/check hashes as transport payload, which is safe but expensive.

3. Fallback path remains transcript-like.
- Freeform fallback had `rp1` = 2229 bytes and `rpb` = 137 bytes for a 1540-byte input.
- Residual dependence prevents near-O(1) behavior outside discoverable families.

4. HDF5 container fixed overhead is very high for these sample sizes.
- HDF5 remains useful for tooling/interoperability, but it is not the compactness-optimal transport for small reconstructive payloads.

### What needs to change to get closer to Julia-style near-O(1)

1. Two-tier payload contract:
- `compact transport` tier with minimal reconstructive fields only.
- `audit bundle` tier optional/sidecar for deep diagnostics.

2. Compact binary reconstructive header:
- Replace large JSON `rp1` with tightly packed msgpack/binary schema for hot-path fields.
- Dictionary-code repeated key sets and profile references.

3. Profile indirection:
- Replace repeated per-stream hashes/threshold literals with versioned profile IDs and one short profile digest.

4. Keep discovered-braid as primary path, but widen equation families and library hit-rate.
- The more inputs map to reusable ruling equations, the more traffic stays in near-constant descriptor mode.

5. Residual minimization roadmap:
- Shift residual from default fallback toward rare exception path via stronger semantic equation discovery coverage.

## 14. Metadata-Zero Target (Braid-Only Storage)

Given the Julia-side measurement (`59` bytes total descriptor for very large `n`), the correct parity target is a braid-only transport where reconstructive state is encoded entirely as braid equations, with no large per-stream JSON contract.

### Clarification on "no metadata"

- For strict transport decoding, only framing bytes are still required (container magic/version/checksum and block boundaries).
- Reconstructive semantics should be self-describing inside the braid word itself (family id, rollout, coefficients, initial state, profile id), not duplicated in external metadata maps.

### Required architectural shift

1. Move from metadata-driven decode to braid-word-driven decode.
- Today decode needs `rp1`/`rpb` contract fields.
- Target decode should parse a canonical braid program header directly from generators.

2. Define a canonical braid program wire grammar.
- Prefix: fixed braid sentinel + schema version.
- Body: equation family code, rollout length, parameter tuple, optional profile code.
- Integrity: commitment/checksum bound to canonical braid payload bytes.

3. Eliminate high-cardinality per-stream metadata from hot path.
- Remove inline `km_seed_vector`, full fidelity bundle JSON, repeated hash literals from transport payload.
- Keep those only in optional audit sidecar (debug/profile mode), not default compact transport.

4. Keep exact fidelity contract.
- Reconstruction authority remains: deterministic synthesis from braid program + byte-exact checksum match.
- If synthesis fails exactness, route to explicit fallback mode (not compact default).

### Practical near-O(1) implication

- For families with reusable ruling equations, descriptor size becomes near-constant w.r.t. input length.
- Remaining byte growth should be from container framing only, not semantic metadata duplication.
- This is the direct path to Julia-style "bits per oscillator" behavior.

### Execution order to realize this

1. Add `braid-program-v2` canonical generator grammar and parser.
2. Add `compact_transport=true` route: emit braid program in blocks, omit `rp1`/`rpb` from default stream metadata.
3. Bind commitment root to canonical braid payload bytes.
4. Move diagnostics/fidelity details to optional sidecar profile (`audit_bundle`).
5. Benchmark scaling (`n=2^k`) and report fixed overhead in bytes and bits/oscillator.

## 15. Implemented Delta: Compact Transport (Braid Program Hot Path)

Status (2026-03-16): implemented in Python as an opt-in path with two compact transport variants.

### What was implemented

1. New reconstructive compact transport policy in encoder:
- `reconstructive_compact_transport = disabled|enabled|required`.

2. Compact metadata emission for hot path:
- Primary compact transport (`rt=ps1`): sidechannel compact payload with short type code (`rtt`) and minimal metadata + commitment.
- Lean transport (`rt=ps2`): sidechannel compact payload with short type code and no reconstructive commitment field (checksum-authoritative path), derived from Julia-reference minimization principles.
- Experimental embedded transport (`rt=pg1`): compact payload encoded into regular braid stream blocks (no payload sidechannel).

3. Decoder + integrity support for compact transport:
- Added compact transport parser/validator path.
- Commitment validation now supports compact form (`compact:<rt>` commitment root + sidechannel bytes).
- Exact decode remains checksum-authoritative.

4. Backward compatibility preserved:
- Legacy full reconstructive payload path (`rp1` / `reconstructive_payload_v1`) remains unchanged by default.
- Compact transport is opt-in and supports all reconstructive program families.

### Measured impact

Measured mode comparison (`generators_per_block=8`):

1. discoverable-constant-2048 (`input=2048`)
- `disabled`: wire `4227`, h5 `22717`, metadata keys `54`
- `enabled`/`required` (`rt=ps1`): wire `330`, h5 `18506`, metadata keys `5`
- `lean` (`rt=ps2`): wire `238`, h5 `18411`, metadata keys `4`
- `embedded` (`rt=pg1`): wire `904`, h5 `20837`, metadata keys `3`, blocks `1`

2. fallback-freeform (`input=1540`)
- `disabled`: wire `5477`, h5 `23956`, metadata keys `53`
- `enabled`/`required` (`rt=ps1`): wire `440`, h5 `18613`, metadata keys `5`
- `lean` (`rt=ps2`): wire `348`, h5 `18518`, metadata keys `4`
- `embedded` (`rt=pg1`): wire `1196`, h5 `20946`, metadata keys `3`, blocks `1`

3. core-json (`input=4246`)
- `disabled`: wire `5530`, h5 `24009`, metadata keys `53`
- `enabled`/`required` (`rt=ps1`): wire `339`, h5 `18512`, metadata keys `5`
- `lean` (`rt=ps2`): wire `247`, h5 `18417`, metadata keys `4`
- `embedded` (`rt=pg1`): wire `898`, h5 `20835`, metadata keys `3`, blocks `1`

Interpretation:

- `ps1` materially improves end-to-end compactness and is now the practical default compact path.
- `ps2` is currently the smallest practical transport in this codebase and most aligned with Julia-style minimal serialization envelope.
- `pg1` now uses one tier-1 transport block and is no longer catastrophic, but remains larger than sidechannel compact modes under current container overhead.
- Therefore, current best path toward Julia-style near-O(1) in this codebase is minimal sidechannel compact transport (`ps1`) plus further envelope reduction.

### Remaining gap to Julia-style endpoint

- Current winning compact path (`ps1`) is still metadata-framed and sidechannel-based.
- Current smallest compact path is now `ps2` (Julia-lean sidechannel), still metadata-framed and sidechannel-based.
- Pure braid-block semantic framing (`pg1`) exists and is improved (single tier-1 block), but still larger than `ps1` in current container schema.
- True Julia-style "braid-only and tiny" endpoint therefore requires a lean transport container for embedded program words (or bypassing heavy per-block invariants for transport-only blocks).
