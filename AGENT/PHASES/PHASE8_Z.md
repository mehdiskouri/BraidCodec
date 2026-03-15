
## Plan: Frequency-Manifold Reconstructive Mode

Deliver a deterministic `reconstructive` codec mode for `Text/JSON/Logs` that stores compact manifold state (not explicit token sequence), reconstructs via K_M fixed-point iteration, and enforces exact byte fidelity by checksum plus topology/coherence constraints. This merges the baseline reconstructive plan and its refinements: frequency-based tokenization, oscillator-constrained normalization, and staged binary portability.

**Implementation status**
- Phase A: Completed in code/docs.
	Deliverables: reconstructive metadata contract validator and architecture contract documentation.
- Phase B: Completed in code/tests.
	Deliverables: `FrequencyTokenizerV1`, canonical text/json/log tokenization, frequency-bin lattice defaults, and required tokenizer metadata emission.
- Phase C: Completed in code/tests.
	Deliverables: `OscillatorNormalizationV1`, deterministic bounded transforms (omega/amplitude/phase/scale), serialized normalization profile hash, and hard-fail input policy.
- Phase D: Completed in code/tests (primitives).
	Deliverables: deterministic hypergraph builder, compact manifold state fitting, state hashing/metadata, and seed-vector extraction for fixed-point initialization.
- Phase E foundation: Completed in code/tests.
	Deliverables: `algebra/fixedpoint.py` K_M deterministic iterator, contraction guard, diagnostics, and verification helpers.

- Phase F: Completed in code/tests.
	Deliverables: `preprocessing_mode="reconstructive"` encoder entry path, domain inference/override, deterministic tokenizer+normalizer+manifold+K_M metadata emission, optional pre-tokenized ingress (`reconstructive_tokenization`) to skip duplicate tokenizer work when caller already has `TokenizationResult`, compact `reconstructive_payload_v1` schema layout, strict metadata/payload contracts, dedicated reconstructive decode route with payload-seeded inverse mapping + hard validation gates, and conditional container policy (`--container auto|wire|hdf5`) selected by serialized size or explicit override.
	Status note: reconstructive compact mode now emits zero block payload (`blocks=tuple()`), and decode/verify reconstruct bytes from compact program metadata for the supported Text/JSON/Logs scenarios.
- Phase G: Completed in code/tests.
	Deliverables: integrity channel treats reconstructive metadata/payload/commitment violations as structural failures; reconstructive commitment is emitted by encoder and validated by both decode hard-gates and integrity verification; CLI exposes reconstructive preprocessing/domain flags and explicit `--diagnostics` verification output; explicit failure taxonomy is surfaced for decode/verify (`contraction`, `convergence`, `invariant`, `checksum`, `reconstructive-contract`, `structural`).
- Phase H: Completed for lightweight scope in code/tests/benchmarks.
	Deliverables: determinism/fidelity/negative-path tests across reconstructive schema+decode+integrity+CLI; lightweight reconstructive benchmark suite for `Text/JSON/Logs` with K_M diagnostics telemetry (`km_residual_max`, `km_residual_mean`, `km_iters_mean`, `km_valid_ratio`).
- Phase I: Planned, not implemented yet.

**Recent rigor updates (post-Phase H)**
- Added direct pre-tokenized reconstructive ingress (`reconstructive_tokenization`) so callers can feed manifold/hypergraph stages without duplicate tokenizer work.
- Added `latent-residual-v1` compact replay fallback (zlib residual correction) for non-template text/json cases to preserve exactness without raw full-literal template replay.
- Removed duplicated top-level `reconstructive_program_payload` serialization (payload remains authoritative inside `reconstructive_payload_v1`), reducing wire overhead significantly while keeping commitment/checksum gates.
- Added deterministic fidelity bundle proxies in metadata: `fidelity_energy`, `fidelity_topology`, `fidelity_coherence`, `fidelity_bundle_v1`.
- Upgraded compact residual program to `latent-residual-v2` with coupling-aware adaptive codec selection (`zlib`/`bz2`/`lzma`) and deterministic spectral predictor support (decode remains backward-compatible with `latent-residual-v1`).

**Observed impact snapshot (NeelNanda/pile-10k subset, ~20k tokens, 128006 input bytes)**
- `program_type=latent-residual-v2`, bitwise exact decode + `verify.valid=True`.
- Wire container: `64204` bytes (`-49.84%` vs input).
- HDF5 container: `82693` bytes (`-35.40%` vs input).
- Post `latent-residual-v3` segmented implementation benchmark (same subset/config) still selected `latent-residual-v2` as smaller serialized payload for this corpus slice; exact decode + verify validity unchanged.
- `latent-residual-v3` path remains implemented, decode-compatible, and quality-gated for cases where segmented heterogeneous residual coding wins on payload size.
- Added `nnz`-aware compact fitting signal path (coupling `nnz` threaded into predictor/codec scoring) and behavior-sensitive tests; corpus benchmark remained at the same measured ratio on this slice (`wire=64204`, `h5=82693`), indicating no immediate size gain from signal injection alone.
- Diagnostic conclusion: primary remaining compaction limiter on this workload is payload representation overhead (JSON/base64 envelope) rather than residual predictor selection quality.
- Implemented lower-overhead residual payload envelope (`base85` with backward-compatible `base64` decode fallback) for latent residual programs.
- Post-envelope benchmark on the same subset/config improved materially with fidelity unchanged:
	- Wire container: `60530` bytes (`-52.71%` vs input), improved from `64204`.
	- HDF5 container: `79019` bytes (`-38.27%` vs input), improved from `82693`.
	- Program remained `latent-residual-v2` (`codec=bz2-xor-v1`, `predictor=zero-v1`), `exact decode=true`, `verify.valid=true`.
- Added Morton-index coupling alongside `nnz` in latent residual scoring/predictor shaping (segment-local fitting now mixes coupling density/radius/nnz and Morton lane/index signal deterministically).
- Post-Morton benchmark on the same subset/config preserved the base85 gain plateau (`wire=60530`, `h5=79019`) with exact fidelity and valid verification, establishing no-regression while broadening the adaptive search signal.
- Reintegrated `nnz_bits` into topology synthesis/recovery v2 (with Morton retained): encoder now propagates per-block `topology_nnz_bits` through v2 transforms, and decoder/compressor/integrity v2 recovery paths consume the same `nnz_bits` signal.
- Compatibility status: full type/lint/tests remain green; reconstructive benchmark remained stable at the improved envelope baseline (`wire=60530`, `h5=79019`, exact decode + verify valid).
- Added compact-key latent residual payload layout (short aliases for predictor/codec/length/segment fields) with backward-compatible decode aliases for previous key names.
- Post compact-key benchmark on the same subset/config yielded an incremental additional gain while preserving fidelity:
	- Wire container: `60461` bytes (`-52.77%` vs input), improved from `60530`.
	- HDF5 container: `78946` bytes (`-38.33%` vs input), improved from `79019`.
	- Program remained `latent-residual-v2`; exact decode and verify validity unchanged.
- Added optional packed program-payload parser/serializer path (`~mp85:` msgpack+zlib+base85) with fallback to compact JSON, while keeping backward compatibility for prior payload forms.
- Added symbol-coded identifiers for latent residual payload values (`predictor`, `codec`, `domain`) to further reduce envelope overhead.
- Post "max overhead" pass benchmark on the same subset/config improved again with fidelity preserved:
	- Wire container: `60443` bytes (`-52.78%` vs input), improved from `60461`.
	- HDF5 container: `78928` bytes (`-38.34%` vs input), improved from `78946`.
	- Program remained `latent-residual-v2`; exact decode and verify validity unchanged.

**Remaining non-binary parity gaps**
- Full Julia parity is still pending: current compact replay is scenario-constrained (`repeat-text-v1`, `logs-seq-v1`, `json-linear-items-v1`, `json-literal-v1`) rather than a general latent-manifold projection for arbitrary Text/JSON/Logs.
- Fidelity object parity is now partially implemented via emitted deterministic proxies (`fidelity_energy`, `fidelity_topology`, `fidelity_coherence`, `fidelity_bundle_v1`), but still lacks full Julia metric semantics.


**Alignment work required for Julia parity (next execution block)**
1. Generalize compact replay beyond fixed scenario templates to a latent-manifold projection path for arbitrary Text/JSON/Logs.
1. Move reconstructive commitment root from program-template transcript to compact latent/topology/dynamics commitments.
1. Keep exactness gate as deterministic replay + checksum match for `Text/JSON/Logs`, with explicit residual fallback policy when replay cannot satisfy exactness.
1. Maintain `legacy`/`topology` codecs unchanged; reconstructive compact path remains mode-scoped.

**Steps**
1. Phase A: Architecture freeze and acceptance contract
1. Define formal mode contract: `tokenize -> normalize -> manifold-fit -> K_M solve -> project -> bytes -> verify` is deterministic for fixed input/config/version.
1. Lock hard decode gates: fail on non-contraction (`L >= 1`), non-convergence, topology/coherence failure, or checksum mismatch.
1. Lock rollout scope: v1 supports only `Text/JSON/Logs`; binary action-token path is experimental.

2. Phase B: Tokenizer spec v1 (explicit defaults)
1. Implement `FrequencyTokenizerV1` with deterministic domain classes.
`Text`: grapheme/word/punctuation/whitespace with phoneme tags.
`JSON`: structural tokens, canonical key/value tokens, normalized numeric tokens.
`Logs`: timestamp/level/component/message-field tokens.
1. Use fixed frequency lattice defaults: `bin_count=128`, deterministic bin index in `[0..127]`, versioned bin-center table; serialize `bin_table_hash` and `vocab_hash`.
1. Enforce canonicalization before tokenization.
`Text`: UTF-8 + NFC + LF.
`JSON`: canonical key ordering + stable number formatting.
`Logs`: canonical timestamp normalization + field ordering.
1. Add schema metadata fields: `tokenizer_id`, `tokenizer_version`, `vocab_hash`, `bin_table_hash`, `normalization_profile_id`, `domain_kind`.

3. Phase C: Oscillator-normalization spec v1
1. Implement `OscillatorNormalizationV1` with deterministic transforms: fixed amplitude clamp, canonical phase normalization, bounded per-token scaling.
1. Serialize full normalization constants/hash into payload (no ambient defaults).
1. Add hard-fail policy for out-of-profile tokens.

4. Phase D: Generative hypergraph manifold pipeline
1. Build deterministic hypergraph from normalized frequency tokens: stable node ids, edge ordering, layer assignment, Morton-style indexing, deterministic tie-breaks.
1. Implement deterministic compact manifold fitting (no stochastic branches).
1. Persist only compact state + commitments in reconstructive mode (no full token sequence; optional debug-only output behind explicit flag).

5. Phase E: K_M fixed-point engine (Julia-aligned)
1. Add `src/braidcodec/algebra/fixedpoint.py`: `K_M(x)=kappa*x + eta*sin(x)`, convergence loop, residual stats, validation mask, iteration counts.
1. Bind deterministic initialization from normalized manifold state so solver trajectory is reproducible.
1. Emit and enforce diagnostics thresholds: `km_residual_max`, `km_residual_mean`, `km_iters_mean`, `km_valid_ratio`.

6. Phase F: Reconstructive schema + routing integration
1. Extend `schema.py` with reconstructive payload: `model/version`, tokenizer/normalization hashes, latent state, K_M params, commitments, metrics, checksum, timestamp.
1. Add `preprocessing_mode="reconstructive"` route in `encoder.py`.
1. Add reconstructive decode route in `decoder.py` with strict hard gates.
1. Keep `legacy` and `topology` unchanged for compatibility.

7. Phase G: Integrity and CLI surface
1. Extend `integrity.py` for reconstructive commitment verification (additive to checksum authority).
1. Extend `main.py` with reconstructive mode flags, domain selector, diagnostics verbosity.
1. Add explicit user-facing failure taxonomy for contraction/convergence/invariant/checksum failures.

8. Phase H: Validation + lightweight benchmarks
1. Determinism tests: repeated runs yield byte-identical payloads for same input/config/version.
1. Fidelity tests: exact checksum roundtrip + topology/coherence + K_M thresholds + decode idempotence.
1. Negative tests: tampered hashes/constants/latent state/commitments and injected non-convergence.
1. Lightweight benchmarks first: size ratio, throughput, K_M diagnostics for `Text/JSON/Logs` (no stress by default).

9. Phase I: Binary portability track (experimental)
1. Implement `src/braidcodec/codec/binary_action_tokenizer.py` with deterministic compiler/IR action extraction.
1. Map action classes to same frequency lattice + oscillator normalization contract.
1. Validate determinism across pinned toolchain/version matrix.
1. Keep binary reconstructive mode behind experimental flag until convergence/fidelity/compression parity is proven.

**Relevant files**
- `src/braidcodec/codec/encoder.py`
- `src/braidcodec/codec/decoder.py`
- `src/braidcodec/codec/schema.py`
- `src/braidcodec/codec/preprocessing.py`
- `src/braidcodec/codec/chunker.py`
- `src/braidcodec/codec/tokenizer_frequency.py` (new)
- `src/braidcodec/codec/oscillator_normalization.py` (new)
- `src/braidcodec/codec/binary_action_tokenizer.py` (new, experimental)
- `src/braidcodec/algebra/fixedpoint.py` (new)
- `src/braidcodec/algebra/tsr_constants.py`
- `src/braidcodec/crypto/integrity.py`
- `src/braidcodec/cli/main.py`
- `tests/codec/test_encoder.py`
- `tests/codec/test_decoder.py`
- `tests/codec/test_roundtrip.py`
- `tests/codec/test_schema.py`
- `tests/crypto/test_integrity.py`
- `tests/benchmarks/bench_core_throughput.py`
- `docs/architecture.md`
- `docs/api.md`
- `docs/benchmarks.md`
- `README.md`
- `CHANGELOG.md`

**Verification**
1. Determinism: identical input/config/version yields identical tokenizer output, latent state, diagnostics, and serialized payload bytes.
1. Solver correctness: enforce `L < 1`, bounded iterations, residual thresholds.
1. Fidelity: checksum-exact decode plus topology/coherence validation.
1. Robustness: tamper tests produce correct hard-fail categories.
1. Performance: lightweight benchmark matrix reports size ratio, throughput, and K_M diagnostics.
1. Compatibility: existing `legacy`/`topology` tests remain green.

**Decisions**
- Domain is `Text/JSON/Logs` for v1.
- Fidelity is bitwise lossless.
- Frequency tokenizer is required for reconstructive mode.
- Oscillator-derived normalization is required and versioned.
- K_M convergence + contraction are hard correctness gates.
- Residual is default quality telemetry (optional strict threshold mode), while checksum/invariants remain authoritative.
- Binary action-frequency transport is staged and experimental.

**Scope boundaries**
- Included: full reconstructive mode for `Text/JSON/Logs` with deterministic contracts.
- Excluded: default support for arbitrary binary in v1.
- Excluded: removal or regression of existing `legacy`/`topology` paths.

**Further considerations**
1. Float profile: start reconstructive solver in Float64 for stability, then evaluate mixed precision.
1. Bin table governance: freeze and version bin tables to avoid silent drift.
1. Toolchain pinning: binary action extraction requires a pinned compiler/IR version matrix for deterministic claims.