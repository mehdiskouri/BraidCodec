## Plan: Compaction Parity Closure (Julia-Focused)

Maximize practical compaction performance for reconstructive mode while preserving exact decode guarantees and backward compatibility. This phase addresses remaining parity gaps:
1) braid-only hot path still larger than sidechannel,
2) residual dependence on hard corpora,
3) commitment rooted in transport payloads rather than a minimal canonical object,
4) metadata/envelope overhead still dominant for small-medium payloads.

This is a compaction-first phase. Fidelity remains non-negotiable: deterministic replay + checksum authority + strict integrity validation.

**Goals**
- Make embedded braid transport (`pg`) size-competitive with sidechannel modes.
- Reduce residual fallback frequency and bytes when fallback is required.
- Introduce canonical compact commitment object with versioned migration.
- Split hot-path compact header from optional audit metadata.

**Non-goals**
- Changing existing `legacy` / `topology` behavior.
- Relaxing exact decode requirements.
- Removing backward parser support for existing compact aliases.

## Workstreams

### Workstream A: Braid-Only Hot Path Competitiveness (Gap 1)

**Objective**
Reduce embedded transport overhead so braid-only program words can win for practical ranges, not only as a conceptual endpoint.

**Steps**
1. Add transport-only block framing class.
- Introduce a compact block class/flag that omits invariant payload fields for transport-only embedded program blocks.
- Keep existing block format readable; only new writers emit compact transport-only block framing.

2. Define `braid-program-v2` compact binary grammar.
- Canonical fields: `family_code`, `rollout_len`, `coefficients`, `initial_state`, `profile_code`.
- Use short numeric enums, fixed/varint lengths, and deterministic ordering.
- No string keys in embedded hot path.

3. Keep dual parser compatibility.
- Decoder parser accepts legacy `pg1` JSON/msgpack forms and new `braid-program-v2` compact payloads.
- Emit new form by default behind version flag.

4. Add adaptive crossover selection.
- At encode time, estimate wire bytes for `ps2` vs `pg` and select minimum.
- Persist mode selection telemetry for benchmark scorecard.

**Acceptance gates**
- `pg` wire bytes <= `ps2` for at least one defined medium-size benchmark tier.
- No decode/integrity regressions in compact transport tests.

---

### Workstream B: Residual Dependence Reduction (Gap 2)

**Objective**
Increase discoverable-equation coverage and reduce fallback byte share.

**Steps**
1. Expand deterministic equation family catalog.
- Add low-cost families before residual fallback.
- Keep exactness pre-check required for promotion.

2. Mixed-program segmentation.
- Permit stream-level composition: discoverable segments + residual correction segments.
- Encode only irreducible residual fragments, not whole payload fallback.

3. Candidate library recall improvements.
- Enrich semantic signature keys and nearest-compatible retrieval path.
- Maintain strict exact replay check before hit acceptance.

4. Residual minimization policy.
- Choose predictor/codec by final wire estimate, not payload-only size.
- Prefer short residual envelopes and segment-level adaptation.

**Acceptance gates**
- Fallback frequency decreases on structured benchmark suites.
- Average residual sidechannel bytes decrease on mixed corpora.
- Exact decode remains 100% for accepted outputs.

---

### Workstream C: Canonical Commitment Object (Gap 3)

**Objective**
Shift commitment semantics to a minimal canonical compact object independent of verbose transport representation.

**Steps**
1. Define canonical commitment object `C_v3`.
- Required fields: `schema_version`, `transport_family`, `program_digest`, `profile_digest`, optional `embedded_digest`.
- Canonical byte serialization (stable field order + encoding).

2. Add `reconstructive_commitment_v3` emission/validation.
- New streams emit v3 commitment for compact paths.
- Validators accept legacy commitment modes (`reconstructive_commitment`, `rc`) for backward compatibility.

3. Bind integrity to canonical object first.
- Verify `C_v3` digest before route-specific decode checks.
- Keep checksum as final byte authority.

4. Migration compatibility matrix.
- Writer: prefer v3.
- Reader: accept v1/v2/v3.
- Inspector output: show commitment version + canonical fields.

**Acceptance gates**
- Sidechannel/transport representation mutations that preserve semantics still fail commitment if canonical object changes.
- Legacy streams remain decodable/verifiable unchanged.

---

### Workstream D: Hot-Path Metadata Split (Gap 4)

**Objective**
Move from metadata-heavy contract to a minimal compact header with optional audit sidecar.

**Steps**
1. Introduce `compact_header_v3` binary map.
- Include only decode-critical fields: transport selector, compact program reference/payload handle, commitment reference, checksum, minimal profile ids.
- Omit verbose diagnostics from default hot path.

2. Add optional `audit_bundle_v1` sidecar.
- Move large diagnostics (`fidelity_*`, solver internals, extended hashes) behind explicit flag.
- Ensure tooling can load audit bundle when present.

3. Dictionary-code repeated identifiers.
- Short enum/profile codes for tokenizer/domain/profile constants.
- Maintain deterministic codebook versioning.

4. Adaptive writer policy.
- If sidecar disabled, emit minimal header only.
- If sidecar requested, keep compact header unchanged and append audit bundle.

**Acceptance gates**
- Compact enabled/lean metadata keyset reduced to decode-critical minimum.
- Wire-size improves on small-medium benchmark payloads vs pre-v3 baseline.

## Cross-Cutting Migration Strategy

1. Reader-first migration.
- Implement parser/validator support for new forms before enabling writer defaults.

2. Compatibility windows.
- Preserve support for:
  - `rp1` and `reconstructive_payload_v1`
  - `rpb` and legacy payload aliases
  - `rt` folded/legacy type handling
  - existing commitment keys

3. Feature flags.
- Gate new writer modes with explicit config until benchmark/fidelity gates pass.

4. Rollout order.
- Workstream D (metadata split) -> A (embedded competitiveness) -> B (fallback reduction) -> C (commitment v3 finalization).

## Files Expected To Change

- `src/braidcodec/codec/encoder.py`
- `src/braidcodec/codec/decoder.py`
- `src/braidcodec/codec/schema.py`
- `src/braidcodec/codec/reconstructive_compact.py`
- `src/braidcodec/codec/equation_discovery.py`
- `src/braidcodec/codec/equation_library.py`
- `src/braidcodec/crypto/integrity.py`
- `src/braidcodec/cli/main.py`
- `tests/codec/test_encoder.py`
- `tests/codec/test_decoder.py`
- `tests/codec/test_schema.py`
- `tests/codec/test_reconstructive_equation_discovery.py`
- `tests/crypto/test_integrity.py`
- `tests/benchmarks/bench_reconstructive_path.py`
- `docs/architecture.md`
- `docs/benchmarks.md`

## Verification

1. Targeted compact-path correctness:
- `PYTHONPATH=src python -m pytest -q tests/codec/test_encoder.py tests/codec/test_decoder.py tests/codec/test_schema.py tests/crypto/test_integrity.py tests/cli/test_cli.py -k 'reconstructive or compact_transport or compact'`

2. Discovery/fallback correctness:
- `PYTHONPATH=src python -m pytest -q tests/codec/test_reconstructive_equation_discovery.py tests/codec/test_reconstructive_generalization.py`

3. End-to-end reconstructive regression:
- `PYTHONPATH=src python -m pytest -q -k reconstructive`

4. Benchmark scorecard:
- `PYTHONPATH=src python -m pytest -q tests/benchmarks/bench_reconstructive_path.py --benchmark-enable -o 'python_files=bench_*.py'`

## Compaction KPI Scorecard (Phase 10 Exit)

- `lean` wire bytes improved or maintained on current benchmark anchor cases.
- `pg` beats or ties `ps2` on at least one designated medium-size class.
- Residual fallback rate reduced on structured corpora.
- Residual payload bytes reduced for fallback cases.
- Commitment v3 path active for new compact streams with legacy read compatibility.
- No regressions in exact decode or integrity validity.

## Risks And Mitigations

1. Risk: New compact header (`rh`/v3) can regress wire size on some payloads.
- Mitigation: adaptive size-based selection and benchmark-gated defaults.

2. Risk: Parser complexity increases compatibility bugs.
- Mitigation: reader-first rollout, exhaustive compatibility tests, explicit versioned parse matrix.

3. Risk: Embedded-only focus may delay practical wins.
- Mitigation: prioritize metadata split and adaptive mode selection first.

4. Risk: Over-optimization harms debuggability.
- Mitigation: keep optional `audit_bundle_v1` sidecar and inspector support.

## Decisions

- Fidelity remains checksum-authoritative and exact.
- Compatibility remains mandatory; no destructive schema cutover.
- Compaction wins are measured on serialized wire bytes, not payload-only proxy sizes.
- Julia parity target interpreted as practical near-O(1) for structured families plus graceful fallback minimization.
