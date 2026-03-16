# Phase 10 Lean Focus: Compression + Perfect Fidelity

Date: 2026-03-16
Scope: Prioritize `lean` (`rt=ps2`) transport to maximize wire compaction while preserving exact decode and integrity validity.

## Objective

Make `lean` the primary high-compression reconstructive path with strict exactness.

Success criteria:
- Maximum practical wire-size reduction on benchmark anchors and mixed corpora.
- Exact decode remains 100% for accepted outputs.
- Verify path remains valid and deterministic.

## Work Items (Lean-First)

1. Residual payload dominance reduction
- Current largest byte contributor is residual fallback payloads.
- Implement mixed/segmented residual minimization so fallback stores less.
- Optimize by final serialized wire size, not intermediate payload size.

2. Discovery hit-rate increase
- Expand discoverable equation families and candidate recall.
- Improve candidate-library retrieval for semantically similar inputs.
- Reduce fallback frequency by promoting exact reusable programs.

3. Sidechannel payload compaction
- Minimize structural overhead of residual payload formats.
- Prefer short coded fields and packed representations where smaller.
- Keep deterministic parser compatibility for existing payloads.

4. Hard-corpus robustness (especially logs)
- Keep exact replay for non-template logs and mixed corpora.
- Avoid brittle template-only rejection when deterministic fallback is possible.

5. Lean integrity semantics finalization
- Keep checksum-authoritative exactness.
- Evaluate optional low-overhead semantic commitment add-ons only if size-neutral or size-positive.

6. Lean-first KPI gates
- Track and gate on wire bytes, fallback rate, residual sidechannel bytes, and exactness.
- Enable lean-default only after stability across benchmark matrix.

7. Fidelity stress expansion
- Add adversarial and mixed-structure corpora tests.
- Maintain strict requirement: decode exact and verify valid.

## Execution Order

1. Item 1 (residual minimization) and Item 3 (sidechannel compaction internals).
2. Item 2 (discovery hit-rate) and Item 4 (hard-corpus robustness).
3. Item 6 (benchmark gates) and Item 7 (stress tests).
4. Item 5 (integrity semantics hardening) as size-safe finalization.

## Non-Negotiables

- No fidelity regressions.
- No decode compatibility break for existing reconstructive payloads.
- Any new compact format must be reader-first compatible before writer defaults.
