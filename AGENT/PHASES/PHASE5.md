

## Plan: Phase 5 — Topological Compressor

**TL;DR**: Implement a 3-level topological compressor that shortens braid generator sequences while preserving decodability via **dual-generator storage** — compressed generators for invariant checks and wire format, original generators preserved for byte recovery. Requires a schema extension, minor decoder update, and the compressor module with Levels 0–2.

### Critical Design Constraint

`generators_to_bytes(gens, n_strands, original_length)` uses `len(gens)` to compute block capacity. Shortened generators produce wrong bytes. **Solution**: compressed blocks carry `decode_generators` (the original sequence for byte recovery) alongside `generators` (the shorter topologically-equivalent form for verification and wire format). The decoder dispatches on this.

All 3 levels preserve writhe (sum of signs unchanged), Jones polynomial (topological invariant), and matrix trace (same unitary).

---

**Steps**

### Phase 5A — Schema Extension (*blocks 5C, 5D; parallel with 5B*)

1. **Modify** schema.py — `EncodedBlock`:
   - Add field `decode_generators: list[int] | None = None`
   - Update `to_dict()` / `from_dict()` to serialize the new field (with `_opt_int_list` helper)
   - Add `__post_init__` validation: if present, all values non-zero with `|g| < n_strands`
   - Add property `effective_decode_generators -> list[int]` — returns `decode_generators ?? generators`
   - `_REQUIRED_KEYS` unchanged — field is optional, backward-compatible

### Phase 5B — Compression Algorithms (*pure braid math, independent of schema*)

2. **Create** `src/braidcodec/codec/compressor.py` (~250 lines):

   **Level 0 — Inverse cancellation** (`_compress_level0`):
   Wraps existing `simplify_braid()`. Cancels adjacent pairs $\sigma_i\sigma_i^{-1} \to e$.

   **Level 1 — Far commutativity + re-cancel** (`_compress_level1`):
   For each cancellable pair $(p, q)$ with `gens[p] == -gens[q]`:
   - Check if all intervening generators commute with `gens[p]` ($|i_m - i_p| \geq 2\ \forall\ p < m < q$)
   - If yes → remove both (justified by bubble-through chain of valid commutations)
   - Repeat until stable. Then re-run Level 0.
   - *Example*: `[1, 3, -1, 2]` on 5 strands → gen 3 commutes with gen 1 ($|3-1|=2$) → cancel 1,-1 → `[3, 2]`

   **Level 2 — Yang-Baxter BFS** (`_compress_level2`):
   - Detect YBE patterns: $(i, i{+}1, i) \leftrightarrow (i{+}1, i, i{+}1)$ and all-negative variant
   - Bounded BFS (`max_rewrites=100`): apply YBE rewrite → Level 0+1 → track shortest
   - *Example*: `[1, 2, 1, -2]` → YBE yields `[2, 1, 2, -2]` → cancel $(2,-2)$ → `[2, 1]`
   - Verify: `verify_yang_baxter_morphism(original, best)` must pass

   **Public API:**
   - `compress_braid(braid, *, level=1) -> BraidEquation` — low-level
   - `compress(stream, key, *, level=1, max_rewrites=100) -> EncodedStream` — stream-level, sets `decode_generators` on compressed blocks
   - `compression_ratio(original, compressed) -> float`

### Phase 5C — Decoder Update (*depends on 5A, ~5 lines*)

3. **Modify** decoder.py:
   - Change `generators_to_bytes(block.generators, ...)` → `generators_to_bytes(block.effective_decode_generators, ...)`
   - Writhe/invariant checks still use `block.generators` (stored values match compressed form)

### Phase 5D — Public API (*depends on 5A, 5B*)

4. **Update** __init__.py — add `compress` re-export

### Phase 5E — Tests (~350 new lines + ~35 lines in existing tests)

5. **Create** `tests/codec/test_compressor.py` (~350 lines, ~22 tests):

   | Test class | Key tests |
   |---|---|
   | **TestLevel0** | inverse cancel, nested cancel, no-op, matrix preserved |
   | **TestLevel1** | distant commute cancel `[1,3,-1,2]→[3,2]`, no-commute case `[1,2,-1]` stays (|2-1|=1<2), multiple distant pairs, Jones preserved |
   | **TestLevel2** | YBE enables cancel `[1,2,1,-2]→[2,1]`, all-negative YBE, bounded search, matrix preserved |
   | **TestCompressStream** | encode→compress→verify decode_generators present, compressed stream decodes correctly, ratio ≤ 1.0, progressive levels |
   | **TestCompressAllSectors** | parametrize 5 sectors |
   | **TestCompressHypothesis** | Jones preserved for random braids, matrix preserved, encode→compress→decode==original |

6. **Update** test_decoder.py — 1 test: decode block with `decode_generators` set
7. **Update** test_schema.py — 2 tests: field round-trips, wire format with compressed blocks

---

**Relevant files**

| File | Action | ~Lines |
|------|--------|--------|
| src/braidcodec/codec/compressor.py | **new** | 250 |
| schema.py | **modify** | +25 |
| decoder.py | **modify** | ~5 |
| __init__.py | **modify** | +2 |
| tests/codec/test_compressor.py | **new** | 350 |
| test_decoder.py | **modify** | +15 |
| test_schema.py | **modify** | +20 |

Reference (unchanged): braid_equations.py (`simplify_braid`, `BraidEquation`, `jones_polynomial`, `writhe`, `contract_braid_tensor`), yang_baxter.py (`verify_yang_baxter_morphism`), encoder.py (`encode`), keys.py (`BraidKey`), _exceptions.py (`CompressionError`, `RewriteVerificationError` already exist)

---

**Verification**

1. `pytest tests/algebra/ -v` — 0 regressions (algebra frozen)
2. `pytest test_schema.py -v` — schema round-trips with `decode_generators`
3. `pytest test_decoder.py -v` — decoder handles compressed blocks
4. `ruff check src/braidcodec/codec/compressor.py` + `mypy --strict` — 0 issues
5. `pytest tests/codec/test_compressor.py -v --cov=braidcodec.codec.compressor` — all pass, ≥95%
6. `pytest tests/ -v --cov=braidcodec --cov-fail-under=95` — full regression green

---

**Decisions**

- **Dual-generator storage** for compressed blocks (user choice) — `decode_generators` preserves original sequence for byte recovery
- **Schema backward-compatible** — optional field, no wire format version bump
- **Level 1 "distant cancellation"** — O(k³) scan for cancellable pairs separated by commuting generators; more targeted than full DAG topological sort
- **Level 2 bounded BFS** — 100 states max; apply YBE → Level 0+1 → measure. Matrix-verified result
- **YBE patterns: all-positive and all-negative triples only** — mixed-sign deferred
- **Level 3 (Markov stabilization) excluded** — per ROADMAP, deferred to v0.2.0
- **Invariants recomputed from compressed braid** — defense-in-depth against algorithm bugs