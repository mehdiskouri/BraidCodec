## Plan: Phase 4 — Decoder, Integrity Verification & Round-trip

**TL;DR**: Implement `decode()` (wire format → bytes with inline integrity checks), `crypto/integrity.py` (5-channel `verify()` + `VerificationResult`), fix the 2 `TestPauliCompat` test failures to expect qiskit behavior, and prove end-to-end correctness with Hypothesis round-trip tests. Completes the codec loop: `decode(encode(data, key), key) == data`.

---

**Steps**

### Phase 4A — Fix TestPauliCompat failures (*independent, no blockers*)

1. **Modify** test_fermion_bounds.py — `TestPauliCompat` class (~6 lines):
   - `test_invalid_label`: qiskit raises `QiskitError` not `ValueError` for `Pauli("ABC")`. Update to `pytest.raises(QiskitError)` with import from `qiskit.exceptions`.
   - `test_eq_non_pauli`: qiskit's `__eq__` returns `False` for non-Pauli (not `NotImplemented`). Update to `assert Pauli("X") != "X"` — tests the behavioral contract.

### Phase 4B — Decoder module (*no new dependencies*)

2. **Create** `src/braidcodec/codec/decoder.py` (~150 lines):

   **`decode(stream, key, *, verify=True) -> bytes`** — pipeline per Architecture §4:
   1. Key validation → `KeyMismatchError`  
   2. Sort blocks by `block_index` (defensive)  
   3. Per-block: writhe pre-check (always) → invariant check (if `verify=True`, branch on tier 2/3) → `generators_to_bytes()`  
   4. Concatenate chunks  
   5. BLAKE3 checksum → `ChecksumError`  

   - Writhe always checked, even `verify=False` (O(k), free)
   - No parallelism — decode arithmetic is cheap

### Phase 4C — Integrity module (*parallel with 4B*)

3. **Create** `src/braidcodec/crypto/integrity.py` (~180 lines):

   **`VerificationResult`** — `@dataclass(frozen=True, slots=True)`: `valid`, `structural_passed`, `writhe_passed`, `invariant_passed`, `fermion_passed` (None if not run), `checksum_passed`, `failed_blocks`, `details`

   **`verify(stream, key, *, fermion_check=False) -> VerificationResult`** — 5 channels per Architecture §6:
   1. **Structural** — `validate_braid_category()` per block (short-circuit)
   2. **Writhe** — recompute, compare
   3. **Invariant** — Jones (tier 2) / trace (tier 3) re-computation
   4. **Fermion** (optional) — map σᵢ→`occupy(i)`, σᵢ⁻¹→`vacate(i)`, catch Pauli exclusion violations
   5. **Checksum** — decode all blocks → BLAKE3 vs `stream.checksum`

### Phase 4D — Public API updates (*depends on 4B + 4C*)

4. **Update** __init__.py — add `decode`, `verify`, `VerificationResult` re-exports

### Phase 4E — Tests (*depends on 4A–4D*)

5. **Create** `tests/codec/test_decoder.py` (~300 lines, ~18 tests):
   - Basic round-trips (hello, empty, single byte, exact block boundary)
   - Key mismatch (sector, strands) → `KeyMismatchError`
   - Tampering: flipped generator → `WritheError`/`JonesError`; modified writhe → `WritheError`; modified jones → `JonesError`; modified trace → `TraceError`; modified checksum → `ChecksumError`
   - `verify=False` skips invariant check (tampered jones still decodes)
   - Multi-block, all 5 sectors

6. **Create** `tests/crypto/test_integrity.py` (~250 lines, ~15 tests):
   - Clean stream → all channels pass
   - Fermion check disabled by default (None), enabled passes for clean data
   - Per-channel corruption detection (structural, writhe, jones, trace, checksum, fermion)
   - Short-circuit: structural failure skips downstream
   - Wrong key → invariant fails

7. **Create** `tests/codec/test_roundtrip.py` (~200 lines, ~10 tests):
   - Parametrized sizes: 0B, 1B, 32B, 1KB, 10KB
   - All 5 sectors × 2 sizes
   - **Hypothesis**: `@given(st.binary(max_size=1024))` with `verify=False` (speed); `@given(st.binary(max_size=256))` with `verify=True`, k=8 (tier 2)
   - Full wire-format round-trip: encode → to_bytes → from_bytes → decode

---

**Relevant files**

| File | Action | ~Lines |
|------|--------|--------|
| src/braidcodec/codec/decoder.py | **new** | 150 |
| src/braidcodec/crypto/integrity.py | **new** | 180 |
| __init__.py | **modify** | +5 |
| test_fermion_bounds.py | **modify** | ~6 |
| tests/codec/test_decoder.py | **new** | 300 |
| tests/crypto/test_integrity.py | **new** | 250 |
| tests/codec/test_roundtrip.py | **new** | 200 |

Reference (unchanged): chunker.py (`generators_to_bytes`), schema.py (`EncodedBlock`/`EncodedStream`), encoder.py (`encode`), keys.py (`BraidKey`), braid_equations.py, yang_baxter.py, fermion_bounds.py, _exceptions.py (all exception classes already exist), _types.py (`JONES_TOLERANCE`, `MATRIX_TOLERANCE`)

**Verification**

1. `pytest test_fermion_bounds.py -v` — 0 failures (Pauli fix)
2. `ruff check src/braidcodec/codec/decoder.py src/braidcodec/crypto/integrity.py` — 0 violations
3. `mypy src/braidcodec/codec/decoder.py src/braidcodec/crypto/integrity.py --strict` — 0 errors
4. `pytest tests/codec/test_decoder.py -v --cov=braidcodec.codec.decoder` — all pass, ≥95%
5. `pytest tests/crypto/test_integrity.py -v --cov=braidcodec.crypto.integrity` — all pass, ≥95%
6. `pytest tests/codec/test_roundtrip.py -v` — all pass, Hypothesis 200+ examples
7. `pytest tests/ -v --cov=braidcodec --cov-fail-under=95` — full regression green

**Decisions**

- `verify=True` default for `decode()` — safe-by-default
- Writhe always checked even with `verify=False` — O(k), free safety net
- No decode parallelism — arithmetic is cheap, only integrity checks are expensive
- `integrity.py` included in Phase 4 (not deferred)
- Pauli tests updated for qiskit behavior (not skipped)
- Tampering tests construct new frozen dataclass instances with modified fields
- Hypothesis round-trips use `verify=False` for large inputs to avoid 2^k state-sum; targeted tests cover `verify=True` with small k
- Fermion channel: σᵢ → `occupy(i)`, σᵢ⁻¹ → `vacate(i)` per Architecture §6