

---

## Plan: Encoder + Keys (Phase 3)

**TL;DR**: Implement `BraidKey` (generation + serialization) and `encode()` (full pipeline from bytes → chunking → braid construction → invariant computation → `EncodedStream`). Requires a backward-compatible algebra extension to inject keyed R-matrices via `_sector_params`. ProcessPoolExecutor for parallel block encoding.

**Steps**

### Phase 3A — Algebra Extension (*blocks all other steps*)

1. **Extend `BraidEquation` with `_sector_params`** in braid_equations.py:
   - Add `"_sector_params"` to `__slots__` (line 53)
   - Add `_sector_params: dict[str, float] | None = None` keyword param to `__init__`, store on `self`
   - Add `theta_override: float | None = None` keyword param to `get_sector_r_matrix()` (line 123) — when set, use it instead of sector-derived theta. Existing calls unchanged.
   - Thread `theta_override` through `get_braid_generator_matrix()` (line 180)
   - In `contract_braid_tensor()` (line 219): extract `theta_ov = braid._sector_params.get("theta") if braid._sector_params else None` and pass to `get_braid_generator_matrix`
   - In `kauffman_bracket()` (line 408): when `braid._sector_params` has `"theta"`, compute `A = exp(i·θ_eff/4)` instead of the sector-specific formula
   - ~20 lines of changes total, fully backward-compatible — 459 algebra tests unchanged
   - **Verify**: `pytest tests/algebra/ -v` — all 459 green

### Phase 3B — BraidKey + keygen (*parallel with 3A if no import needed*)

2. **Create** `src/braidcodec/crypto/keys.py` (~200 lines):
   - `BraidKey` — `@dataclass(frozen=True, slots=True)`: `sector` (str), `n_strands` (int), `theta_offset` (float in [0, 2π)), `key_id` (str, 32 hex chars). `__post_init__` validates sector/n_strands/range. Properties: `.theta_effective`, `.sector_params` → `{"theta": theta_effective}`
   - `keygen(sector="TSR", n_strands=4, theta_offset=None) -> BraidKey` — if offset is None, generate from `secrets.token_bytes(32)` → `struct.unpack('>d', raw[:8])[0] % (2π)` with NaN/inf guard. Compute `key_id = blake3(sector + n_strands + theta_offset).hexdigest()[:32]`. Validate keyed R-matrix unitarity before returning.
   - `_base_sector_theta(sector)` — returns base theta per sector (TSR→π·C, Ising→π/4, Fibonacci→4π/5, SU2k2→π/2, Identity→π·C)
   - `_SECTOR_TO_ENUM` / `_ENUM_TO_SECTOR` mappings for serialization

3. **Key serialization** in same file:
   - `key_to_bytes(key) -> bytes` — 50-byte binary: `b"BRDK"` (4) + version (1) + sector_enum (1) + n_strands uint32 (4) + theta_offset float64 (8) + BLAKE3 of preceding 18 bytes (32)
   - `key_from_bytes(data) -> BraidKey` — verify magic + length + digest, parse, reconstruct via `keygen()` for re-validation. Raises `FormatError`/`KeyValidationError`

4. **Update** __init__.py — re-export `BraidKey`, `keygen`, `key_to_bytes`, `key_from_bytes`

### Phase 3C — Encoder (*depends on 3A + 3B*)

5. **Create** `src/braidcodec/codec/encoder.py` (~200 lines):
   - `encode(data: bytes, key: BraidKey, *, generators_per_block: int = 32, max_workers: int | None = None) -> EncodedStream`
     1. `block_size = compute_block_size(key.n_strands, generators_per_block)`
     2. `checksum = blake3.blake3(data).digest()`
     3. `chunks = list(chunk_stream(data, block_size))`
     4. Encode blocks (parallel or serial via `_encode_blocks()`)
     5. Return `EncodedStream(...)`
   - `_encode_block(chunk, n_strands, sector, sector_params, generators_per_block, block_index, original_length) -> EncodedBlock` — module-level (picklable):
     1. `generators = bytes_to_generators(chunk, n_strands, generators_per_block)`
     2. `braid = BraidEquation(n_strands, generators, sector=sector, _sector_params=sector_params)`
     3. `matrix = contract_braid_tensor(braid)`
     4. `w = writhe(braid)`
     5. Tier: `2 if len(generators) <= 24 else 3`
     6. Tier 2: `j = jones_polynomial(braid)` → `jones_real/imag`
     7. Tier 3: `tr = complex(np.trace(matrix))` → `trace_real/imag`
     8. `simplified = simplify_braid(braid)`
     9. Return `EncodedBlock(generators=simplified.generators, ...)`
   - `ProcessPoolExecutor` with `min(max_workers or cpu_count()-1, 8, len(chunks))` workers. Serial fallback for ≤1 block.

### Phase 3D — Tests (*depends on 3A, 3B, 3C*)

6. **Create** `tests/crypto/__init__.py` (empty)

7. **Create** `tests/crypto/test_keys.py` (~250 lines, ~17 tests):
   - Key construction: default keygen, explicit offset, all 5 sectors, deterministic key_id, unique key_ids
   - Validation: bad sector, bad n_strands, offset out of range, unitarity check
   - Serialization: 50-byte length, magic, round-trip, bad magic, corrupted digest, wrong length
   - Hypothesis: arbitrary offset round-trip, keyed R-matrix unitarity for random keys

8. **Create** `tests/codec/test_encoder.py` (~350 lines, ~15 tests):
   - Basic: encode "Hello", stream field verification, simplified generators, writhe correctness, jones correctness (tier 2), trace correctness (tier 3)
   - Sectors: all 5 sectors produce valid streams, different keys → different invariants
   - Tiers: k≤24 → tier 2, k>24 → tier 3
   - Multi-block: 100 bytes → sequential block indices, empty data
   - Parallelism: serial=parallel (excluding timestamp), max_workers=1 works
   - Checksum: stream.checksum == blake3(data).digest()

**Relevant files**
- braid_equations.py — **modify** ~20 lines: `_sector_params` slot, `theta_override` in R-matrix/contraction/jones
- `src/braidcodec/crypto/keys.py` — **new**, ~200 lines
- __init__.py — **update**: re-exports
- `src/braidcodec/codec/encoder.py` — **new**, ~200 lines
- `tests/crypto/__init__.py` — **new** (empty)
- `tests/crypto/test_keys.py` — **new**, ~250 lines
- `tests/codec/test_encoder.py` — **new**, ~350 lines

**Verification**
1. `pytest tests/algebra/ -v` — all 459 green (algebra backward-compat regression)
2. `ruff check` + `ruff format --check` on all new/modified files — 0 issues
3. `mypy --strict` on `crypto/keys.py` and `codec/encoder.py` — 0 errors
4. `pytest tests/crypto/test_keys.py -v --cov=braidcodec.crypto.keys` — all pass, ≥95%
5. `pytest tests/codec/test_encoder.py -v --cov=braidcodec.codec.encoder` — all pass, ≥95%
6. `pytest tests/ -v` — full regression (algebra + chunker + schema + keys + encoder)

**Decisions**
- **Architecture-prescribed algebra extension** (Option A) for keyed R-matrix — backward-compatible `_sector_params` slot + `theta_override` params. Avoids duplicating contraction/jones logic in encoder, decoder, verifier, compressor.
- **Invariants computed before simplification** — simplification preserves topology but compute-first-then-simplify avoids edge cases
- **Tier threshold**: `len(generators) > 24` → tier 3 (trace); `≤ 24` → tier 2 (Jones)
- **Fresh ProcessPoolExecutor per encode()** — simple, testable, clean resource lifecycle
- **Workers capped at 8** — memory safety for multiple matrix computations
- **`_encode_block` is module-level** — required for `pickle` (ProcessPoolExecutor)
- **NaN/inf guard in keygen** — `struct.unpack('>d', random_bytes)` can produce special values; regenerate

**Further Considerations**
1. **Kauffman A with keyed theta**: The general formula is `A = exp(i·θ_eff/4)`. Verify this reproduces the base sector A values (it should: `exp(i·π·C/4)` for TSR matches `get_kauffman_A("TSR")`). If there's a mismatch, fall back to a separate `_keyed_kauffman_A(theta_eff)` helper.
2. **Algebra "frozen" policy**: The `_sector_params` addition is backward-compatible (no existing behavior changes), but it IS a runtime change to the algebra layer. If strict "zero changes" is preferred, Option B (standalone keyed functions in encoder, ~50 lines duplicated) is available.