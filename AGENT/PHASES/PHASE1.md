

---

## Plan: Phase 1 — Chunker (Bijective Numeration)

**TL;DR**: Implement the pure-arithmetic bijection between byte blocks and braid generator sequences. This is the mathematical foundation the entire codec rests on. Zero algebra dependencies — just integer math and streaming I/O.

### Background

With `n` strands, valid generators are `{-(n-1), ..., -1, 1, ..., (n-1)}` — `b = 2(n-1)` values total. A block of `k` generators encodes values in `[0, b^k - 1]`, holding `floor(k * log₂(b) / 8)` bytes.

| n_strands | base | k=32 → bytes | k=16 → bytes |
|-----------|------|---------------|---------------|
| 2 | 2 | 4 | 2 |
| 3 | 4 | 8 | 4 |
| 4 | 6 | 10 | 5 |
| 5 | 8 | 12 | 6 |
| 6 | 10 | 13 | 6 |

**Digit ↔ Generator mapping** (n_strands=4, base=6 example):

| digit | 0 | 1 | 2 | 3 | 4 | 5 |
|-------|---|---|---|---|---|---|
| generator | +1 | +2 | +3 | -1 | -2 | -3 |

Encode: `d < (n-1)` → `+(d+1)`; `d >= (n-1)` → `-(d - n + 2)`
Decode: `g > 0` → `g - 1`; `g < 0` → `(n-1) + |g| - 1`

**Steps**

1. **Implement `src/braidcodec/codec/chunker.py`** (~120 lines, 5 functions):
   - `compute_block_size(n_strands, generators_per_block) -> int` — byte capacity per block
   - `compute_generators_needed(n_strands, block_bytes) -> int` — inverse of above
   - `bytes_to_generators(chunk, n_strands, generators_per_block) -> list[int]` — big-endian int → base-b decomposition → digit-to-generator mapping
   - `generators_to_bytes(generators, n_strands, original_length) -> bytes` — inverse mapping → base-b reconstruction → `int.to_bytes`
   - `chunk_stream(data: bytes | BinaryIO, block_size) -> Iterator[tuple[int, bytes, int]]` — yields `(block_index, padded_chunk, original_length)` with zero-padding on last block

2. **Write `tests/codec/test_chunker.py`** (~250 lines):
   - **Known vectors**: n=4/k=32 → block_size=10; n=3/k=16 → 4; n=2/k=8 → 1
   - **Encoding/decoding**: all-zeros→all `+1` generators; max-value chunk; "Hello" trace
   - **`chunk_stream`**: exact multiple, remainder/padding, empty input, single byte, `BytesIO` input
   - **Validation**: n_strands<2, generators_per_block<1, oversized chunks, generator=0, out-of-range generators → `ValueError`
   - **Hypothesis round-trips** (4 strategies):
     - `bytes → generators → bytes` for n ∈ [2,6], k ∈ [1,64], arbitrary chunks
     - `generators → bytes → generators` for valid generator sequences
     - All output generators satisfy `1 ≤ |g| < n_strands`
     - `chunk_stream` reassembly recovers original data for arbitrary inputs up to 10KB

**Relevant files**
- src/braidcodec/codec/chunker.py — new (~120 lines)
- `tests/codec/__init__.py` — new (empty)
- `tests/codec/test_chunker.py` — new (~250 lines)
- _types.py — `GeneratorSeq` already defined, used by chunker

**Verification**
1. `ruff check src/braidcodec/codec/ tests/codec/` — 0 violations
2. `mypy src/braidcodec/codec/chunker.py --strict` — 0 errors
3. `pytest tests/codec/test_chunker.py -v --cov=braidcodec.codec.chunker` — all pass, ≥95% coverage
4. Hypothesis round-trips: 1000+ examples per property, zero failures
5. `pytest tests/ -v` — regression: all algebra + chunker tests pass

**Decisions**
- `bytes_to_generators` takes explicit `generators_per_block` (not inferred) — the encoder owns this parameter, chunker is a pure function
- `chunk_stream` yields 3-tuples `(index, padded_chunk, original_length)` — the `original_length` is needed by the encoder to store on `EncodedBlock` for last-block padding removal
- No algebra imports — Phase 1 is pure `math` + `typing` + `collections.abc`
- Block size computed via `floor(k * log₂(base) / 8)` — conservative floor avoids overflow at the boundary