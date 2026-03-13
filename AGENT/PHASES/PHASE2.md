
Compacted conversationGood, the plan is saved. Here's the Phase 2 plan:

---

## Plan: Schema Wire Format + Exceptions

Build `EncodedBlock`/`EncodedStream` dataclasses with msgpack binary serialization and the full exception hierarchy. Two new source files, one test file.

**Steps**

### Phase 2A — Exception Hierarchy (no dependencies on other steps)

1. Create src/braidcodec/_exceptions.py (~80 lines, 14 exception classes):

   ```
   BraidCodecError(Exception)                  # base; stores **context dict
   ├── FormatError                             # wire format parsing
   │   ├── MagicMismatchError                  # magic != b"BRDC"
   │   ├── VersionError                        # unsupported version
   │   └── DigestError                         # trailing BLAKE3 mismatch
   ├── KeyError_(BraidCodecError)              # underscore avoids shadowing builtin
   │   ├── KeyMismatchError
   │   └── KeyValidationError
   ├── IntegrityError
   │   ├── WritheError
   │   ├── JonesError
   │   ├── TraceError
   │   ├── FermionError
   │   └── ChecksumError
   ├── EncodingError
   │   ├── ChunkError
   │   └── ContractionError
   └── CompressionError
       └── RewriteVerificationError
   ```

   `BraidCodecError.__init__(self, message: str, **context: object)` stores `self.context = context` for structured diagnostics. Leaf classes are simple `pass` bodies.

### Phase 2B — `EncodedBlock` dataclass (*parallel with 2A*)

2. Create src/braidcodec/codec/schema.py — start with `EncodedBlock`:

   `@dataclass(frozen=True, slots=True)` with 11 fields: `generators` (`list[int]`), `n_strands` (`int`), `sector` (`str`), `writhe` (`int`), `block_index` (`int`), `original_length` (`int`), `invariant_tier` (`int`), `jones_real`/`jones_imag` (`float | None`), `trace_real`/`trace_imag` (`float | None`).

   - Properties: `.jones -> complex | None`, `.trace_invariant -> complex | None`
   - `__post_init__` enforces: tier 1 → jones & trace must be None; tier 2 → jones required, trace None; tier 3 → trace required
   - `to_dict() -> dict[str, object]` / `@classmethod from_dict(cls, d) -> EncodedBlock` (validates keys/types, raises `FormatError`)

### Phase 2C — `EncodedStream` + Wire Format (*depends on 2A + 2B*)

3. Add `EncodedStream` to schema.py — `@dataclass(frozen=True, slots=True)` with 8 fields: `version` (`int`, default 1), `blocks` (`tuple[EncodedBlock, ...]`), `n_strands`, `sector`, `total_bytes`, `checksum` (`bytes`, 32B), `timestamp` (`int`, nanoseconds), `metadata` (`dict[str, str]`).

4. Implement `to_bytes() -> bytes`:
   - `b"BRDC"` (4B) + `uint16 version` (2B big-endian) + `uint32 header_length` (4B) + msgpack header + `uint32 block_count` (4B) + per-block length-prefixed msgpack payloads + trailing 32-byte BLAKE3 digest
   - Uses `bytearray` accumulator, single `bytes()` at end

5. Implement `@classmethod from_bytes(data) -> EncodedStream`:
   - **Verify trailing BLAKE3 digest BEFORE unpacking msgpack** — defense-in-depth against attacker-controlled payloads
   - Check minimum 46 bytes, magic == `b"BRDC"`, version ≤ 1
   - Parse header via `msgpack.unpackb`, iterate length-prefixed blocks via `EncodedBlock.from_dict`
   - Raises: `MagicMismatchError`, `VersionError`, `DigestError`, `FormatError`

### Phase 2D — Module Wiring (*depends on 2C*)

6. Update __init__.py — re-export `EncodedBlock`, `EncodedStream`

### Phase 2E — Tests (*depends on 2C*)

7. Create `tests/codec/test_schema.py` (~400 lines, ~32 tests):

   - **Exception tests** (~10): inheritance chains, `context` dict storage, `IntegrityError` subtypes
   - **EncodedBlock tests** (~12): construction per tier (1/2/3), `.jones`/`.trace_invariant` properties, `to_dict`/`from_dict` round-trip, missing field → `FormatError`, bad type → `FormatError`, invariant tier validation, frozen enforcement, empty generators
   - **Stream round-trip tests** (~10): single block, multiple blocks with mixed tiers, empty blocks, large metadata, binary checksum preservation, version field
   - **Corruption tests** (~8): bad magic → `MagicMismatchError`, version=99 → `VersionError`, truncated header, truncated blocks, corrupted digest (flip bit in last 32B), corrupted header content, corrupted block content, data < 46 bytes
   - **Hypothesis property tests** (~3): composite strategy generates valid `EncodedBlock`/`EncodedStream` with correlated invariant_tier↔jones/trace fields; arbitrary round-trip; random bit-flip always detected

**Relevant files**
- src/braidcodec/_exceptions.py — **new**, ~80 lines (14 exception classes)
- src/braidcodec/codec/schema.py — **new**, ~250-300 lines (2 dataclasses + wire format)
- __init__.py — **update**: re-export `EncodedBlock`, `EncodedStream`
- tests/codec/test_schema.py — **new**, ~400 lines (~32 tests)
- Dependencies used: `msgpack` (packb/unpackb), `blake3`, `struct` (pack/unpack)

**Verification**
1. `ruff check` + `ruff format --check` on all new/modified files — 0 issues
2. `mypy --strict` on `_exceptions.py` and `codec/schema.py` — 0 errors
3. `pytest tests/codec/test_schema.py -v --cov=braidcodec.codec.schema --cov=braidcodec._exceptions --cov-report=term-missing` — all pass, ≥95% on schema, 100% on exceptions
4. Hypothesis: 1000+ examples for round-trip + corruption detection
5. `pytest tests/ -v` — full regression (algebra + chunker + schema)

**Decisions**
- Exceptions in `_exceptions.py` (not schema) — used by encoder, decoder, compressor, integrity, CLI across phases
- `frozen=True, slots=True` — immutability + memory efficiency; `tuple[EncodedBlock, ...]` for blocks (not list)
- Jones stored as `(real, imag)` pair — msgpack has no native complex type
- Digest verified before msgpack deserialization — defense-in-depth
- `KeyError_` trailing underscore — avoids shadowing Python builtin `KeyError`
- `__post_init__` enforces invariant_tier consistency — fail-fast on construction