

## Plan: CLI & Public API (Phase 6)

Implement a click-based CLI (`braidcodec`) with 6 commands, proper exit codes (0-4), verbose/quiet flags, and full CliRunner test coverage. Public API is already complete — no changes needed. Wire format bridge: `EncodedStream.to_bytes()` / `EncodedStream.from_bytes(data)`.

---

### Phase A: Scaffold & Shared Infrastructure

**Step 1** — Create src/braidcodec/cli/main.py scaffold
- Top-level `@click.group()` named `cli` with `--verbose`/`--quiet` (mutually exclusive)
- Exit code constants: OK=0, INTEGRITY=1, KEY_MISMATCH=2, FORMAT=3, IO=4
- Shared error-handling decorator mapping exceptions → exit codes:
  - `KeyMismatchError` → 2 *(checked before IntegrityError — different parent: BraidKeyError)*
  - `IntegrityError` + subclasses → 1
  - `FormatError` + subclasses → 3
  - `OSError` / `FileNotFoundError` → 4
  - `BraidCodecError` catch-all → 1
- `_echo()` helper respecting verbose/quiet from `ctx.obj`

**Step 2** — Create `tests/cli/__init__.py` (empty)

**Step 3** — Create `tests/cli/test_cli.py` scaffold with CliRunner + shared fixtures (`key_file`, `sample_file`, `encoded_file`)

---

### Phase B: Commands (Steps 4-9 are independent after Step 1)

**Step 4** — `keygen` command
`braidcodec keygen [--sector TSR] [--strands 4] -o <key-file>`
- Calls `keygen()` → `key_to_bytes()` → write file
- Tests: output is 50 bytes, `key_from_bytes` roundtrips

**Step 5** — `encode` command
`braidcodec encode <input-file> -o <output-file> [--key <key-file>] [--sector TSR] [--strands 4] [--compression-level 0] [--workers N]`
- If `--key`: load via `key_from_bytes()`. Else: auto-generate + print key_id to stderr
- `encode()` → optional `compress()` if level > 0 → `stream.to_bytes()` → write file
- Tests: output exists, non-empty, decodable

**Step 6** — `decode` command
`braidcodec decode <input-file> -o <output-file> --key <key-file>`
- `EncodedStream.from_bytes(data)` → `decode(stream, key)` → write result
- Tests: full keygen→encode→decode roundtrip matches original

**Step 7** — `verify` command
`braidcodec verify <encoded-file> --key <key-file> [--fermion-check]`
- `verify(stream, key, fermion_check=...)` → print summary → exit 0 if valid, else 1
- Tests: valid → exit 0, wrong key → exit 2

**Step 8** — `inspect` command
`braidcodec inspect <encoded-file>`
- No key needed. Prints YAML-like metadata: version, sector, n_strands, block_count, total_bytes, checksum, timestamp
- Verbose: per-block summary (index, invariant_tier, generator count)
- Tests: output contains expected field names

**Step 9** — `benchmark` command
`braidcodec benchmark <input-file>`
- Auto-generates ephemeral key. Times encode/compress/decode/verify. Prints table with timing + throughput + compression ratio
- Tests: exit 0, output contains table headers

---

### Phase C: Integration & Polish

**Step 10** — End-to-end pipeline test *(depends on 4-9)*
- Full CliRunner pipeline: keygen → encode → inspect → verify → decode → compare
- Error scenarios: wrong key (exit 2), corrupt file (exit 3), missing file (exit 4)

**Step 11** — Update __init__.py *(parallel with 10)*
- Replace placeholder with `from braidcodec.cli.main import cli` + `__all__`

**Step 12** — Verify entry point *(parallel with 10)*
- `braidcodec = "braidcodec.cli.main:cli"` already in pyproject.toml — no changes needed
- `pip install -e ".[cli]"` → `braidcodec --help` shows 6 commands

---

### Relevant Files

| Action | File | Purpose |
|--------|------|---------|
| Create | `src/braidcodec/cli/main.py` | 6 click commands + error handler (~300-400 lines) |
| Create | `tests/cli/__init__.py` | Empty init |
| Create | `tests/cli/test_cli.py` | CliRunner tests (~250-300 lines) |
| Modify | __init__.py | Replace placeholder with `cli` import |

---

### Exception → Exit Code Mapping

| Exception | Exit | Note |
|-----------|------|------|
| *(none)* | 0 | Success |
| `KeyMismatchError` | 2 | Must check before IntegrityError (parent: BraidKeyError) |
| `IntegrityError` + subclasses | 1 | Writhe, Jones, Trace, Fermion, Checksum |
| `FormatError` + subclasses | 3 | Magic, Version, Digest |
| `OSError` / `FileNotFoundError` | 4 | I/O errors |
| `BraidCodecError` catch-all | 1 | Encoding, compression, etc. |

---

### Verification

1. `pytest tests/cli/ -v` — all CLI tests pass
2. `pip install -e ".[cli]"` → `braidcodec --help` shows 6 commands
3. Manual pipeline: keygen → encode → inspect → verify → decode → diff
4. Error exits: wrong key → 2, truncated file → 3, missing file → 4
5. `mypy src/braidcodec/cli/ --strict` passes
6. `ruff check src/braidcodec/cli/` clean
7. `pytest --cov=braidcodec.cli --cov-fail-under=85`

---

### Decisions

- **Public API**: Already complete (12 symbols) — no changes to __init__.py
- **Wire format**: `EncodedStream.to_bytes()` / `.from_bytes()` confirmed in schema.py
- **`encode` without `--key`**: Auto-generates key, prints key_id to stderr — does NOT auto-save key file
- **`--compression-level` default**: 0 (no compression) — user opts in explicitly
- **`benchmark` key**: Always auto-generated (ephemeral measurement tool)
- **click guard**: Not needed — CLI entry point naturally fails with ImportError if click absent