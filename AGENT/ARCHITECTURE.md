# BraidCodec — Architecture Document

## 1. System Overview

BraidCodec is a layered codec with three tiers: a **physics substrate** (the algebra modules providing mathematical primitives), a **codec engine** (the serialization pipeline converting between bytes and braid representations), and a **security surface** (key management and integrity verification). The CLI and public API sit on top as thin orchestration layers.

The fundamental data flow is a pipeline:

```
ENCODE:  bytes → chunks → generator sequences → BraidEquations → contracted matrices → Jones invariants → EncodedBlocks → wire format bytes

DECODE:  wire format bytes → EncodedBlocks → generator sequences → integrity verification → chunks → bytes
```

The encode path touches every layer. The decode path skips matrix contraction entirely for the data recovery — it only contracts when running integrity checks. This asymmetry is important: decoding is significantly faster than encoding because `generators_to_bytes` is a pure arithmetic operation (mixed-radix conversion) while encoding requires the full tensor contraction for invariant computation.

---

## 2. Layer Architecture

### Layer 0 — Physics Substrate (`algebra/`)

This layer is **complete, verified, and frozen**. No codec or crypto module modifies it. It provides five capabilities consumed by the layers above:

**Capability A — R-matrix construction** (`braid_equations.get_sector_r_matrix`): Given a sector name and inverse flag, returns a 4×4 unitary matrix. This is the cryptographic primitive — the R-matrix family parameterized by sector and phase is the trapdoor. The phased-SWAP structure guarantees unitarity and Yang-Baxter satisfaction for any phase triple (α, β, γ), which means the crypto layer can perturb phases freely without breaking the mathematical guarantees.

**Capability B — Braid contraction** (`braid_equations.contract_braid_tensor`): Given a `BraidEquation`, produces the full 2^n × 2^n unitary matrix by Kronecker-producting generator matrices and multiplying left-to-right. This is the most expensive operation in the system — O(k · 4^n) where k is generator count and n is strand count. The result is cached on the `BraidEquation` object, so repeated access is O(1).

**Capability C — Topological invariants** (`braid_equations.jones_polynomial`, `writhe`, `kauffman_bracket`): Given a `BraidEquation`, computes the Jones polynomial via the Kauffman bracket state-sum model. The state sum iterates over 2^m smoothing states where m is the number of crossings (generators). This is the integrity fingerprint — it's a topological invariant, meaning it doesn't change under valid braid transformations but does change if generators are corrupted.

**Capability D — Equivalence checking** (`yang_baxter.verify_yang_baxter_morphism`, `check_yb_equivalence_catlab`): Given two `BraidEquation` objects, determines whether they represent the same topological object by contracting both and comparing matrices. Used by the compressor to verify that rewrite rules preserve meaning, and by the integrity checker to validate encoded data.

**Capability E — Occupation constraints** (`fermion_bounds.*`): A completely independent validation channel. Maps generator sequences to fermionic occupation patterns and verifies that physical constraints (Pauli exclusion, parity conservation) are maintained. This provides defense-in-depth — even if an attacker manages to produce a corrupted braid that passes Jones polynomial checks, the fermion bounds check catches inconsistencies in the occupation pattern.

The key architectural property of Layer 0 is that it has **zero upward dependencies**. It knows nothing about chunks, wire formats, keys, or CLI. It is a pure mathematical library.

### Layer 1 — Codec Engine (`codec/`)

This layer owns the data transformation pipeline. It has five modules, each with a single responsibility:

**`chunker`** owns the bijection between byte blocks and generator sequences. It is a pure arithmetic module — no linear algebra, no braids, no physics. It implements a mixed-radix numeration system where each digit position has base `2(n-1)` corresponding to the `n-1` positive and `n-1` negative generators on `n` strands. The chunker computes block sizes, handles padding, and provides streaming iteration over input data.

**`schema`** owns the data structures and wire format. It defines `EncodedBlock` and `EncodedStream` as dataclasses with `to_bytes`/`from_bytes` methods. It knows about msgpack serialization and the binary framing (magic bytes, length prefixes, trailing digest). It does not know how blocks are created — only how they are stored and retrieved.

**`encoder`** orchestrates the encode pipeline by calling chunker, constructing `BraidEquation` objects, invoking Layer 0 capabilities (contraction, Jones polynomial, writhe), and assembling `EncodedBlock` objects into an `EncodedStream`. It owns the parallelism strategy (process pool over independent blocks).

**`decoder`** orchestrates the decode pipeline. It deserializes the wire format via schema, extracts generator sequences from blocks, invokes the chunker's inverse mapping, and reassembles the original bytes. It performs inline integrity checks (writhe pre-check, optional Jones re-computation) and raises `IntegrityError` on failure.

**`compressor`** is a post-processing module that operates on already-encoded `EncodedBlock` objects. It applies topological rewrite rules (inverse cancellation, far commutativity, Yang-Baxter rewrites) to shorten generator sequences while preserving the braid's topological class. It calls Layer 0's equivalence checking to verify each rewrite.

The codec layer imports from Layer 0 but never modifies Layer 0 objects except by constructing new `BraidEquation` instances. The `braid_matrix` cache on `BraidEquation` is the only mutation, and it's idempotent.

### Layer 2 — Security Surface (`crypto/`)

This layer wraps the codec with key management and integrity verification. It sits above the codec — it calls `encode`/`decode` and adds cryptographic context.

**`keys`** owns `BraidKey` generation, serialization, and validation. A key is a tuple of `(sector, n_strands, theta_offset)` that fully determines the R-matrix used in encoding. The `theta_offset` is the secret — it perturbs the sector's base phase angle, creating a custom R-matrix that still satisfies Yang-Baxter but produces different invariants. Key generation uses `secrets.token_bytes` for the offset, and key serialization uses a simple binary format with BLAKE3-based key ID derivation.

**`integrity`** owns the full verification triad: Yang-Baxter structural checks, Jones polynomial re-computation, and optional fermion bounds validation. It aggregates per-block results into a `VerificationResult` with detailed failure diagnostics.

The security layer never touches raw bytes or generator arithmetic. It delegates all data transformation to the codec layer and adds the key context and verification logic.

### Layer 3 — Interface (`cli/`, `__init__.py`)

The thinnest layer. The CLI module parses arguments via click and dispatches to codec and crypto functions. The package `__init__.py` re-exports the six public symbols (`encode`, `decode`, `compress`, `keygen`, `verify`, `BraidKey`). No logic lives here — only wiring.

---

## 3. Data Flow — Encode Path (Detailed)

I'll trace a concrete example: encoding the 5-byte input `b"Hello"` with a TSR key on 4 strands and 32 generators per block.

**Step 1 — Key resolution.** The caller provides a `BraidKey(sector="TSR", n_strands=4, theta_offset=0.42)`. The encoder extracts `sector` and `n_strands` from the key. The `theta_offset` is passed down to a modified `get_sector_r_matrix` call that adds the offset to the base sector angle. This modified R-matrix is used for all subsequent contraction and invariant computation for this encode session.

This is an important architectural detail: **the encoder does not call the base `get_sector_r_matrix` directly**. Instead, it constructs a `_KeyedRMatrixProvider` (internal class, not exported) that wraps the key's phase offset. This provider is injected into `BraidEquation` construction via a `sector_params` dict that overrides the default sector angles. This keeps the algebra layer pure — it still receives `(sector, theta)` pairs — while the crypto layer controls what those pairs are.

The alternative design (modifying `get_sector_r_matrix` to accept an offset parameter) would work but couples the algebra layer to the key concept. The provider pattern keeps the layers cleanly separated.

**Step 2 — Chunking.** The chunker computes the block byte capacity. With `n_strands=4`, each generator index is in `{-3, -2, -1, 1, 2, 3}` — six values. Each generator carries `floor(log2(6))` = 2.58 bits. With 32 generators per block, each block encodes `32 × log2(6) = 82.6` bits ≈ 10 bytes. So the chunker sets `block_size = 10`.

The 5-byte input `b"Hello"` fits in a single block. The chunker yields one tuple: `(block_index=0, chunk=b"Hello\x00\x00\x00\x00\x00", original_length=5)`. The chunk is right-padded to 10 bytes.

**Step 3 — Generator mapping.** `bytes_to_generators` interprets the 10-byte chunk as a big-endian integer N, then converts N to base-6 with 32 digits. Each digit d maps to a generator: d ∈ {0,1,2} → `+(d+1)` = σ₁, σ₂, σ₃; d ∈ {3,4,5} → `-(d-2)` = σ₁⁻¹, σ₂⁻¹, σ₃⁻¹.

The output is a `list[int]` of length 32, e.g., `[1, 3, -2, 1, 1, -3, 2, ...]`.

**Step 4 — Braid construction.** The encoder constructs `BraidEquation(n_strands=4, generators=[1, 3, -2, ...], sector="TSR")`. At this point the braid exists but has no contracted matrix.

**Step 5 — Contraction.** `contract_braid_tensor` builds the 16×16 unitary matrix by Kronecker-producting 32 generator matrices and multiplying. For each generator g in the sequence:

- Compute `abs(g)` to get the strand index i.
- Call the keyed R-matrix provider to get the 4×4 R-matrix (with phase offset applied).
- Build the 16×16 matrix: `I_{2^{i-1}} ⊗ R ⊗ I_{2^{n-i-1}}`.
- Left-multiply into the running product.

After all 32 generators, the result is cached on `braid.braid_matrix`.

**Step 6 — Invariant computation.** The encoder calls `writhe(braid)` — this is O(32), just summing signs. Then `jones_polynomial(braid)` — this computes the Kauffman bracket via the 2^32 state sum... which is 4 billion states.

This is the computational bottleneck and it reveals a critical architectural decision: **the state-sum Jones polynomial is infeasible for blocks with more than ~20 generators**. For 32 generators, 2^32 states is tractable on modern hardware (a few seconds), but barely. For 64 generators, 2^64 states is impossible.

The architecture handles this with a **tiered invariant strategy**:

- **Tier 1 (always computed)**: Writhe. O(k) where k is generator count. Catches generator sign corruption.
- **Tier 2 (computed for k ≤ 24)**: Full Jones polynomial via Kauffman bracket state sum. Catches any single-generator corruption with high probability.
- **Tier 3 (computed for k > 24)**: Matrix-trace invariant. Instead of the full Jones polynomial, the encoder computes `trace(contracted_matrix)` and stores it. The trace of a unitary matrix is a weaker invariant than Jones but is O(k · 4^n) to compute (just the contraction cost, no state sum). This catches most corruptions but is not a topological invariant — it depends on the matrix representation, not just the braid's topology.
- **Tier 4 (always computed)**: BLAKE3 checksum on the original bytes. The last line of defense, independent of all braid mathematics.

The schema stores which tier of invariant was used, so the decoder and verifier know what to recompute.

**Step 7 — Simplification.** `simplify_braid` runs adjacent inverse cancellation. If the generator sequence `[..., 2, -2, ...]` appears, the pair is removed. This shortens the generator list, which is the compression. The simplified generators replace the original in the `EncodedBlock`. The Jones polynomial (or trace invariant) remains the same — simplification preserves topology.

**Step 8 — Block assembly.** The encoder constructs:

```python
EncodedBlock(
    generators=[1, 3, -2, ...],  # post-simplification
    n_strands=4,
    sector="TSR",
    jones=(0.234+0.891j),  # or None if tier 3
    trace_invariant=(2.45-1.23j),  # or None if tier 2
    writhe=7,
    block_index=0,
    original_length=5,
    invariant_tier=2,
)
```

**Step 9 — Stream assembly.** The single block is wrapped in:

```python
EncodedStream(
    version=1,
    blocks=[block],
    n_strands=4,
    sector="TSR",
    total_bytes=5,
    checksum=blake3(b"Hello").digest(),
    timestamp=time.time_ns(),
    metadata={"encoder_version": "0.1.0"},
)
```

**Step 10 — Wire format serialization.** `EncodedStream.to_bytes()` produces the binary wire format: magic + version + header + blocks + trailing digest.

---

## 4. Data Flow — Decode Path (Detailed)

Continuing the example, decoding the wire format bytes back to `b"Hello"`.

**Step 1 — Deserialization.** `EncodedStream.from_bytes(wire_data)` parses the binary framing, verifies the trailing BLAKE3 digest covers the entire payload (if it doesn't, the file itself is corrupted — raise `FormatError` before any braid computation), unpacks the msgpack header and blocks.

**Step 2 — Key validation.** The decoder checks `stream.sector == key.sector` and `stream.n_strands == key.n_strands`. If they don't match, raise `KeyMismatchError`. This is a fast rejection — no computation needed.

**Step 3 — Per-block integrity (fast path).** For each block, the decoder recomputes `writhe` from the stored generators and compares to `block.writhe`. Writhe computation is O(k) — essentially free. If it mismatches, the block is corrupted and the decoder raises `IntegrityError` immediately without proceeding to the expensive Jones recomputation.

**Step 4 — Per-block integrity (optional full path).** If the caller requested full verification (the default for `decode`, skippable via `verify=False` for trusted sources), the decoder recomputes the invariant according to `block.invariant_tier`. For tier 2, this means recomputing the Jones polynomial — which requires constructing a `BraidEquation`, contracting it (using the key's R-matrix), and running the state sum. For tier 3, it means contracting and taking the trace. The recomputed value is compared to the stored value within floating-point tolerance (`|Δ| < 1e-8` for Jones, `|Δ| < 1e-10` for trace).

**Step 5 — Generator-to-bytes.** `generators_to_bytes(block.generators, block.n_strands, block.original_length)` reverses the mixed-radix mapping: each generator maps to a digit in base `2(n-1)`, the digits are assembled into a big-endian integer, converted to bytes, and truncated to `original_length`.

**Step 6 — Stream reassembly.** Blocks are processed in `block_index` order (the schema guarantees they're stored in order, but the decoder sorts by index defensively). Chunks are concatenated.

**Step 7 — Final checksum.** BLAKE3 of the reassembled bytes is compared to `stream.checksum`. This catches any systematic bug in the decode pipeline — if the mixed-radix arithmetic has an off-by-one, or if a block was decoded in the wrong order, this catches it.

---

## 5. Data Flow — Compression Path

Compression operates on an already-encoded `EncodedStream`, producing a new `EncodedStream` with shorter generator sequences. It is a pure-to-pure transformation — no side effects, no mutation of the input.

**Level 0 — Inverse cancellation.** This is what `simplify_braid` already does during encoding. The compressor at level 0 is a no-op if the encoder already simplified. It exists for the case where the caller encoded with `simplify=False` (useful for debugging or when measuring raw vs. compressed size).

The algorithm is a single pass with restarts: scan left-to-right for adjacent inverse pairs `(g, -g)`, delete the pair, restart from the beginning. Worst case O(k²) for a sequence that reduces to empty (e.g., `[1, 2, -2, -1]` requires 2 restarts). Average case is much better — O(k) with a small constant for typical random generator sequences.

**Level 1 — Far commutativity normalization.** Two generators σᵢ and σⱼ commute when |i-j| ≥ 2 (they act on disjoint strand pairs). The compressor identifies maximal windows of pairwise-commuting generators and sorts them into a canonical order (ascending by |index|, positive before negative). After sorting, it re-runs level 0 cancellation. This exposes cancellable pairs that were hidden by intervening commuting generators.

Example: `[1, 3, -1, 2]` on 5 strands. Generators σ₁ and σ₃ commute (|1-3|=2 ≥ 2), so we can reorder to `[1, -1, 3, 2]`, and now the `(1, -1)` pair cancels, yielding `[3, 2]`.

The algorithm: build a dependency graph where generators are nodes and edges connect non-commuting pairs (|i-j| < 2). Find connected components. Within each component, the order is constrained; across components, we're free to reorder. Use topological sort within components and canonical ordering across them. Then re-run level 0.

Correctness is guaranteed because commutativity of distant generators is an algebraic identity of the braid group, not an approximation. The compressor verifies the result by calling `verify_yang_baxter_morphism` on the original and compressed braids — both must produce the same contracted matrix.

**Level 2 — Yang-Baxter rewriting.** The braid relation σᵢσᵢ₊₁σᵢ = σᵢ₊₁σᵢσᵢ₊₁ can sometimes be applied to enable further cancellations. The compressor searches for patterns where applying the YB relation (in either direction) followed by level 0+1 compression yields a shorter sequence.

This is a bounded breadth-first search. The state space is the set of generator sequences equivalent to the original braid. The compressor explores up to `max_rewrites` (default 100) YB applications, keeping track of the shortest sequence found. Each candidate is verified via matrix comparison.

The search is not exhaustive — the word problem for braid groups is decidable but the search space grows exponentially. The bound keeps it practical. For typical block sizes (16-32 generators), level 2 completes in milliseconds.

**Compression output.** The compressor returns a new `EncodedBlock` with shorter `generators`, the same `jones`/`trace_invariant`/`writhe` (recomputed from the compressed braid to verify), and updated `original_length` (unchanged — compression doesn't affect the data, only the encoding).

---

## 6. Data Flow — Integrity Verification

The verification pipeline is the most layered part of the system, using four independent validation channels.

**Channel 1 — Structural validation.** For each block, verify that all generator indices are in range `[1, n_strands-1]` and non-zero. This catches gross corruption (e.g., a generator value of 0 or 99 on a 4-strand braid). This is `validate_braid_category` from the yang_baxter module. Cost: O(k) per block.

**Channel 2 — Writhe verification.** Recompute writhe from generators, compare to stored value. Cost: O(k) per block. Catches any corruption that changes the sign or index of a generator (since writhe = Σ sign(gᵢ)).

**Channel 3 — Invariant verification.** Recompute the Jones polynomial or trace invariant (depending on `invariant_tier`), compare to stored value. This requires full matrix contraction and (for Jones) the state-sum computation. Cost: O(k · 4^n) for contraction plus O(2^k) for Jones state sum. This is the expensive check but catches corruptions that preserve writhe (e.g., swapping σ₁ with σ₂ on the same sign preserves writhe but changes Jones).

**Channel 4 — Fermion bounds verification .** Map the generator sequence to a fermionic occupation trajectory: positive generator σᵢ → `occupy(site=i)`, negative generator σᵢ⁻¹ → `vacate(site=i)`. Walk the sequence and verify that no `occupy` hits an already-occupied site (Pauli exclusion violation) and no `vacate` hits an already-empty site. Track total parity throughout and verify it matches the expected value at the end.

This channel catches a class of corruption that the other channels might miss: sequences that produce valid unitary matrices and valid Jones polynomials but are physically inconsistent under fermionic interpretation. It's defense-in-depth — the probability of all four channels simultaneously passing on corrupted data is negligible.

The verifier runs channels 1-3 by default and channel 4 on request (`fermion_check=True`). It short-circuits: if channel 1 fails, channels 2-4 are skipped for that block (why compute Jones on a structurally invalid braid?).

**Channel 5 — Global checksum.** After all blocks pass individual checks, the verifier decodes the full byte stream and computes BLAKE3, comparing to the stored checksum. This is the only verification channel that requires full decoding. It catches systematic errors that per-block checks miss (e.g., blocks are individually valid but in the wrong order, or a block is duplicated/missing).

---

## 7. Key Architecture

### 7.1 Key Structure

A `BraidKey` fully determines the R-matrix used in encoding. It has three parameters:

- **`sector`**: selects the base phase angle family (TSR → πC, Ising → π/4, etc.)
- **`n_strands`**: determines matrix dimension (2^n × 2^n) and generator alphabet size
- **`theta_offset`**: a float in [0, 2π) that perturbs the base sector angle

The effective angle for encoding is `θ_effective = θ_sector + theta_offset`. The phased-SWAP R-matrix is then parameterized as `α = θ_eff, β = θ_eff, γ = 3·θ_eff`.

### 7.2 Key-Algebra Interface

The key must modify the R-matrix computation without altering the algebra layer's code. The architecture uses a **context injection pattern**:

The encoder constructs a `_SectorOverride` dict:

```python
sector_override = {
    "theta": base_sector_theta(key.sector) + key.theta_offset
}
```

This dict is threaded through to `BraidEquation` construction. The `BraidEquation` class gains an optional `_sector_params` slot (default None, meaning "use base sector angles"). When set, `get_sector_r_matrix` uses the overridden theta instead of computing it from the sector name.

This keeps the algebra module backward-compatible — existing code that constructs `BraidEquation` without `_sector_params` continues to work identically. The crypto layer is the only code that ever sets `_sector_params`.

### 7.3 Key Derivation

`keygen()` produces a key by:

1. Generating 32 random bytes via `secrets.token_bytes(32)`.
2. Interpreting the first 8 bytes as a big-endian float64 and mapping to `[0, 2π)` via modular arithmetic: `theta_offset = struct.unpack('>d', raw[:8])[0] % (2 * math.pi)`.
3. Computing `key_id = blake3(sector.encode() + n_strands.to_bytes(4, 'big') + struct.pack('>d', theta_offset)).hexdigest()[:32]`.
4. Validating the resulting R-matrix: construct it, verify unitarity (`||R†R - I|| < 1e-12`), verify Yang-Baxter on 3 strands.

### 7.4 Key Serialization

Binary format:

```
[4 bytes: magic "BRDK"]
[1 byte: version]
[1 byte: sector enum (0=Identity, 1=TSR, 2=Ising, 3=Fibonacci, 4=SU2k2)]
[4 bytes: n_strands, big-endian uint32]
[8 bytes: theta_offset, big-endian float64]
[32 bytes: BLAKE3 of the above]
```

Total: 50 bytes per key. Human-readable export (for CLI) is base64-encoded with a `braidkey:` prefix.

---

## 8. Wire Format — Binary Layout

The complete encoded file has this structure:

```
Offset   Size   Field
──────   ────   ─────
0        4      Magic: ASCII "BRDC"
4        2      Format version: uint16 big-endian (currently 1)
6        4      Header length: uint32 big-endian (byte count of header blob)
10       H      Header blob: msgpack-encoded dict {
                    "n_strands": int,
                    "sector": str,
                    "total_bytes": int,
                    "checksum": bytes(32),
                    "timestamp": int,
                    "metadata": dict[str, str],
                }
10+H     4      Block count: uint32 big-endian
14+H     ...    Block array: for each block:
                    [4 bytes: block payload length L, uint32 big-endian]
                    [L bytes: msgpack-encoded dict {
                        "generators": list[int],
                        "n_strands": int,
                        "sector": str,
                        "jones_real": float | None,
                        "jones_imag": float | None,
                        "trace_real": float | None,
                        "trace_imag": float | None,
                        "writhe": int,
                        "block_index": int,
                        "original_length": int,
                        "invariant_tier": int,
                    }]
EOF-32   32     Trailing digest: BLAKE3 of everything from offset 0 to EOF-32
```

Design rationale for key decisions:

**Why msgpack inside a custom framing instead of pure msgpack?** The outer framing (magic, version, header length, block count, per-block length) allows streaming decode — a decoder can read the header, then seek to any block by offset without deserializing every preceding block. Pure msgpack would require sequential deserialization. This matters for large files with thousands of blocks.

**Why store Jones polynomial as two floats (real, imag) instead of a complex?** Msgpack has no native complex type. Storing as two named floats is unambiguous and avoids any serialization library's custom complex encoding.

**Why a trailing digest instead of a header digest?** The trailing digest covers the entire file including all blocks. A header-only digest would not detect block corruption or truncation. The trailing position means the encoder can stream blocks without knowing the final digest until the end.

**Version field**: allows future format changes without breaking old decoders. Version 1 is the initial format. If the block schema changes, version bumps to 2, and decoders can dispatch on version.

---

## 9. Concurrency Model

### 9.1 Encode Parallelism

Blocks are independent — block i's encoding depends only on its chunk of input bytes and the key, not on any other block. This makes encoding embarrassingly parallel.

The encoder uses `concurrent.futures.ProcessPoolExecutor` (not `ThreadPoolExecutor`) because the workload is CPU-bound (matrix multiplication, state-sum iteration). NumPy releases the GIL for array operations, so thread-based parallelism would give some benefit, but process-based parallelism avoids GIL contention entirely and scales linearly with core count.

The parallelism architecture:

```
Main process:
  1. Chunk input → list of (block_index, chunk, original_length)
  2. Submit each to process pool as encode_block(chunk, key_params, block_index, original_length)
  3. Collect futures in block_index order
  4. Assemble EncodedStream

Worker process (encode_block):
  1. bytes_to_generators(chunk, n_strands) → generators
  2. BraidEquation(n_strands, generators, sector, _sector_params)
  3. contract_braid_tensor(braid) → matrix
  4. jones_polynomial(braid) or trace(matrix) → invariant
  5. writhe(braid) → int
  6. simplify_braid(braid) → compressed_generators
  7. Return EncodedBlock
```

Each worker is fully self-contained. The `key_params` are serialized as a simple dict (sector, n_strands, theta_offset) and passed to each worker — no shared state.

**Worker count**: defaults to `os.cpu_count() - 1` (leave one core for the main process assembling results), capped at 8 to avoid memory pressure from multiple large matrix computations. Overridable via `max_workers` parameter.

**Memory consideration**: each worker holds at most one 2^n × 2^n complex matrix in memory. For n_strands=4, that's 16×16×16 bytes = 4KB — negligible. For n_strands=8, it's 256×256×16 = 1MB — still fine. The memory bottleneck is the state-sum for Jones polynomial (2^k intermediate values), which for k=32 is 4 billion complex numbers × 16 bytes = 64GB. This is why the tiered invariant strategy exists — blocks with k > 24 fall back to the trace invariant.

### 9.2 Decode Parallelism

Decoding is lighter but also parallelizable. If full integrity verification is enabled, each block's invariant recomputation is independent. The decoder uses the same process pool pattern but with smaller workers (just contraction + invariant computation, no chunking or simplification).

If integrity verification is disabled (`verify=False`), decoding is purely sequential and arithmetic — `generators_to_bytes` for each block, concatenate. This is fast enough that parallelism adds overhead rather than speedup.

### 9.3 Compression Parallelism

Level 0 and level 1 compression are per-block and parallelizable. Level 2 (YB rewriting) is also per-block but more expensive per worker. The compressor uses the same pool pattern.

---

## 10. Error Handling Architecture

### 10.1 Exception Hierarchy

```
BraidCodecError (base)
├── FormatError           — wire format parsing failures
│   ├── MagicMismatchError
│   ├── VersionError
│   └── DigestError       — trailing BLAKE3 doesn't match
├── KeyError_             — key-related failures (named to avoid shadowing builtins)
│   ├── KeyMismatchError  — key doesn't match encoded stream
│   └── KeyValidationError — key's R-matrix fails unitarity/YBE
├── IntegrityError        — data corruption detected
│   ├── WritheError       — writhe mismatch
│   ├── JonesError        — Jones polynomial mismatch
│   ├── TraceError        — trace invariant mismatch
│   ├── FermionError      — fermion bounds violation
│   └── ChecksumError     — BLAKE3 mismatch on decoded bytes
├── EncodingError         — failures during encode
│   ├── ChunkError        — input chunking failure
│   └── ContractionError  — matrix contraction failure (numerical)
└── CompressionError      — failures during compression
    └── RewriteVerificationError — YB rewrite didn't preserve matrix
```

Every exception carries structured data (block index, expected vs actual values, tolerance used) so that callers can programmatically diagnose failures.

### 10.2 Error Recovery

The decoder supports a **partial decode mode** (`on_error="skip"`) where corrupted blocks are replaced with zero-bytes of the correct length and a warning is emitted. This is useful for forensic recovery of partially corrupted encoded files. The default mode (`on_error="raise"`) aborts on the first corrupted block.

### 10.3 Numerical Tolerance

Floating-point comparison is used in three places:

- Jones polynomial comparison: `|Δ| < 1e-8`. This tolerance accounts for accumulation of floating-point error through the Kauffman bracket state sum (2^m additions of complex products). For m ≤ 24, empirical testing shows the error stays below 1e-10; the 1e-8 threshold provides a safety margin.
- Trace invariant comparison: `|Δ| < 1e-10`. The trace is a single sum of 2^n diagonal elements — much less error accumulation than the state sum.
- R-matrix unitarity validation: `||R†R - I||_F < 1e-12`. This is the tightest tolerance because R-matrices are constructed analytically (exponentials of known angles), so error should be at machine epsilon.

These tolerances are defined as module-level constants in `_types.py` and used consistently throughout.

---

## 11. Performance Characteristics

### 11.1 Computational Complexity

| Operation | Time Complexity | Dominant Cost |
|---|---|---|
| `bytes_to_generators` | O(k) | Mixed-radix division |
| `generators_to_bytes` | O(k) | Mixed-radix multiplication |
| `contract_braid_tensor` | O(k · 4^n) | k matrix multiplications of 2^n × 2^n matrices |
| `writhe` | O(k) | Sum of signs |
| `jones_polynomial` | O(2^k · k) | State-sum over 2^k smoothings |
| `simplify_braid` | O(k²) worst case | Repeated scan for adjacent inverses |
| `compress (level 1)` | O(k log k) | Sort + rescan |
| `compress (level 2)` | O(R · k · 4^n) | R rewrites, each verified by contraction |
| `verify (channels 1-2)` | O(k) per block | Structural check + writhe |
| `verify (channel 3)` | O(k · 4^n + 2^k) per block | Contraction + state sum |

Where k = generators per block, n = n_strands, R = max rewrites (default 100).

### 11.2 Practical Parameter Guidance

| n_strands | Matrix size | Encode throughput (est.) | Good for |
|---|---|---|---|
| 3 | 8×8 | ~5 MB/s | Fast encoding, low security margin |
| 4 | 16×16 | ~2 MB/s | Default: good balance |
| 5 | 32×32 | ~500 KB/s | Higher security, slower |
| 6 | 64×64 | ~100 KB/s | Research use only |
| 8 | 256×256 | ~10 KB/s | Impractical for large files |

These estimates assume serial encoding with Jones polynomial computation on a single core. Parallelism scales near-linearly: 4 cores → ~4x throughput.

### 11.3 Memory Profile

Base memory (import + key setup): ~50MB (numpy overhead).

Per-block peak memory during encoding:
- Matrix: 2^n × 2^n × 16 bytes (complex128). For n=4: 4KB. For n=6: 64KB.
- State sum workspace: 2^k × 16 bytes. For k=24: 256MB. For k=32: 64GB (triggers tier 3 fallback).

The tier 3 fallback threshold is set at k=24 by default, keeping per-block peak memory under 300MB.

---

## 12. Extensibility Points

### 12.1 Custom Sectors

The `VALID_SECTORS` frozenset in `braid_equations.py` is the gatekeeper. To add a new sector:

1. Add the sector name to `VALID_SECTORS`.
2. Add a new `elif` branch in `get_sector_r_matrix` mapping the name to a base angle θ.
3. Add a corresponding branch in `get_kauffman_A`.

The rest of the system (encoder, decoder, verifier, compressor) works unchanged because it's parameterized by sector name strings, not sector-specific logic.

### 12.2 Custom Wire Formats

The `schema.py` module defines `to_bytes`/`from_bytes` as methods on `EncodedStream`. Alternative formats (e.g., JSON for debugging, protobuf for production) can be added as additional methods (`to_json`/`from_json`) without changing any other module.

### 12.3 Alternative Invariants

The tiered invariant strategy is implemented as a dispatch on `invariant_tier`. New tiers can be added (e.g., HOMFLY polynomial, Alexander polynomial) by:

1. Adding the computation function to `braid_equations.py`.
2. Adding a new tier number to the schema.
3. Adding the dispatch branch in encoder and verifier.

### 12.4 Plugin Compressors

The `compressor.py` module's `compress` function accepts a `level` parameter. Levels 0-2 are built-in. Level 3 (Markov stabilization) is documented in the PRD as optional. The architecture supports arbitrary levels via a registry pattern:

```python
COMPRESSORS: dict[int, Callable[[EncodedBlock], EncodedBlock]] = {
    0: _compress_level0,
    1: _compress_level1,
    2: _compress_level2,
}
```

Users or future contributors can register additional compressors.

---

## 13. Security Architecture

### 13.1 Threat Model

BraidCodec's security properties are **best-effort and unaudited**. The architecture assumes the following threat model for design purposes only:

**Attacker capabilities**: observes the encoded wire format (ciphertext-only). Does not possess the key. May modify the wire format (active attacker).

**Confidentiality goal**: without the `theta_offset`, the attacker cannot reconstruct the R-matrix and therefore cannot invert the generator-to-bytes mapping. The security rests on the hardness of determining an R-matrix from braid invariants (related to the #P-hardness of Jones polynomial computation).

**Integrity goal**: any modification to the wire format (generator values, block order, metadata) is detected with overwhelming probability by the verification triad.

**What BraidCodec does NOT protect against**: side-channel attacks, known-plaintext attacks (if the attacker knows part of the plaintext, they may be able to recover the key by solving for `theta_offset`), or quantum attacks (a quantum computer with a Jones polynomial oracle could potentially break the scheme). These limitations are documented in the README and theory docs.

### 13.2 Defense-in-Depth Layers

```
Layer 5: BLAKE3 checksum (global, independent of braid math)
Layer 4: Fermion bounds (independent physics channel)
Layer 3: Jones polynomial / trace invariant (topological)
Layer 2: Writhe (algebraic, fast)
Layer 1: Structural validation (generator range check)
Layer 0: Wire format digest (detects file-level tampering)
```

An attacker must simultaneously fool all six layers to inject undetected corrupted data. Each layer uses a different mathematical property, so a corruption that evades one layer is likely caught by another.

---

## 14. Testing Architecture

### 14.1 Test Pyramid

```
                    ┌─────────┐
                    │   E2E   │  ← 5 tests: full CLI encode/decode/verify cycle
                   ┌┴─────────┴┐
                   │Integration │  ← 20 tests: round-trip, cross-module
                  ┌┴───────────┴┐
                  │   Property   │  ← 15 strategies: hypothesis-driven
                 ┌┴─────────────┴┐
                 │     Unit       │  ← 200+ tests: per-function, per-module
                 └───────────────┘
```

### 14.2 Critical Test Invariants

These properties must hold across all test runs and are tested both explicitly and via hypothesis:

1. **Round-trip**: `decode(encode(data, key), key) == data` for all `data` and valid `key`.
2. **Jones invariance**: `jones_polynomial(b) == jones_polynomial(simplify_braid(b))` for all braids.
3. **YBE**: `contract(σᵢσᵢ₊₁σᵢ) == contract(σᵢ₊₁σᵢσᵢ₊₁)` for all i, all sectors.
4. **Unitarity**: `R†R == I` for all sector + theta_offset combinations.
5. **Parity conservation**: `get_total_parity` is unchanged through any sequence of hops.
6. **Writhe additivity**: `writhe(b1 * b2) == writhe(b1) + writhe(b2)`.
7. **Bijective numeration**: `generators_to_bytes(bytes_to_generators(chunk, n), n, len(chunk)) == chunk` for all chunks.
8. **Integrity detection**: single-bit flip in any generator → at least one verification channel fails.

### 14.3 Hypothesis Strategies

```python
# Braid strategy: random valid braids
braids = st.builds(
    BraidEquation,
    n_strands=st.integers(2, 6),
    generators=st.lists(st.integers(-5, 5).filter(lambda x: x != 0), max_size=20),
    sector=st.sampled_from(list(VALID_SECTORS)),
).filter(lambda b: all(abs(g) < b.n_strands for g in b.generators))

# Byte stream strategy
byte_streams = st.binary(min_size=0, max_size=10_000)

# Key strategy
keys = st.builds(
    BraidKey,
    sector=st.sampled_from(list(VALID_SECTORS)),
    n_strands=st.integers(3, 5),
    theta_offset=st.floats(0, 2 * math.pi, allow_nan=False, allow_infinity=False),
)
```

---

## 15. Operational Characteristics

### 15.1 Logging

BraidCodec uses Python's `logging` module with a `"braidcodec"` logger hierarchy. No print statements anywhere.

- `braidcodec.codec` — encode/decode progress (INFO: block count, timing; DEBUG: per-block details)
- `braidcodec.crypto` — key operations (INFO: key generated/loaded; WARNING: weak theta_offset)
- `braidcodec.algebra` — mathematical operations (DEBUG only: matrix norms, invariant values)

The CLI configures logging via `--verbose` / `--quiet` flags.

### 15.2 Progress Reporting

For long-running operations (encoding large files), the encoder emits progress callbacks. The CLI uses these to display a progress bar (via `click.progressbar`). The library API accepts an optional `progress: Callable[[int, int], None]` parameter (current block, total blocks).

### 15.3 Determinism

Given the same input bytes and the same key, BraidCodec produces **bit-identical** output except for the `timestamp` field. This is important for reproducibility. The `timestamp` can be overridden via `metadata={"timestamp": "..."}` for fully deterministic output in testing.

The compressor at level 2 (YB rewriting) explores rewrites in a deterministic order (lexicographic by rewrite position), so it is also deterministic. There is no randomness anywhere in the pipeline except for key generation.

---

This architecture gives you a complete engineering blueprint. Every module has a single responsibility, every data flow is traceable end-to-end, and the layer boundaries are strict enough that you can implement and test each module independently. The chunker is the first thing to build — it has zero dependencies on anything else and the bijective numeration is the foundation that the entire codec rests on. Once `bytes_to_generators` and `generators_to_bytes` pass the round-trip hypothesis test, everything else composes on top of it.