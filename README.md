# BraidCodec

Topological data codec that encodes arbitrary binary data as braid group equations — simultaneously encrypted, compressed, and integrity-verified through the mathematics of knot invariants.

## Overview

BraidCodec unifies encryption, compression, and integrity verification into a single mathematical framework based on braid groups. Instead of stacking AES + zstd + SHA-256, data is encoded into topological structures where:

- **Confidentiality** comes from sector-parameterized R-matrices acting as trapdoor functions
- **Compression** comes from topological equivalence class collapse
- **Integrity** comes from Yang-Baxter constraint violation detection

## Installation

```bash
pip install braidcodec
```

### Optional dependencies

```bash
pip install braidcodec[cli]      # Command-line interface
pip install braidcodec[quantum]  # Full qiskit for fermion bounds
pip install braidcodec[dev]      # Development tools
```

## Quick Start

```python
import braidcodec

# Generate a key
key = braidcodec.keygen(sector="TSR", n_strands=4)

# Encode data
encoded = braidcodec.encode(b"Hello, topology!", key)

# Decode data
decoded = braidcodec.decode(encoded, key)
assert decoded == b"Hello, topology!"

# Verify integrity
result = braidcodec.verify(encoded, key)
assert result.valid
```

## Architecture

```
bytes → chunks → generators → BraidEquations → Jones invariants → wire format
```

The algebra layer provides five core capabilities:
1. **R-matrix construction** — sector-parameterized 4×4 unitary matrices
2. **Braid contraction** — Kronecker products producing 2ⁿ×2ⁿ unitaries
3. **Topological invariants** — Jones polynomial via Kauffman bracket
4. **Equivalence checking** — Yang-Baxter morphism verification
5. **Occupation constraints** — fermionic bounds via Jordan-Wigner

## Security Disclaimer

BraidCodec is an experimental research project demonstrating topological data encoding. It is **not** a replacement for production cryptographic systems (AES-256, etc.). The security rests on the computational hardness of inverting the Jones polynomial (#P-hard in general), but this specific instantiation has not been cryptanalyzed.

## License

MIT
