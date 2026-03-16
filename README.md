# BraidCodec

[![CI](https://github.com/mehdiskouri/BraidCodec/actions/workflows/ci.yml/badge.svg)](https://github.com/mehdiskouri/BraidCodec/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mehdiskouri/BraidCodec/branch/main/graph/badge.svg)](https://codecov.io/gh/mehdiskouri/BraidCodec)
[![PyPI](https://img.shields.io/pypi/v/braidcodec)](https://pypi.org/project/braidcodec/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Topological data codec that encodes arbitrary binary data as braid group equations — simultaneously encrypted, compressed, and integrity-verified through the mathematics of knot invariants.

## How It Works

```mermaid
flowchart LR
    A[bytes] --> B[chunker]
    B --> C[generators]
    C --> D[BraidEquations]
    D --> E[invariants]
    E --> F[wire format .brdc]
    F --> G[verify / decode]
    G --> H[bytes]
```

Data is split into chunks, profiled with deterministic topology signatures, transformed into topology-conditioned braid generators (while preserving decode bijection), then assembled into `BraidEquation` objects with sector-parameterized R-matrices. Topological invariants (writhe, Jones/trace tiers, BLAKE3 checksum) are computed per block for integrity verification. The result is a self-verifying `.brdc` wire format (`v2`, with `v1` read compatibility).

## Overview

BraidCodec unifies encryption, compression, and integrity verification into a single mathematical framework based on braid groups. Instead of stacking AES + zstd + SHA-256, data is encoded into topological structures where:

- **Confidentiality** comes from sector-parameterized R-matrices acting as trapdoor functions
- **Compression** comes from topological equivalence class collapse (3 levels)
- **Integrity** comes from 5+1 channel verification (structural, writhe, invariant, fermion, topology, checksum)

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

# Verify integrity (topology channel optional)
result = braidcodec.verify(encoded, key, topology_check=True)
assert result.valid
```

## CLI Usage

```bash
# Generate a key
braidcodec keygen -o my.key

# Encode a file
braidcodec encode data.bin -o data.brdc --key my.key

# Decode a file
braidcodec decode data.brdc -o restored.bin --key my.key

# Verify integrity (topology channel optional)
braidcodec verify data.brdc --key my.key --topology-check

# Inspect stream metadata
braidcodec inspect data.brdc

# Run encode/decode/verify benchmark
braidcodec benchmark
```

### Reconstructive Compact Transport

Reconstructive mode supports compact transport with two policies:

- `enabled`: `ps1.*` compact transport with commitment validation (`rc3`)
- `lean`: `ps2.*` compact transport without reconstructive commitment metadata

Example:

```bash
braidcodec encode input.bin -o output.brdc --key my.key \
    --preprocessing-mode reconstructive \
    --reconstructive-domain logs \
    --reconstructive-compact-transport enabled
```

Exit codes: `0` OK, `1` integrity failure, `2` key mismatch, `3` format error, `4` I/O error.

## Performance

| n_strands | Matrix size | Encode throughput (est.) | Use case |
|-----------|-------------|------------------------|----------|
| 3 | 8×8 | ~5 MB/s | Fast encoding, low security margin |
| 4 | 16×16 | ~2 MB/s | Default: good balance |
| 5 | 32×32 | ~500 KB/s | Higher security, slower |
| 6 | 64×64 | ~100 KB/s | Research use only |

Estimates assume serial encoding with `generators_per_block=32` on a single core. Parallelism scales near-linearly.

## Public Showcase Benchmark

Use the public-facing benchmark to compare on-disk footprint and reconstruction fidelity against traditional storage baselines (`gzip`, `zlib`, `lzma`) on a deterministic Pile-like markdown subset. The run uses reconstructive `lean` transport and evaluates shard scaling (`x1`, `x4`, `x16`, `x64`).

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_public_showcase.py
```

Artifacts are written to `benchmarks/public_showcase/`:

- `latest_results.json` and timestamped `showcase_*.json`
- `latest_table.csv` and timestamped `showcase_*.csv`
- `latest_report.md` and timestamped `showcase_*.md`
- `latest_plot.html` and timestamped `showcase_*.html`
- `latest_plot.png` and timestamped `showcase_*.png`

This is intended for public demos where you want to show one reproducible run with exact reconstruction checks and compact footprint comparisons. Open `benchmarks/public_showcase/latest_plot.png` directly in VS Code for a rendered chart.

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

See [docs/architecture.md](docs/architecture.md) for the full layer diagram and data flow.

## Contributing

```bash
# Install dev dependencies
pip install -e ".[dev,cli]"

# Run tests
pytest tests/

# Run core benchmarks
pytest tests/benchmarks/bench_core_throughput.py --benchmark-enable -o 'python_files=bench_*.py'

# Lint & format
ruff check src/ tests/ examples/
ruff format src/ tests/ examples/

# Type check
mypy src/braidcodec --strict
```

All PRs must pass CI (lint, typecheck, test with ≥95% coverage).

## Security Disclaimer

BraidCodec is an experimental research project demonstrating topological data encoding. It is **not** a replacement for production cryptographic systems (AES-256, etc.). The security rests on the computational hardness of inverting the Jones polynomial (#P-hard in general), but this specific instantiation has not been cryptanalyzed.

## License

MIT
