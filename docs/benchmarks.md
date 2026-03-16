# Benchmarks

## Running Benchmarks Locally

```bash
# Run all benchmark suites (bench_*.py files need the python_files override)
pytest tests/benchmarks/ --benchmark-enable -o 'python_files=bench_*.py' -v

# Run the lightweight core-throughput suite
pytest tests/benchmarks/bench_core_throughput.py --benchmark-enable -o 'python_files=bench_*.py'

# Run the stress-regime suite (pathological combinations)
pytest tests/benchmarks/bench_stress_regime.py --benchmark-enable -o 'python_files=bench_*.py'

# Run a quick subset (e.g., 1 KB / TSR / 3 strands)
pytest tests/benchmarks/ --benchmark-enable -o 'python_files=bench_*.py' -k '1KB and TSR and 3'

# Save results to JSON
pytest tests/benchmarks/ --benchmark-enable -o 'python_files=bench_*.py' --benchmark-json=bench.json

# Compare against a saved baseline
pytest tests/benchmarks/ --benchmark-enable -o 'python_files=bench_*.py' --benchmark-compare=bench.json
```

Benchmarks are **excluded** from the normal test suite via `--ignore=tests/benchmarks` in `pyproject.toml`.
The `-o 'python_files=bench_*.py'` override is required because benchmark files use the `bench_` prefix instead of `test_`.

## Parametrization Dimensions

### Core Throughput Benchmarks (`bench_core_throughput.py`)

| Dimension | Values |
|-----------|--------|
| Input size | 10 KB, 100 KB |
| n_strands | 3, 4, 5 |
| preprocessing_mode | legacy, topology |
| generators_per_block | 16 |

Tests: `test_core_encode_throughput`

### Stress Regime Benchmarks (`bench_stress_regime.py`)

| Dimension | Values |
|-----------|--------|
| Input size | 10 KB, 100 KB |
| n_strands | 6 |
| preprocessing_mode | legacy, topology |
| generators_per_block | 16 |

Tests: `test_stress_encode_regime`

### Compression Benchmarks (`bench_compression.py`)

| Dimension | Values |
|-----------|--------|
| Level | 0, 1, 2 |
| Input size | 1 KB, 10 KB |
| Entropy | zeros, random, mixed |

Tests: `test_compress_throughput`, `test_compression_ratio`

### Integrity Benchmarks (`bench_integrity.py`)

| Dimension | Values |
|-----------|--------|
| Block count | 1, 5, 10, 50 |
| Fermion check | False, True |

Tests: `test_verify_throughput`, `test_writhe_only`

## Performance Expectations

| n_strands | Matrix size | Encode throughput (est.) | Use case |
|-----------|-------------|------------------------|----------|
| 3 | 8×8 | ~5 MB/s | Fast encoding, low security margin |
| 4 | 16×16 | ~2 MB/s | Default: good balance |
| 5 | 32×32 | ~500 KB/s | Higher security, slower |
| 6 | 64×64 | ~100 KB/s | Research use only |

Estimates assume serial encoding with `generators_per_block=32` on a single core.

## Compression Ratio Expectations

| Level | Zeros (low entropy) | Random (high entropy) | Mixed |
|-------|--------------------|-----------------------|-------|
| 0 | 0.85–0.95 | 0.95–1.00 | 0.90–0.98 |
| 1 | 0.70–0.85 | 0.85–0.95 | 0.75–0.90 |
| 2 | 0.60–0.80 | 0.80–0.90 | 0.65–0.85 |

Ratio = compressed generators / original generators. Lower is better.

## CI Dashboard

The weekly benchmark CI (Mondays at 6:00 UTC) runs two paths:

- `core-throughput` publishes stable trend data to GitHub Pages.
- `stress-regime` runs in non-blocking reporting mode and emits threshold alerts.

Core trend dashboard:

```
https://<owner>.github.io/BraidCodec/benchmarks/
```
