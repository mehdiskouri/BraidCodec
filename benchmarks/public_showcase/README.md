# Public Showcase Benchmark

This folder contains a reproducible benchmark intended for public demonstrations.

## What it does

- Samples a deterministic subset of text files from:
  - `README.md`
  - `docs/*.md`
  - `AGENT/PHASES/*.md`
- Caps each sampled file to `64 KiB` to keep runs deterministic.
- Builds a base corpus and benchmarks shard scaling with repeat factors: `x1`, `x4`, `x16`, `x64`.
- Encodes each shard with BraidCodec reconstructive lean transport.
- Reconstructs and verifies each shard for exactness.
- Compares storage footprint with traditional baselines:
  - raw bytes
  - `gzip -9`
  - `zlib -9`
  - `lzma -9`
  - BraidCodec wire and HDF5 containers

## Run

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_public_showcase.py
```

## Generated artifacts

Each run writes both timestamped and latest copies:

- JSON summary: `showcase_YYYYMMDD_HHMMSS.json`, `latest_results.json`
- CSV table: `showcase_YYYYMMDD_HHMMSS.csv`, `latest_table.csv`
- Markdown summary: `showcase_YYYYMMDD_HHMMSS.md`, `latest_report.md`
- Plotly chart HTML: `showcase_YYYYMMDD_HHMMSS.html`, `latest_plot.html`
- Static plot PNG: `showcase_YYYYMMDD_HHMMSS.png`, `latest_plot.png`

## Interpretation notes

- `exact_match` must stay `true` for every shard.
- `braid_wire_vs_raw` below `1.0` means the wire format is smaller than raw bytes.
- This benchmark is a storage+fidelity showcase, not a full throughput stress test.
