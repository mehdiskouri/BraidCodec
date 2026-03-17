# Public Showcase Benchmark

This folder contains a reproducible benchmark intended for public demonstrations.

## What it does

- Pulls text from `NeelNanda/pile-10k` (`default/train`) via Hugging Face datasets server.
- Builds one non-duplicated corpus and benchmarks contiguous size slices at:
  - `100KB`
  - `200KB`
  - `500KB`
- Encodes each size slice with BraidCodec reconstructive lean transport.
- Reconstructs and verifies each size slice for exactness.
- Compares storage footprint with traditional baselines:
  - raw bytes
  - `gzip -9`
  - `zlib -9`
  - `lzma -9`
  - BraidCodec wire and HDF5 containers

The script caches fetched corpus text in `benchmarks/public_showcase/neelnanda_pile_cache.txt` to make repeated runs deterministic and faster.

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

- `exact_match` must stay `true` for every size point.
- `braid_wire_vs_raw` below `1.0` means the wire format is smaller than raw bytes.
- This benchmark is a storage+fidelity showcase, not a full throughput stress test.
