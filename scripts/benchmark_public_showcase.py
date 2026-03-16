from __future__ import annotations

import contextlib
import csv
import gzip
import hashlib
import json
import lzma
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from braidcodec import decode, encode, keygen, verify

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "benchmarks" / "public_showcase"

MAX_FILES = 12
MAX_BYTES_PER_FILE = 64 * 1024
SOURCE_GLOBS: tuple[str, ...] = (
    "README.md",
    "docs/*.md",
    "AGENT/PHASES/*.md",
)
REPEAT_FACTORS: tuple[int, ...] = (1, 4, 16, 64)


@dataclass(slots=True)
class ShardBenchmarkRow:
    repeat_factor: int
    source_files: int
    raw_bytes: int
    gzip_bytes: int
    zlib_bytes: int
    lzma_bytes: int
    braid_wire_bytes: int
    braid_hdf5_bytes: int
    encode_ms: float
    decode_ms: float
    verify_ms: float
    exact_match: bool
    source_sha256: str
    decoded_sha256: str


def _collect_subset() -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in SOURCE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
            if len(found) >= MAX_FILES:
                return found
    return found


def _sample_bytes(data: bytes) -> bytes:
    if len(data) <= MAX_BYTES_PER_FILE:
        return data
    return data[:MAX_BYTES_PER_FILE]


def _build_base_corpus(paths: list[Path]) -> bytes:
    chunks: list[bytes] = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        payload = _sample_bytes(path.read_bytes())
        header = f"\n\n<<<SOURCE:{relative}>>>\n".encode()
        chunks.append(header + payload)
    return b"".join(chunks)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _run_shard_benchmark(
    base_corpus: bytes,
    *,
    repeat_factor: int,
    source_files: int,
) -> ShardBenchmarkRow:
    shard = base_corpus * repeat_factor
    key = keygen(sector="TSR", n_strands=4)

    encode_start = time.perf_counter()
    stream = encode(
        shard,
        key,
        generators_per_block=8,
        preprocessing_mode="reconstructive",
        reconstructive_domain="text",
        reconstructive_discovery="enabled",
        reconstructive_compact_transport="lean",
        reconstructive_compact_audit_bundle=False,
    )
    encode_ms = (time.perf_counter() - encode_start) * 1000.0

    decode_start = time.perf_counter()
    decoded = decode(stream, key, verify=False)
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    verify_start = time.perf_counter()
    verification = verify(stream, key)
    verify_ms = (time.perf_counter() - verify_start) * 1000.0

    source_sha = hashlib.sha256(shard).hexdigest()
    decoded_sha = hashlib.sha256(decoded).hexdigest()

    return ShardBenchmarkRow(
        repeat_factor=repeat_factor,
        source_files=source_files,
        raw_bytes=len(shard),
        gzip_bytes=len(gzip.compress(shard, compresslevel=9)),
        zlib_bytes=len(zlib.compress(shard, level=9)),
        lzma_bytes=len(lzma.compress(shard, preset=9)),
        braid_wire_bytes=len(stream.to_bytes()),
        braid_hdf5_bytes=len(stream.to_hdf5_bytes()),
        encode_ms=round(encode_ms, 3),
        decode_ms=round(decode_ms, 3),
        verify_ms=round(verify_ms, 3),
        exact_match=decoded == shard and verification.valid and source_sha == decoded_sha,
        source_sha256=source_sha,
        decoded_sha256=decoded_sha,
    )


def _build_plot(
    rows: list[ShardBenchmarkRow],
    run_label: str,
    out_html: Path,
    out_png: Path,
) -> None:
    labels = [f"x{r.repeat_factor}" for r in rows]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Storage footprint by shard size",
            "Compression ratio vs raw bytes",
        ),
        vertical_spacing=0.14,
    )

    bar_series = [
        ("raw", [r.raw_bytes for r in rows], "#7f8c8d"),
        ("gzip", [r.gzip_bytes for r in rows], "#27ae60"),
        ("zlib", [r.zlib_bytes for r in rows], "#16a085"),
        ("lzma", [r.lzma_bytes for r in rows], "#2ecc71"),
        ("braid_wire_lean", [r.braid_wire_bytes for r in rows], "#e67e22"),
    ]
    for name, values, color in bar_series:
        fig.add_trace(
            go.Bar(x=labels, y=values, name=name, marker_color=color),
            row=1,
            col=1,
        )

    line_series = [
        ("gzip/raw", [_ratio(r.gzip_bytes, r.raw_bytes) for r in rows], "#27ae60"),
        ("zlib/raw", [_ratio(r.zlib_bytes, r.raw_bytes) for r in rows], "#16a085"),
        ("lzma/raw", [_ratio(r.lzma_bytes, r.raw_bytes) for r in rows], "#2ecc71"),
        (
            "braid_wire_lean/raw",
            [_ratio(r.braid_wire_bytes, r.raw_bytes) for r in rows],
            "#e67e22",
        ),
    ]
    for name, values, color in line_series:
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=values,
                mode="lines+markers",
                name=name,
                line={"width": 3, "color": color},
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=f"BraidCodec Lean Public Showcase ({run_label})",
        barmode="group",
        height=980,
        template="plotly_white",
        margin={"l": 70, "r": 30, "t": 80, "b": 70},
    )
    fig.update_yaxes(title_text="bytes", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=2, col=1)

    fig.write_html(str(out_html), include_plotlyjs="cdn")

    # Keep HTML output even if static image export backend is unavailable.
    with contextlib.suppress(Exception):
        fig.write_image(str(out_png), scale=2)


def _write_outputs(rows: list[ShardBenchmarkRow], started_at: datetime) -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    run_label = started_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    rows_dict = [asdict(r) for r in rows]
    totals = {
        "rows": len(rows),
        "all_exact": all(r.exact_match for r in rows),
        "exact_matches": sum(1 for r in rows if r.exact_match),
    }

    peak = rows[-1]
    headline = {
        "repeat_factor": peak.repeat_factor,
        "raw_bytes": peak.raw_bytes,
        "braid_wire_bytes": peak.braid_wire_bytes,
        "braid_wire_vs_raw": round(_ratio(peak.braid_wire_bytes, peak.raw_bytes), 4),
        "braid_wire_reduction_pct": round(
            (1.0 - _ratio(peak.braid_wire_bytes, peak.raw_bytes)) * 100.0,
            2,
        ),
    }

    payload = {
        "run_id": run_id,
        "run_label": run_label,
        "subset_source_globs": list(SOURCE_GLOBS),
        "max_files": MAX_FILES,
        "max_bytes_per_file": MAX_BYTES_PER_FILE,
        "repeat_factors": list(REPEAT_FACTORS),
        "totals": totals,
        "headline": headline,
        "rows": rows_dict,
    }

    versioned_json = OUTPUT_DIR / f"showcase_{run_id}.json"
    latest_json = OUTPUT_DIR / "latest_results.json"
    versioned_csv = OUTPUT_DIR / f"showcase_{run_id}.csv"
    latest_csv = OUTPUT_DIR / "latest_table.csv"
    versioned_md = OUTPUT_DIR / f"showcase_{run_id}.md"
    latest_md = OUTPUT_DIR / "latest_report.md"
    versioned_html = OUTPUT_DIR / f"showcase_{run_id}.html"
    latest_html = OUTPUT_DIR / "latest_plot.html"
    versioned_png = OUTPUT_DIR / f"showcase_{run_id}.png"
    latest_png = OUTPUT_DIR / "latest_plot.png"

    json_text = json.dumps(payload, indent=2, sort_keys=True)
    versioned_json.write_text(json_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")

    fieldnames = list(rows_dict[0].keys()) if rows_dict else ["repeat_factor", "raw_bytes"]
    with versioned_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_dict)
    with latest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_dict)

    md_lines = [
        "# Public Showcase Benchmark (Lean Path)",
        "",
        f"- Run: `{run_id}`",
        f"- Repeat factors: `{', '.join(f'x{r.repeat_factor}' for r in rows)}`",
        f"- All exact reconstruction matches: `{totals['all_exact']}`",
        (
            f"- Headline (x{headline['repeat_factor']}): "
            f"raw `{headline['raw_bytes']}` -> braid wire `{headline['braid_wire_bytes']}` "
            f"({headline['braid_wire_reduction_pct']:.2f}% reduction)"
        ),
        "",
        "This benchmark uses a deterministic Pile-like subset of local markdown corpus, "
        "builds progressively larger text shards, and compares BraidCodec lean transport "
        "against traditional compressors while enforcing exact reconstruction quality.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{versioned_json.relative_to(ROOT).as_posix()}`",
        f"- CSV: `{versioned_csv.relative_to(ROOT).as_posix()}`",
        f"- Plot HTML: `{versioned_html.relative_to(ROOT).as_posix()}`",
        f"- Plot PNG: `{versioned_png.relative_to(ROOT).as_posix()}`",
        "",
        "## Plot",
        "",
        "![Lean showcase plot](latest_plot.png)",
    ]
    md_text = "\n".join(md_lines)
    versioned_md.write_text(md_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    _build_plot(rows=rows, run_label=run_label, out_html=versioned_html, out_png=versioned_png)
    _build_plot(rows=rows, run_label=run_label, out_html=latest_html, out_png=latest_png)

    return {
        "json": versioned_json.relative_to(ROOT).as_posix(),
        "csv": versioned_csv.relative_to(ROOT).as_posix(),
        "markdown": versioned_md.relative_to(ROOT).as_posix(),
        "html": versioned_html.relative_to(ROOT).as_posix(),
        "png": versioned_png.relative_to(ROOT).as_posix(),
        "latest_json": latest_json.relative_to(ROOT).as_posix(),
        "latest_csv": latest_csv.relative_to(ROOT).as_posix(),
        "latest_markdown": latest_md.relative_to(ROOT).as_posix(),
        "latest_html": latest_html.relative_to(ROOT).as_posix(),
        "latest_png": latest_png.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    started_at = datetime.now(UTC)
    subset = _collect_subset()
    if not subset:
        raise RuntimeError("No source files found for showcase subset")

    base = _build_base_corpus(subset)
    rows = [
        _run_shard_benchmark(base, repeat_factor=repeat, source_files=len(subset))
        for repeat in REPEAT_FACTORS
    ]
    outputs = _write_outputs(rows, started_at)

    print("Public lean showcase benchmark complete")
    print(f"Files sampled: {len(subset)}")
    print(f"All exact matches: {all(r.exact_match for r in rows)}")
    print(
        "Headline reduction (%):",
        round((1.0 - _ratio(rows[-1].braid_wire_bytes, rows[-1].raw_bytes)) * 100.0, 2),
    )
    for key in (
        "json",
        "csv",
        "markdown",
        "html",
        "png",
        "latest_json",
        "latest_csv",
        "latest_markdown",
        "latest_html",
        "latest_png",
    ):
        print(f"{key}: {outputs[key]}")


if __name__ == "__main__":
    main()
