from __future__ import annotations

import contextlib
import csv
import gzip
import hashlib
import json
import lzma
import time
import urllib.parse
import urllib.request
import zlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from braidcodec import decode, encode, keygen, verify

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "benchmarks" / "public_showcase"

PILE_DATASET = "NeelNanda/pile-10k"
PILE_CONFIG = "default"
PILE_SPLIT = "train"
TARGET_SIZES_BYTES: tuple[int, ...] = (100 * 1024, 200 * 1024, 500 * 1024)
ROWS_PAGE_SIZE = 100
MAX_FETCH_ROWS = 5000
PILE_CACHE_PATH = OUTPUT_DIR / "neelnanda_pile_cache.txt"


@dataclass(slots=True)
class SizeBenchmarkRow:
    target_bytes: int
    target_label: str
    source_rows: int
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


def _fetch_pile_rows(*, offset: int, length: int) -> list[str]:
    params = urllib.parse.urlencode(
        {
            "dataset": PILE_DATASET,
            "config": PILE_CONFIG,
            "split": PILE_SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "BraidCodec-public-showcase/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    out: list[str] = []
    for item in payload.get("rows", []):
        row_obj = item.get("row", {})
        text = row_obj.get("text")
        if isinstance(text, str) and text:
            out.append(text)
    return out


def _load_or_fetch_pile_corpus(max_bytes: int) -> tuple[bytes, int]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if PILE_CACHE_PATH.exists():
        cached = PILE_CACHE_PATH.read_bytes()
        if len(cached) >= max_bytes:
            return cached[:max_bytes], -1

    chunks: list[bytes] = []
    total = 0
    offset = 0
    rows_used = 0

    while total < max_bytes and offset < MAX_FETCH_ROWS:
        rows = _fetch_pile_rows(offset=offset, length=ROWS_PAGE_SIZE)
        if not rows:
            break
        for text in rows:
            block = text.encode("utf-8", errors="ignore") + b"\n"
            chunks.append(block)
            total += len(block)
            rows_used += 1
            if total >= max_bytes:
                break
        offset += ROWS_PAGE_SIZE

    corpus = b"".join(chunks)
    if len(corpus) < max_bytes:
        raise RuntimeError(
            f"Unable to fetch enough data from {PILE_DATASET}: "
            f"got {len(corpus)} bytes, need {max_bytes}"
        )

    PILE_CACHE_PATH.write_bytes(corpus)
    return corpus[:max_bytes], rows_used


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _run_size_benchmark(
    corpus: bytes,
    *,
    target_bytes: int,
    source_rows: int,
) -> SizeBenchmarkRow:
    sample = corpus[:target_bytes]
    key = keygen(sector="TSR", n_strands=4)

    encode_start = time.perf_counter()
    stream = encode(
        sample,
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

    source_sha = hashlib.sha256(sample).hexdigest()
    decoded_sha = hashlib.sha256(decoded).hexdigest()

    return SizeBenchmarkRow(
        target_bytes=target_bytes,
        target_label=f"{target_bytes // 1024}KB",
        source_rows=source_rows,
        raw_bytes=len(sample),
        gzip_bytes=len(gzip.compress(sample, compresslevel=9)),
        zlib_bytes=len(zlib.compress(sample, level=9)),
        lzma_bytes=len(lzma.compress(sample, preset=9)),
        braid_wire_bytes=len(stream.to_bytes()),
        braid_hdf5_bytes=len(stream.to_hdf5_bytes()),
        encode_ms=round(encode_ms, 3),
        decode_ms=round(decode_ms, 3),
        verify_ms=round(verify_ms, 3),
        exact_match=decoded == sample and verification.valid and source_sha == decoded_sha,
        source_sha256=source_sha,
        decoded_sha256=decoded_sha,
    )


def _build_plot(
    rows: list[SizeBenchmarkRow],
    run_label: str,
    out_html: Path,
    out_png: Path,
) -> None:
    labels = [r.target_label for r in rows]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Storage footprint by NeelNanda Pile subset size",
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


def _write_outputs(
    rows: list[SizeBenchmarkRow],
    started_at: datetime,
    *,
    source_rows: int,
    used_cache: bool,
) -> dict[str, str]:
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
        "target_label": peak.target_label,
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
        "data_source": {
            "dataset": PILE_DATASET,
            "config": PILE_CONFIG,
            "split": PILE_SPLIT,
            "cache_path": PILE_CACHE_PATH.relative_to(ROOT).as_posix(),
            "used_cache": used_cache,
            "source_rows": source_rows,
        },
        "target_sizes_bytes": list(TARGET_SIZES_BYTES),
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
        f"- Data source: `{PILE_DATASET}` (`{PILE_CONFIG}/{PILE_SPLIT}`)",
        f"- Target sizes: `{', '.join(r.target_label for r in rows)}`",
        f"- Source rows consumed: `{source_rows}`",
        f"- Used local cache: `{used_cache}`",
        f"- All exact reconstruction matches: `{totals['all_exact']}`",
        (
            f"- Headline ({headline['target_label']}): "
            f"raw `{headline['raw_bytes']}` -> braid wire `{headline['braid_wire_bytes']}` "
            f"({headline['braid_wire_reduction_pct']:.2f}% reduction)"
        ),
        "",
        "This benchmark uses non-duplicated text sampled from NeelNanda's Pile subset "
        "and compares BraidCodec lean transport against traditional compressors while "
        "enforcing exact reconstruction quality.",
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
    max_target = max(TARGET_SIZES_BYTES)
    corpus, source_rows = _load_or_fetch_pile_corpus(max_target)
    used_cache = source_rows < 0
    if used_cache:
        source_rows = 0

    rows = [
        _run_size_benchmark(corpus, target_bytes=size, source_rows=source_rows)
        for size in TARGET_SIZES_BYTES
    ]
    outputs = _write_outputs(
        rows,
        started_at,
        source_rows=source_rows,
        used_cache=used_cache,
    )

    print("Public lean showcase benchmark complete")
    print(f"Dataset: {PILE_DATASET} ({PILE_CONFIG}/{PILE_SPLIT})")
    print(f"Target sizes: {', '.join(r.target_label for r in rows)}")
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
