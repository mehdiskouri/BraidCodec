from __future__ import annotations

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
MAX_BYTES_PER_FILE = 32 * 1024
SOURCE_GLOBS: tuple[str, ...] = (
    "README.md",
    "docs/*.md",
    "AGENT/PHASES/*.md",
)


@dataclass(slots=True)
class FileBenchmarkRow:
    source: str
    sampled_bytes: int
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


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _run_file_benchmark(path: Path) -> FileBenchmarkRow:
    source = path.relative_to(ROOT).as_posix()
    raw = path.read_bytes()
    sampled = _sample_bytes(raw)
    key = keygen(sector="TSR", n_strands=4)

    encode_start = time.perf_counter()
    stream = encode(
        sampled,
        key,
        generators_per_block=8,
        preprocessing_mode="reconstructive",
        reconstructive_domain="text",
        reconstructive_discovery="enabled",
        reconstructive_compact_transport="enabled",
        reconstructive_compact_audit_bundle=True,
    )
    encode_ms = (time.perf_counter() - encode_start) * 1000.0

    decode_start = time.perf_counter()
    decoded = decode(stream, key, verify=False)
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    verify_start = time.perf_counter()
    verification = verify(stream, key)
    verify_ms = (time.perf_counter() - verify_start) * 1000.0

    source_sha = hashlib.sha256(sampled).hexdigest()
    decoded_sha = hashlib.sha256(decoded).hexdigest()

    return FileBenchmarkRow(
        source=source,
        sampled_bytes=len(sampled),
        raw_bytes=len(sampled),
        gzip_bytes=len(gzip.compress(sampled, compresslevel=9)),
        zlib_bytes=len(zlib.compress(sampled, level=9)),
        lzma_bytes=len(lzma.compress(sampled, preset=9)),
        braid_wire_bytes=len(stream.to_bytes()),
        braid_hdf5_bytes=len(stream.to_hdf5_bytes()),
        encode_ms=round(encode_ms, 3),
        decode_ms=round(decode_ms, 3),
        verify_ms=round(verify_ms, 3),
        exact_match=decoded == sampled and verification.valid and source_sha == decoded_sha,
        source_sha256=source_sha,
        decoded_sha256=decoded_sha,
    )


def _build_plot(rows: list[FileBenchmarkRow], run_label: str, out_html: Path) -> None:
    methods = [
        "raw",
        "gzip",
        "zlib",
        "lzma",
        "braid_wire",
        "braid_hdf5",
    ]
    totals = {
        "raw": sum(r.raw_bytes for r in rows),
        "gzip": sum(r.gzip_bytes for r in rows),
        "zlib": sum(r.zlib_bytes for r in rows),
        "lzma": sum(r.lzma_bytes for r in rows),
        "braid_wire": sum(r.braid_wire_bytes for r in rows),
        "braid_hdf5": sum(r.braid_hdf5_bytes for r in rows),
    }

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Total storage footprint across sampled corpus",
            "Per-file storage footprint (raw vs Braid wire)",
        ),
        vertical_spacing=0.14,
    )

    fig.add_trace(
        go.Bar(
            x=methods,
            y=[totals[m] for m in methods],
            text=[str(totals[m]) for m in methods],
            textposition="outside",
            marker_color=["#7f8c8d", "#2ecc71", "#27ae60", "#16a085", "#f39c12", "#d35400"],
            name="totals",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    sources = [r.source for r in rows]
    fig.add_trace(
        go.Bar(
            x=sources,
            y=[r.raw_bytes for r in rows],
            name="raw",
            marker_color="#95a5a6",
            opacity=0.75,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=sources,
            y=[r.braid_wire_bytes for r in rows],
            name="braid_wire",
            marker_color="#e67e22",
            opacity=0.85,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"BraidCodec Public Showcase Benchmark ({run_label})",
        barmode="group",
        height=980,
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 80, "b": 140},
    )
    fig.update_yaxes(title_text="bytes", row=1, col=1)
    fig.update_yaxes(title_text="bytes", row=2, col=1)
    fig.update_xaxes(tickangle=25, row=2, col=1)

    fig.write_html(str(out_html), include_plotlyjs="cdn")


def _write_outputs(rows: list[FileBenchmarkRow], started_at: datetime) -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    run_label = started_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    rows_dict = [asdict(r) for r in rows]
    totals = {
        "files": len(rows),
        "raw_bytes": sum(r.raw_bytes for r in rows),
        "gzip_bytes": sum(r.gzip_bytes for r in rows),
        "zlib_bytes": sum(r.zlib_bytes for r in rows),
        "lzma_bytes": sum(r.lzma_bytes for r in rows),
        "braid_wire_bytes": sum(r.braid_wire_bytes for r in rows),
        "braid_hdf5_bytes": sum(r.braid_hdf5_bytes for r in rows),
        "exact_matches": sum(1 for r in rows if r.exact_match),
        "all_exact": all(r.exact_match for r in rows),
        "mean_encode_ms": round(sum(r.encode_ms for r in rows) / max(len(rows), 1), 3),
        "mean_decode_ms": round(sum(r.decode_ms for r in rows) / max(len(rows), 1), 3),
        "mean_verify_ms": round(sum(r.verify_ms for r in rows) / max(len(rows), 1), 3),
    }
    ratios = {
        "gzip_vs_raw": round(_ratio(totals["gzip_bytes"], totals["raw_bytes"]), 4),
        "zlib_vs_raw": round(_ratio(totals["zlib_bytes"], totals["raw_bytes"]), 4),
        "lzma_vs_raw": round(_ratio(totals["lzma_bytes"], totals["raw_bytes"]), 4),
        "braid_wire_vs_raw": round(_ratio(totals["braid_wire_bytes"], totals["raw_bytes"]), 4),
        "braid_hdf5_vs_raw": round(_ratio(totals["braid_hdf5_bytes"], totals["raw_bytes"]), 4),
    }

    payload = {
        "run_id": run_id,
        "run_label": run_label,
        "subset_source_globs": list(SOURCE_GLOBS),
        "max_files": MAX_FILES,
        "max_bytes_per_file": MAX_BYTES_PER_FILE,
        "totals": totals,
        "ratios": ratios,
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

    json_text = json.dumps(payload, indent=2, sort_keys=True)
    versioned_json.write_text(json_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")

    fieldnames = list(rows_dict[0].keys()) if rows_dict else ["source", "sampled_bytes"]
    with versioned_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_dict)
    with latest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_dict)

    md_lines = [
        "# Public Showcase Benchmark",
        "",
        f"- Run: `{run_id}`",
        f"- Files sampled: `{totals['files']}`",
        f"- All exact reconstruction matches: `{totals['all_exact']}`",
        f"- Total raw bytes: `{totals['raw_bytes']}`",
        f"- Braid wire vs raw: `{ratios['braid_wire_vs_raw']:.4f}`",
        f"- Braid hdf5 vs raw: `{ratios['braid_hdf5_vs_raw']:.4f}`",
        f"- gzip vs raw: `{ratios['gzip_vs_raw']:.4f}`",
        f"- zlib vs raw: `{ratios['zlib_vs_raw']:.4f}`",
        f"- lzma vs raw: `{ratios['lzma_vs_raw']:.4f}`",
        "",
        (
            "This benchmark is intended for public demos as an on-disk footprint "
            "and reconstruction-fidelity comparison across a small, deterministic "
            "Pile-like subset."
        ),
        "",
        "## Artifacts",
        "",
        f"- JSON: `{versioned_json.relative_to(ROOT).as_posix()}`",
        f"- CSV: `{versioned_csv.relative_to(ROOT).as_posix()}`",
        f"- Plot: `{versioned_html.relative_to(ROOT).as_posix()}`",
        "",
    ]
    md_text = "\n".join(md_lines)
    versioned_md.write_text(md_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    _build_plot(rows=rows, run_label=run_label, out_html=versioned_html)
    _build_plot(rows=rows, run_label=run_label, out_html=latest_html)

    return {
        "json": versioned_json.relative_to(ROOT).as_posix(),
        "csv": versioned_csv.relative_to(ROOT).as_posix(),
        "markdown": versioned_md.relative_to(ROOT).as_posix(),
        "html": versioned_html.relative_to(ROOT).as_posix(),
        "latest_json": latest_json.relative_to(ROOT).as_posix(),
        "latest_csv": latest_csv.relative_to(ROOT).as_posix(),
        "latest_markdown": latest_md.relative_to(ROOT).as_posix(),
        "latest_html": latest_html.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    started_at = datetime.now(UTC)
    subset = _collect_subset()
    if not subset:
        raise RuntimeError("No source files found for showcase subset")

    rows = [_run_file_benchmark(path) for path in subset]
    outputs = _write_outputs(rows, started_at)

    print("Public showcase benchmark complete")
    print(f"Files sampled: {len(rows)}")
    print(f"All exact matches: {all(r.exact_match for r in rows)}")
    for key in (
        "json",
        "csv",
        "markdown",
        "html",
        "latest_json",
        "latest_csv",
        "latest_markdown",
        "latest_html",
    ):
        print(f"{key}: {outputs[key]}")


if __name__ == "__main__":
    main()
