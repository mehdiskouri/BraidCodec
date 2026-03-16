#!/usr/bin/env python3
"""Emit stress-regime threshold alerts from pytest-benchmark JSON.

This script is intentionally non-blocking for subbranch CI reporting mode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_stress_thresholds.py <benchmark-json>")
        return 0

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"::warning::stress benchmark JSON not found: {path}")
        return 0

    data = json.loads(path.read_text(encoding="utf-8"))
    benches = data.get("benchmarks", [])

    alert_count = 0
    for bench in benches:
        extra = bench.get("extra_info", {}) or {}
        regime = extra.get("regime_label")
        if regime != "stress-regime":
            continue

        target = extra.get("target_max_seconds")
        mean_seconds = extra.get("mean_seconds")
        mode = extra.get("preprocessing_mode", "unknown")
        n_strands = extra.get("n_strands", "?")
        nbytes = extra.get("input_bytes", "?")
        if target is None or mean_seconds is None:
            continue

        try:
            target_f = float(target)
            mean_f = float(mean_seconds)
        except (TypeError, ValueError):
            continue

        if mean_f > target_f:
            alert_count += 1
            print(
                "::warning::stress threshold exceeded "
                f"(mode={mode}, n_strands={n_strands}, input_bytes={nbytes}, "
                f"mean={mean_f:.3f}s, target={target_f:.3f}s)"
            )

    if alert_count == 0:
        print("Stress thresholds: no alerts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
