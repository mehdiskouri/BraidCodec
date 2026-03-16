"""Stress-regime topology benchmarks.

Phase 6 stress suite:
- Focuses on known pathological combinations.
- Runs in reporting mode in CI (non-blocking trend + threshold alert).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from braidcodec import encode, keygen

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

_SIZES = {
    "10KB": 10_240,
    "100KB": 102_400,
}

_MODES = ["legacy", "topology"]
_TARGET_MAX_SECONDS: dict[tuple[int, int], float] = {
    (6, 10_240): 8.0,
    (6, 102_400): 80.0,
}


def _make_data(size: int) -> bytes:
    return os.urandom(size)


@pytest.mark.parametrize("size_label", list(_SIZES))
@pytest.mark.parametrize("mode", _MODES)
def test_stress_encode_regime(
    benchmark: BenchmarkFixture,
    size_label: str,
    mode: str,
) -> None:
    n_strands = 6
    gpb = 16
    nbytes = _SIZES[size_label]
    key = keygen(sector="TSR", n_strands=n_strands)
    data = _make_data(nbytes)

    benchmark(
        encode,
        data,
        key,
        generators_per_block=gpb,
        preprocessing_mode=mode,
    )

    probe_stream = encode(
        data,
        key,
        generators_per_block=gpb,
        preprocessing_mode=mode,
    )

    stats_obj: Any = benchmark.stats
    mean_time = float(stats_obj.stats.mean)
    extra_info: dict[str, object] = benchmark.extra_info  # type: ignore[assignment]
    if mean_time and mean_time > 0:
        extra_info["throughput_MBps"] = round(nbytes / mean_time / 1e6, 6)
        extra_info["mean_seconds"] = round(mean_time, 6)
    wire_size = len(probe_stream.to_bytes())
    extra_info["raw_input_bytes"] = nbytes
    extra_info["encoded_wire_bytes"] = wire_size
    extra_info["encoded_minus_raw_bytes"] = wire_size - nbytes
    extra_info["wire_expansion_ratio"] = round(wire_size / nbytes, 6)
    extra_info["input_bytes"] = nbytes
    extra_info["n_strands"] = n_strands
    extra_info["gpb"] = gpb
    extra_info["preprocessing_mode"] = mode
    extra_info["regime_label"] = "stress-regime"

    target = _TARGET_MAX_SECONDS[(n_strands, nbytes)]
    extra_info["target_max_seconds"] = target
