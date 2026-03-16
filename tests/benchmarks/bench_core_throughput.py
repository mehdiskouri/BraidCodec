"""Lightweight core throughput benchmarks (no stress cases).

Phase 6 core suite:
- Excludes known stress regimes (e.g. n_strands=6 and 1MB cases).
- Compares topology-first pipeline against legacy compatibility mode.
- Reports delta versus recorded baseline means from AGENT/benchmark_outputs.md.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import pytest

from braidcodec import encode, keygen

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

# Lightweight sizes only.
_SIZES = {
    "10KB": 10_240,
    "100KB": 102_400,
}

# Core non-stress strands.
_STRANDS = [3, 4, 5]

# Keep one stable gpb matching historical baseline cases.
_GPB = 16

_MODES = ["legacy", "topology"]
_LEGACY_WIRE_SIZE_BY_CASE: dict[tuple[int, int], int] = {}

# Recorded baseline means (microseconds) from AGENT/benchmark_outputs.md.
_BASELINE_US: dict[tuple[int, int, str], float] = {
    (3, 10_240, "legacy"): 590_472.7606,
    (4, 10_240, "legacy"): 511_668.2494,
    (5, 10_240, "legacy"): 429_371.6490,
    (3, 102_400, "legacy"): 4_906_984.6748,
    (4, 102_400, "legacy"): 4_046_882.8824,
    (5, 102_400, "legacy"): 3_335_087.6532,
}


def _make_data(size: int) -> bytes:
    return os.urandom(size)


@pytest.mark.parametrize("size_label", list(_SIZES))
@pytest.mark.parametrize("n_strands", _STRANDS)
@pytest.mark.parametrize("mode", _MODES)
def test_core_encode_throughput(
    benchmark: BenchmarkFixture,
    size_label: str,
    n_strands: int,
    mode: str,
) -> None:
    nbytes = _SIZES[size_label]
    data = _make_data(nbytes)
    key = keygen(sector="TSR", n_strands=n_strands)

    benchmark(
        encode,
        data,
        key,
        generators_per_block=_GPB,
        preprocessing_mode=mode,
    )
    start = time.perf_counter()
    probe_stream = encode(
        data,
        key,
        generators_per_block=_GPB,
        preprocessing_mode=mode,
    )
    mean_time = max(time.perf_counter() - start, 1e-12)
    extra_info: dict[str, object] = benchmark.extra_info  # type: ignore[assignment]
    if mean_time and mean_time > 0:
        extra_info["throughput_MBps"] = round(nbytes / mean_time / 1e6, 6)

    baseline_us = _BASELINE_US.get((n_strands, nbytes, mode))
    if baseline_us is not None:
        current_us = mean_time * 1e6
        delta_pct = ((current_us - baseline_us) / baseline_us) * 100.0
        extra_info["baseline_mean_us"] = round(baseline_us, 4)
        extra_info["current_mean_us"] = round(current_us, 4)
        extra_info["delta_vs_baseline_pct"] = round(delta_pct, 2)

    extra_info["input_bytes"] = nbytes
    extra_info["n_strands"] = n_strands
    extra_info["gpb"] = _GPB
    extra_info["preprocessing_mode"] = mode
    extra_info["regime_label"] = "core-throughput"

    stream_meta = probe_stream.metadata
    for key_name in (
        "timing_preprocess_s",
        "timing_layer_order_s",
        "timing_encode_core_s",
        "timing_total_s",
        "execution_mode",
        "batch_count",
        "batch_size",
    ):
        if key_name in stream_meta:
            extra_info[key_name] = stream_meta[key_name]

    t0 = time.perf_counter()
    wire = probe_stream.to_bytes()
    serialize_elapsed = time.perf_counter() - t0
    wire_size = len(wire)
    extra_info["raw_input_bytes"] = nbytes
    extra_info["encoded_wire_bytes"] = wire_size
    extra_info["encoded_minus_raw_bytes"] = wire_size - nbytes
    extra_info["wire_expansion_ratio"] = round(wire_size / nbytes, 6)
    extra_info["generator_count"] = sum(len(b.generators) for b in probe_stream.blocks)
    extra_info["decode_generator_count"] = sum(
        len(b.decode_generators) if b.decode_generators is not None else 0
        for b in probe_stream.blocks
    )

    case_key = (n_strands, nbytes)
    if mode == "legacy":
        _LEGACY_WIRE_SIZE_BY_CASE[case_key] = wire_size
    else:
        legacy_wire = _LEGACY_WIRE_SIZE_BY_CASE.get(case_key)
        if legacy_wire is not None and legacy_wire > 0:
            extra_info["legacy_encoded_wire_bytes"] = legacy_wire
            size_delta_pct = ((wire_size - legacy_wire) / legacy_wire) * 100.0
            extra_info["encoded_size_delta_vs_legacy_pct"] = round(size_delta_pct, 2)

    extra_info["timing_serialize_s"] = round(serialize_elapsed, 6)
