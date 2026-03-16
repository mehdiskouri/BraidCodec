"""Throughput benchmarks for encode / decode pipeline.

PRD §13.2: measure encode/decode/verify throughput in MB/s,
scaling vs n_strands (3–6) and generators_per_block (8, 16, 32).
Reports bytes/second via benchmark extra_info for downstream analysis.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from braidcodec import BraidKey, EncodedStream, decode, encode, keygen

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

# Realistic sizes: 10KB baseline, 100KB typical, 1MB stress
_SIZES = {
    "10KB": 10_240,
    "100KB": 102_400,
    "1MB": 1_048_576,
}

_STRANDS: list[int] = [3, 4, 5, 6]
_GPB: list[int] = [8, 16, 32]


def _make_data(size: int) -> bytes:
    return os.urandom(size)


def _make_key(n_strands: int) -> BraidKey:
    return keygen(sector="TSR", n_strands=n_strands)


def _pre_encode(data: bytes, key: BraidKey, gpb: int) -> EncodedStream:
    return encode(data, key, generators_per_block=gpb)


# ── encode throughput ─────────────────────────────────────────────────


@pytest.mark.parametrize("size_label", list(_SIZES))
@pytest.mark.parametrize("n_strands", _STRANDS)
@pytest.mark.parametrize("gpb", _GPB)
def test_encode_throughput(
    benchmark: BenchmarkFixture,
    size_label: str,
    n_strands: int,
    gpb: int,
) -> None:
    nbytes = _SIZES[size_label]
    data = _make_data(nbytes)
    key = _make_key(n_strands)

    benchmark(encode, data, key, generators_per_block=gpb)

    # Report throughput in extra_info for JSON export
    mean_time = benchmark.stats.stats.mean if hasattr(benchmark, "stats") else None
    if mean_time and mean_time > 0:
        benchmark.extra_info["throughput_MBps"] = round(nbytes / mean_time / 1e6, 3)
    benchmark.extra_info["input_bytes"] = nbytes
    benchmark.extra_info["n_strands"] = n_strands
    benchmark.extra_info["gpb"] = gpb


# ── decode throughput (no verify) ─────────────────────────────────────


@pytest.mark.parametrize("size_label", list(_SIZES))
@pytest.mark.parametrize("n_strands", _STRANDS)
@pytest.mark.parametrize("gpb", _GPB)
def test_decode_throughput(
    benchmark: BenchmarkFixture,
    size_label: str,
    n_strands: int,
    gpb: int,
) -> None:
    nbytes = _SIZES[size_label]
    data = _make_data(nbytes)
    key = _make_key(n_strands)
    stream = _pre_encode(data, key, gpb)

    benchmark(decode, stream, key, verify=False)

    mean_time = benchmark.stats.stats.mean if hasattr(benchmark, "stats") else None
    if mean_time and mean_time > 0:
        benchmark.extra_info["throughput_MBps"] = round(nbytes / mean_time / 1e6, 3)
    benchmark.extra_info["input_bytes"] = nbytes


# ── decode + verify throughput ────────────────────────────────────────


@pytest.mark.parametrize("size_label", list(_SIZES))
@pytest.mark.parametrize("n_strands", _STRANDS)
@pytest.mark.parametrize("gpb", _GPB)
def test_decode_verified_throughput(
    benchmark: BenchmarkFixture,
    size_label: str,
    n_strands: int,
    gpb: int,
) -> None:
    nbytes = _SIZES[size_label]
    data = _make_data(nbytes)
    key = _make_key(n_strands)
    stream = _pre_encode(data, key, gpb)

    benchmark(decode, stream, key, verify=True)

    mean_time = benchmark.stats.stats.mean if hasattr(benchmark, "stats") else None
    if mean_time and mean_time > 0:
        benchmark.extra_info["throughput_MBps"] = round(nbytes / mean_time / 1e6, 3)
    benchmark.extra_info["input_bytes"] = nbytes
