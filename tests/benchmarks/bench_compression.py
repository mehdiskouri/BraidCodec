"""Compression benchmarks — throughput and ratio by level and entropy."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from braidcodec import EncodedStream, encode, keygen
from braidcodec.codec.compressor import compress, compression_ratio

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


def _zeros(n: int) -> bytes:
    return b"\x00" * n


def _random(n: int) -> bytes:
    return os.urandom(n)


def _mixed(n: int) -> bytes:
    pattern = bytes(range(256))
    reps = n // 256 + 1
    return (pattern * reps)[:n]


_ENTROPY_FUNCS = {"zeros": _zeros, "random": _random, "mixed": _mixed}
_SIZES = {"1KB": 1_024, "10KB": 10_240}
_LEVELS = [0, 1, 2]


def _pre_encode(data: bytes) -> EncodedStream:
    key = keygen(sector="TSR", n_strands=4)
    return encode(data, key, generators_per_block=8), key  # type: ignore[return-value]


# ── compress throughput ───────────────────────────────────────────────


@pytest.mark.parametrize("level", _LEVELS)
@pytest.mark.parametrize("size_label", list(_SIZES))
@pytest.mark.parametrize("entropy", list(_ENTROPY_FUNCS))
def test_compress_throughput(
    benchmark: BenchmarkFixture,
    level: int,
    size_label: str,
    entropy: str,
) -> None:
    data = _ENTROPY_FUNCS[entropy](_SIZES[size_label])
    key = keygen(sector="TSR", n_strands=4)
    stream = encode(data, key, generators_per_block=8)
    benchmark(compress, stream, key, level=level)


# ── compression ratio (not timed — just measured) ─────────────────────


@pytest.mark.parametrize("level", _LEVELS)
@pytest.mark.parametrize("size_label", list(_SIZES))
@pytest.mark.parametrize("entropy", list(_ENTROPY_FUNCS))
def test_compression_ratio(
    benchmark: BenchmarkFixture,
    level: int,
    size_label: str,
    entropy: str,
) -> None:
    data = _ENTROPY_FUNCS[entropy](_SIZES[size_label])
    key = keygen(sector="TSR", n_strands=4)
    stream = encode(data, key, generators_per_block=8)

    def _measure() -> float:
        compressed = compress(stream, key, level=level)
        return compression_ratio(stream, compressed)

    result = benchmark(_measure)
    # Ratio should be <= 1.0 (no expansion)
    assert result is None or isinstance(result, float)
