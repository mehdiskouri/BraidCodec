"""Throughput benchmarks for encode / decode pipeline."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from braidcodec import BraidKey, EncodedStream, decode, encode, keygen

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

_SIZES = {
    "1KB": 1_024,
    "10KB": 10_240,
    "100KB": 102_400,
}

_SECTORS: list[str] = ["TSR", "Ising", "Fibonacci"]
_STRANDS: list[int] = [3, 4, 5]


def _make_data(size: int) -> bytes:
    return os.urandom(size)


def _make_key(sector: str, n_strands: int) -> BraidKey:
    return keygen(sector=sector, n_strands=n_strands)


def _pre_encode(data: bytes, key: BraidKey) -> EncodedStream:
    return encode(data, key, generators_per_block=8)


# ── encode throughput ─────────────────────────────────────────────────


@pytest.mark.parametrize("size_label", list(_SIZES))
@pytest.mark.parametrize("n_strands", _STRANDS)
@pytest.mark.parametrize("sector", _SECTORS)
def test_encode_throughput(
    benchmark: BenchmarkFixture,
    size_label: str,
    n_strands: int,
    sector: str,
) -> None:
    data = _make_data(_SIZES[size_label])
    key = _make_key(sector, n_strands)
    benchmark(encode, data, key, generators_per_block=8)


# ── decode throughput (no verify) ─────────────────────────────────────


@pytest.mark.parametrize("size_label", list(_SIZES))
@pytest.mark.parametrize("n_strands", _STRANDS)
@pytest.mark.parametrize("sector", _SECTORS)
def test_decode_throughput(
    benchmark: BenchmarkFixture,
    size_label: str,
    n_strands: int,
    sector: str,
) -> None:
    data = _make_data(_SIZES[size_label])
    key = _make_key(sector, n_strands)
    stream = _pre_encode(data, key)
    benchmark(decode, stream, key, verify=False)


# ── decode throughput (with verify) ───────────────────────────────────


@pytest.mark.parametrize("size_label", list(_SIZES))
@pytest.mark.parametrize("n_strands", _STRANDS)
@pytest.mark.parametrize("sector", _SECTORS)
def test_decode_verified_throughput(
    benchmark: BenchmarkFixture,
    size_label: str,
    n_strands: int,
    sector: str,
) -> None:
    data = _make_data(_SIZES[size_label])
    key = _make_key(sector, n_strands)
    stream = _pre_encode(data, key)
    benchmark(decode, stream, key, verify=True)
