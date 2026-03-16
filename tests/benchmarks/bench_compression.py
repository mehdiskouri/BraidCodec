"""Compression benchmarks — ratio delta vs gzip/zstd, throughput by level and entropy.

PRD §13.1: report compression ratio per-file and aggregate, compare against
raw (no compression), gzip-6, zstd-3.  Report delta between BraidCodec
topological compression and baseline compressors.
"""

from __future__ import annotations

import gzip
import os
import zlib
from typing import TYPE_CHECKING

import pytest

from braidcodec import encode, keygen
from braidcodec.codec.compressor import compress, compression_ratio

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


# ── Input corpora ─────────────────────────────────────────────────────


def _zeros(n: int) -> bytes:
    """Maximally compressible."""
    return b"\x00" * n


def _random(n: int) -> bytes:
    """Incompressible baseline."""
    return os.urandom(n)


def _text(n: int) -> bytes:
    """Structured English-like repetition."""
    sentence = b"The braid group acts on the topological state space. "
    return (sentence * (n // len(sentence) + 1))[:n]


def _json(n: int) -> bytes:
    """Structured JSON-like data."""
    record = b'{"id":1,"name":"alice","score":0.42,"tags":["a","b"]},'
    return b"[" + (record * (n // len(record) + 1))[:n] + b"]"


def _binary(n: int) -> bytes:
    """Repeating binary pattern (e.g. protobuf-like)."""
    pattern = bytes(range(256))
    return (pattern * (n // 256 + 1))[:n]


_CORPUS = {
    "zeros": _zeros,
    "random": _random,
    "text": _text,
    "json": _json,
    "binary": _binary,
}

_SIZES = {"10KB": 10_240, "100KB": 102_400}
_LEVELS = [0, 1, 2]


# ── Helpers ───────────────────────────────────────────────────────────


def _gzip_ratio(data: bytes) -> float:
    """gzip -6 compression ratio (compressed / original)."""
    return len(gzip.compress(data, compresslevel=6)) / len(data)


def _zstd_ratio(data: bytes) -> float:
    """zlib (deflate, level 3) compression ratio as zstd proxy."""
    return len(zlib.compress(data, level=3)) / len(data)


def _braid_generator_ratio(data: bytes, level: int) -> float:
    """BraidCodec topological compression ratio (generator counts)."""
    key = keygen(sector="TSR", n_strands=4)
    stream = encode(data, key, generators_per_block=32)
    compressed = compress(stream, key, level=level)
    return compression_ratio(stream, compressed)


def _braid_byte_ratio(data: bytes, level: int) -> float:
    """BraidCodec wire-format byte ratio (compressed bytes / original bytes)."""
    key = keygen(sector="TSR", n_strands=4)
    stream = encode(data, key, generators_per_block=32)
    compressed = compress(stream, key, level=level)
    return len(compressed.to_bytes()) / len(data)


# ── Compression throughput ────────────────────────────────────────────


@pytest.mark.parametrize("level", _LEVELS)
@pytest.mark.parametrize("size_label", list(_SIZES))
@pytest.mark.parametrize("corpus", list(_CORPUS))
def test_compress_throughput(
    benchmark: BenchmarkFixture,
    level: int,
    size_label: str,
    corpus: str,
) -> None:
    nbytes = _SIZES[size_label]
    data = _CORPUS[corpus](nbytes)
    key = keygen(sector="TSR", n_strands=4)
    stream = encode(data, key, generators_per_block=32)

    benchmark(compress, stream, key, level=level)

    benchmark.extra_info["input_bytes"] = nbytes
    benchmark.extra_info["corpus"] = corpus
    benchmark.extra_info["level"] = level


# ── Compression ratio with delta vs gzip/zstd ────────────────────────


@pytest.mark.parametrize("level", _LEVELS)
@pytest.mark.parametrize("size_label", list(_SIZES))
@pytest.mark.parametrize("corpus", list(_CORPUS))
def test_compression_ratio_delta(
    benchmark: BenchmarkFixture,
    level: int,
    size_label: str,
    corpus: str,
) -> None:
    """Measure BraidCodec generator ratio AND report delta vs gzip-6/zlib-3."""
    nbytes = _SIZES[size_label]
    data = _CORPUS[corpus](nbytes)

    # Baseline compressor ratios (not timed — just reference)
    gzip_r = _gzip_ratio(data)
    zstd_r = _zstd_ratio(data)

    # BraidCodec topological compression
    key = keygen(sector="TSR", n_strands=4)
    stream = encode(data, key, generators_per_block=32)

    def _measure() -> float:
        compressed = compress(stream, key, level=level)
        return compression_ratio(stream, compressed)

    braid_gen_r = benchmark(_measure)

    # Wire-format byte ratio (encoding overhead included)
    compressed_stream = compress(stream, key, level=level)
    braid_byte_r = len(compressed_stream.to_bytes()) / nbytes

    # Report all ratios and deltas
    benchmark.extra_info["braid_generator_ratio"] = round(braid_gen_r, 4) if braid_gen_r else None
    benchmark.extra_info["braid_byte_ratio"] = round(braid_byte_r, 4)
    benchmark.extra_info["gzip6_ratio"] = round(gzip_r, 4)
    benchmark.extra_info["zlib3_ratio"] = round(zstd_r, 4)
    benchmark.extra_info["delta_vs_gzip"] = (
        round(braid_byte_r - gzip_r, 4) if braid_gen_r else None
    )
    benchmark.extra_info["delta_vs_zlib"] = (
        round(braid_byte_r - zstd_r, 4) if braid_gen_r else None
    )
    benchmark.extra_info["corpus"] = corpus
    benchmark.extra_info["level"] = level
    benchmark.extra_info["input_bytes"] = nbytes
