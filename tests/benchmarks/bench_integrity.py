"""Integrity fidelity benchmarks — detection rate, false-positive rate, latency.

PRD §13.3: encode a file, randomly flip 1/2/4/8/16 bits in the encoded
representation. Measure detection rate (should be 100%), false positive
rate (should be 0%), and time-to-detect (first-failing-block latency).
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from braidcodec import encode, keygen, verify

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

_BIT_FLIPS = [1, 2, 4, 8, 16]
_TRIALS = 5  # trials per flip count for statistical confidence


def _make_stream(nbytes: int = 1024, gpb: int = 32):
    """Encode random data, return (stream, key)."""
    key = keygen(sector="TSR", n_strands=4)
    data = os.urandom(nbytes)
    stream = encode(data, key, generators_per_block=gpb)
    return stream, key


def _flip_bits_in_stream(stream, n_flips: int):
    """Corrupt n_flips generators across random blocks.

    Each corruption changes either the sign or the magnitude of a
    generator while keeping |g| in [1, n_strands-1] so that the
    structural check still passes — this forces the deeper channels
    (writhe, invariant, checksum) to catch it.
    """
    blocks = list(stream.blocks)
    rng = random.Random(42 + n_flips)

    flips_done = 0
    attempts = 0
    while flips_done < n_flips and attempts < n_flips * 10:
        attempts += 1
        block_idx = rng.randrange(len(blocks))
        block = blocks[block_idx]
        if not block.generators:
            continue
        gen_idx = rng.randrange(len(block.generators))
        gen_val = block.generators[gen_idx]

        # Strategy: negate sign (guaranteed to change writhe)
        flipped_val = -gen_val

        new_gens = list(block.generators)
        new_gens[gen_idx] = flipped_val
        blocks[block_idx] = replace(block, generators=new_gens)
        flips_done += 1

    return replace(stream, blocks=tuple(blocks))


# ── Detection rate: must be 100% ──────────────────────────────────────


@pytest.mark.parametrize("n_flips", _BIT_FLIPS)
def test_detection_rate(
    benchmark: BenchmarkFixture,
    n_flips: int,
) -> None:
    """Flip n_flips generators, run verify, assert detected. Repeat _TRIALS times."""
    stream, key = _make_stream(nbytes=512, gpb=8)

    detected = 0
    total = _TRIALS

    def _run_detection() -> int:
        nonlocal detected
        detected = 0
        for _trial in range(total):
            corrupted = _flip_bits_in_stream(stream, n_flips)
            result = verify(corrupted, key)
            if not result.valid:
                detected += 1
        return detected

    benchmark(_run_detection)

    detection_rate = detected / total
    benchmark.extra_info["n_flips"] = n_flips
    benchmark.extra_info["trials"] = total
    benchmark.extra_info["detected"] = detected
    benchmark.extra_info["detection_rate"] = detection_rate
    assert detection_rate == 1.0, (
        f"Detection rate {detection_rate:.0%} < 100% for {n_flips} bit flips"
    )


# ── False positive rate: must be 0% ──────────────────────────────────


def test_false_positive_rate(benchmark: BenchmarkFixture) -> None:
    """Verify unmodified data — should always pass."""
    total = _TRIALS

    def _run_fp_check() -> int:
        false_positives = 0
        for _ in range(total):
            stream, key = _make_stream(nbytes=512, gpb=8)
            result = verify(stream, key)
            if not result.valid:
                false_positives += 1
        return false_positives

    fp_count = benchmark(_run_fp_check)

    fp_rate = (fp_count if fp_count else 0) / total
    benchmark.extra_info["trials"] = total
    benchmark.extra_info["false_positives"] = fp_count if fp_count else 0
    benchmark.extra_info["false_positive_rate"] = fp_rate
    assert fp_rate == 0.0, f"False positive rate {fp_rate:.0%} > 0%"


# ── First-failure latency ─────────────────────────────────────────────


@pytest.mark.parametrize("n_flips", _BIT_FLIPS)
def test_first_failure_latency(
    benchmark: BenchmarkFixture,
    n_flips: int,
) -> None:
    """Measure wall-clock time from verify() call to detection."""
    stream, key = _make_stream(nbytes=512, gpb=8)
    corrupted = _flip_bits_in_stream(stream, n_flips)

    def _measure_latency() -> float:
        t0 = time.perf_counter()
        result = verify(corrupted, key)
        elapsed = time.perf_counter() - t0
        assert not result.valid
        return elapsed

    latency = benchmark(_measure_latency)

    benchmark.extra_info["n_flips"] = n_flips
    benchmark.extra_info["n_blocks"] = len(stream.blocks)
    if latency:
        benchmark.extra_info["first_failure_latency_ms"] = round(latency * 1000, 3)


# ── Verify throughput by block count ──────────────────────────────────


_BLOCK_COUNTS = [1, 5, 10, 50]


@pytest.mark.parametrize("n_blocks", _BLOCK_COUNTS)
@pytest.mark.parametrize("fermion_check", [False, True])
def test_verify_throughput(
    benchmark: BenchmarkFixture,
    n_blocks: int,
    fermion_check: bool,
) -> None:
    """Verify throughput on clean data, scaling with block count."""
    key = keygen(sector="TSR", n_strands=4)
    data = os.urandom(max(4 * n_blocks, 1))
    stream = encode(data, key, generators_per_block=8)

    benchmark(verify, stream, key, fermion_check=fermion_check)

    benchmark.extra_info["n_blocks_actual"] = len(stream.blocks)
    benchmark.extra_info["fermion_check"] = fermion_check
