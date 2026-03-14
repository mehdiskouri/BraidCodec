"""Integrity verification benchmarks."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from braidcodec import encode, keygen, verify
from braidcodec.algebra.braid_equations import BraidEquation, writhe

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

_BLOCK_COUNTS = [1, 5, 10, 50]


def _make_stream(n_blocks: int) -> tuple:
    """Create a stream with approximately *n_blocks* blocks."""
    key = keygen(sector="TSR", n_strands=4)
    # ~8 generators per block → each block ≈ a few bytes of input
    # 1 block ≈ 3-4 bytes with generators_per_block=8 and n_strands=4
    data = os.urandom(max(4 * n_blocks, 1))
    stream = encode(data, key, generators_per_block=8)
    return stream, key


# ── verify throughput ─────────────────────────────────────────────────


@pytest.mark.parametrize("n_blocks", _BLOCK_COUNTS)
@pytest.mark.parametrize("fermion_check", [False, True])
def test_verify_throughput(
    benchmark: BenchmarkFixture,
    n_blocks: int,
    fermion_check: bool,
) -> None:
    stream, key = _make_stream(n_blocks)
    benchmark(verify, stream, key, fermion_check=fermion_check)


# ── writhe-only throughput ────────────────────────────────────────────


@pytest.mark.parametrize("n_blocks", _BLOCK_COUNTS)
def test_writhe_only(
    benchmark: BenchmarkFixture,
    n_blocks: int,
) -> None:
    stream, _key = _make_stream(n_blocks)

    def _compute_writhes() -> list[int]:
        return [
            writhe(
                BraidEquation(
                    stream.n_strands,
                    block.generators,
                    sector=stream.sector,
                )
            )
            for block in stream.blocks
        ]

    benchmark(_compute_writhes)
