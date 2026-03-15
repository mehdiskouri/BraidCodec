"""Lightweight reconstructive-path benchmarks.

Phase H validation for current reconstructive routing:
- text/json/logs domains
- encode/decode/verify timing and diagnostics visibility
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import pytest

from braidcodec import decode, encode, keygen, verify

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


_CASES = [
    ("text", "Cafe\u0301\nlog line\n" * 200),
    ("json", json.dumps({"items": [{"x": i, "y": i * 2} for i in range(200)]}, sort_keys=True)),
    ("logs", "\n".join(f"2026-03-15T12:00:{i:02d}Z INFO core event={i}" for i in range(200))),
]


@pytest.mark.parametrize(("domain", "payload"), _CASES)
def test_reconstructive_encode_decode_verify(
    benchmark: BenchmarkFixture,
    domain: str,
    payload: str,
) -> None:
    key = keygen(sector="TSR", n_strands=4)
    data = payload.encode("utf-8")

    def _run() -> bool:
        stream = encode(
            data,
            key,
            generators_per_block=8,
            preprocessing_mode="reconstructive",
            reconstructive_domain=domain,
        )
        decoded = decode(stream, key, verify=False)
        result = verify(stream, key)
        return decoded == data and result.valid

    ok = benchmark(_run)
    assert ok is True

    start = time.perf_counter()
    stream = encode(
        data,
        key,
        generators_per_block=8,
        preprocessing_mode="reconstructive",
        reconstructive_domain=domain,
    )
    elapsed = max(time.perf_counter() - start, 1e-12)

    extra_info: dict[str, object] = benchmark.extra_info  # type: ignore[assignment]
    extra_info["regime_label"] = "reconstructive-core"
    extra_info["domain"] = domain
    extra_info["input_bytes"] = len(data)
    extra_info["throughput_MBps"] = round(len(data) / elapsed / 1e6, 6)
    extra_info["km_residual_max"] = float(stream.metadata["km_residual_max"])
    extra_info["km_residual_mean"] = float(stream.metadata["km_residual_mean"])
    extra_info["km_iters_mean"] = float(stream.metadata["km_iters_mean"])
    extra_info["km_valid_ratio"] = float(stream.metadata["km_valid_ratio"])
