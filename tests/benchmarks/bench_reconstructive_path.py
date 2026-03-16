"""Lightweight reconstructive-path benchmarks.

Phase H validation for current reconstructive routing:
- text/json/logs domains
- encode/decode/verify timing and diagnostics visibility
"""

from __future__ import annotations

import json
import time
import zlib
from base64 import b85decode
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from braidcodec import decode, encode, keygen, verify
from braidcodec.codec.schema import parse_reconstructive_payload_metadata

_DISCOVERED_PROGRAM_TYPES = {"discovered-equation-v1", "discovered-braid-equation-v1"}

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


_CASES = [
    ("text", "Cafe\u0301\nlog line\n" * 200),
    ("json", json.dumps({"items": [{"x": i, "y": i * 2} for i in range(200)]}, sort_keys=True)),
    ("logs", "\n".join(f"2026-03-15T12:00:{i:02d}Z INFO core event={i}" for i in range(200))),
]

_DISCOVERY_CASES = [
    ("discoverable-constant", "text", "A" * 2048, True),
    (
        "fallback-freeform",
        "text",
        "This freeform utf-8 sentence intentionally resists exact byte-law discovery. " * 20,
        False,
    ),
]


def _program_type_from_stream(stream_metadata: dict[str, str]) -> str:
    try:
        payload = parse_reconstructive_payload_metadata(stream_metadata)
    except Exception:
        return ""
    return str(payload.get("reconstructive_program_type", ""))


def _audit_bundle(stream_metadata: dict[str, str]) -> dict[str, str]:
    packed = stream_metadata.get("ra1", "")
    if not packed.startswith("~ra85:"):
        return {}
    try:
        raw = zlib.decompress(b85decode(packed[len("~ra85:") :].encode("ascii")))
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    return {str(k): str(v) for k, v in obj.items()}


def _compact_metadata_form(stream_metadata: dict[str, str]) -> str:
    if "rh" in stream_metadata:
        return "header-rh"
    if "rt" in stream_metadata and "rpb" in stream_metadata:
        return "rt-rpb"
    if "rt" in stream_metadata:
        return "rt-only"
    return "legacy"


def _commitment_version(stream_metadata: dict[str, str]) -> str:
    if "rc3" in stream_metadata or "reconstructive_commitment_v3" in stream_metadata:
        return "v3"
    if "rc" in stream_metadata or "reconstructive_commitment" in stream_metadata:
        return "v2"
    return "none"


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
            reconstructive_compact_audit_bundle=True,
        )
        decoded = decode(stream, key, verify=False)
        result = verify(stream, key)
        return decoded == data and result.valid

    ok = cast("bool", benchmark(_run))
    assert ok is True

    start = time.perf_counter()
    stream = encode(
        data,
        key,
        generators_per_block=8,
        preprocessing_mode="reconstructive",
        reconstructive_domain=domain,
        reconstructive_compact_audit_bundle=True,
    )
    elapsed = max(time.perf_counter() - start, 1e-12)
    audit = _audit_bundle(stream.metadata)

    extra_info: dict[str, object] = benchmark.extra_info  # type: ignore[assignment]
    extra_info["regime_label"] = "reconstructive-core"
    extra_info["domain"] = domain
    extra_info["input_bytes"] = len(data)
    extra_info["throughput_MBps"] = round(len(data) / elapsed / 1e6, 6)
    extra_info["wire_bytes"] = len(stream.to_bytes())
    extra_info["hdf5_bytes"] = len(stream.to_hdf5_bytes())
    extra_info["wire_compression_ratio"] = round(len(data) / max(len(stream.to_bytes()), 1), 6)
    extra_info["hdf5_compression_ratio"] = round(len(data) / max(len(stream.to_hdf5_bytes()), 1), 6)
    extra_info["km_residual_max"] = float(audit.get("km_residual_max", "0"))
    extra_info["km_residual_mean"] = float(audit.get("km_residual_mean", "0"))
    extra_info["km_iters_mean"] = float(audit.get("km_iters_mean", "0"))
    extra_info["km_valid_ratio"] = float(audit.get("km_valid_ratio", "0"))
    extra_info["fidelity_energy"] = float(audit.get("fidelity_energy", "0"))
    extra_info["fidelity_topology"] = float(audit.get("fidelity_topology", "0"))
    extra_info["fidelity_coherence"] = float(audit.get("fidelity_coherence", "0"))
    extra_info["discovery_profile_id"] = audit.get("discovery_profile_id", "")
    extra_info["equation_library_hit"] = int(audit.get("equation_library_hit", "0"))
    extra_info["equation_library_hit_mode"] = audit.get("equation_library_hit_mode", "none")
    extra_info["equation_library_signature"] = audit.get("equation_library_signature", "")
    extra_info["equation_symbolic_hash"] = audit.get("equation_symbolic_hash", "")

    program_sidechannel = stream.metadata.get("rpb", "")
    if program_sidechannel:
        extra_info["reconstructive_program_payload_bytes"] = len(program_sidechannel.encode("utf-8"))
    program_type = _program_type_from_stream(stream.metadata)
    if program_type:
        extra_info["reconstructive_program_type"] = program_type
        extra_info["used_discovered_equation"] = int(program_type in _DISCOVERED_PROGRAM_TYPES)
        extra_info["used_residual_fallback"] = int(program_type in {"latent-residual-v2", "latent-residual-v3"})
    extra_info["compact_metadata_form"] = _compact_metadata_form(stream.metadata)
    extra_info["commitment_version"] = _commitment_version(stream.metadata)


@pytest.mark.parametrize(("label", "domain", "payload", "expect_discovered"), _DISCOVERY_CASES)
def test_reconstructive_discovery_program_mix(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
    label: str,
    domain: str,
    payload: str,
    expect_discovered: bool,
) -> None:
    key = keygen(sector="TSR", n_strands=4)
    data = payload.encode("utf-8")
    library_path = tmp_path / f"equation-library-{label}.json"

    def _run() -> bool:
        stream = encode(
            data,
            key,
            generators_per_block=8,
            preprocessing_mode="reconstructive",
            reconstructive_domain=domain,
            reconstructive_discovery="enabled",
            reconstructive_library_path=str(library_path),
            reconstructive_compact_audit_bundle=True,
        )
        decoded = decode(stream, key, verify=False)
        return decoded == data

    ok = cast("bool", benchmark(_run))
    assert ok is True

    stream = encode(
        data,
        key,
        generators_per_block=8,
        preprocessing_mode="reconstructive",
        reconstructive_domain=domain,
        reconstructive_discovery="enabled",
        reconstructive_library_path=str(library_path),
        reconstructive_compact_audit_bundle=True,
    )
    audit = _audit_bundle(stream.metadata)
    program_type = _program_type_from_stream(stream.metadata)
    used_discovered = program_type in _DISCOVERED_PROGRAM_TYPES

    assert used_discovered is expect_discovered

    extra_info: dict[str, object] = benchmark.extra_info  # type: ignore[assignment]
    extra_info["regime_label"] = "reconstructive-discovery-program-mix"
    extra_info["case_label"] = label
    extra_info["domain"] = domain
    extra_info["input_bytes"] = len(data)
    extra_info["program_type"] = program_type
    extra_info["reconstructive_program_type"] = program_type
    program_sidechannel = stream.metadata.get("rpb", "")
    if program_sidechannel:
        extra_info["reconstructive_program_payload_bytes"] = len(program_sidechannel.encode("utf-8"))
    extra_info["used_discovered_equation"] = int(used_discovered)
    extra_info["used_residual_fallback"] = int(
        program_type in {"latent-residual-v2", "latent-residual-v3"}
    )
    extra_info["equation_library_hit"] = int(audit.get("equation_library_hit", "0"))
    extra_info["equation_library_hit_mode"] = audit.get("equation_library_hit_mode", "none")
    extra_info["compact_metadata_form"] = _compact_metadata_form(stream.metadata)
    extra_info["commitment_version"] = _commitment_version(stream.metadata)


def test_reconstructive_library_reuse_progression(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
) -> None:
    key = keygen(sector="TSR", n_strands=4)
    data = ("A" * 2048).encode("utf-8")
    library_path = tmp_path / "equation-library-reuse.json"

    def _run() -> int:
        hits = 0
        for _ in range(8):
            stream = encode(
                data,
                key,
                generators_per_block=8,
                preprocessing_mode="reconstructive",
                reconstructive_domain="text",
                reconstructive_discovery="enabled",
                reconstructive_library_path=str(library_path),
                reconstructive_compact_audit_bundle=True,
            )
            hits += int(_audit_bundle(stream.metadata).get("equation_library_hit", "0"))
        return hits

    hits_bench = cast("int", benchmark(_run))
    assert hits_bench >= 1

    # Re-run once for deterministic metrics snapshot.
    hits = 0
    discovered = 0
    fallback = 0
    runs = 8
    last_stream = None
    for _ in range(runs):
        stream = encode(
            data,
            key,
            generators_per_block=8,
            preprocessing_mode="reconstructive",
            reconstructive_domain="text",
            reconstructive_discovery="enabled",
            reconstructive_library_path=str(library_path),
            reconstructive_compact_audit_bundle=True,
        )
        hits += int(_audit_bundle(stream.metadata).get("equation_library_hit", "0"))
        program_type = _program_type_from_stream(stream.metadata)
        discovered += int(program_type in _DISCOVERED_PROGRAM_TYPES)
        fallback += int(program_type in {"latent-residual-v2", "latent-residual-v3"})
        last_stream = stream

    extra_info: dict[str, object] = benchmark.extra_info  # type: ignore[assignment]
    extra_info["regime_label"] = "reconstructive-library-reuse"
    extra_info["runs"] = runs
    extra_info["library_hit_count"] = hits
    extra_info["library_hit_rate"] = round(hits / max(runs, 1), 6)
    extra_info["discovered_count"] = discovered
    extra_info["fallback_count"] = fallback
    if last_stream is not None:
        extra_info["compact_metadata_form"] = _compact_metadata_form(last_stream.metadata)
        extra_info["commitment_version"] = _commitment_version(last_stream.metadata)


def test_reconstructive_strict_profile_failure_cost(benchmark: BenchmarkFixture) -> None:
    key = keygen(sector="TSR", n_strands=4)
    data = ("Cafe\u0301\nlog line\n" * 80).encode("utf-8")

    def _run() -> bool:
        try:
            encode(
                data,
                key,
                generators_per_block=8,
                preprocessing_mode="reconstructive",
                reconstructive_domain="text",
                reconstructive_strict_gate_profile="strict-v1",
            )
        except ValueError:
            return True
        return False

    rejected = cast("bool", benchmark(_run))
    assert rejected is True

    extra_info: dict[str, object] = benchmark.extra_info  # type: ignore[assignment]
    extra_info["regime_label"] = "reconstructive-strict-profile"
    extra_info["strict_profile"] = "strict-v1"
    extra_info["rejected"] = 1
