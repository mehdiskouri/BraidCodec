"""Tests for equation discovery and discovered-equation compact synthesis."""

from __future__ import annotations

import base64
import json
import zlib

import pytest

from braidcodec.codec.equation_discovery import (
    compile_discovered_braid_equation_program,
    compile_discovered_equation_program,
    discover_equations,
)
from braidcodec.codec.reconstructive_compact import synthesize_reconstructive_bytes


def test_discovery_finds_linear_mod_sequence() -> None:
    source = bytes((5 + (7 * i)) % 256 for i in range(128))
    discovered = discover_equations(source, domain_kind="text", profile_id="default-v1")

    assert discovered.equation is not None
    assert discovered.equation.equation_family in {
        "byte-linear-mod-v1",
        "byte-affine-recursion-v1",
    }

    program = compile_discovered_equation_program(discovered.equation)
    reconstructed = synthesize_reconstructive_bytes(program)
    assert reconstructed == source


def test_discovery_handles_constant_sequence() -> None:
    source = b"X" * 64
    discovered = discover_equations(source, domain_kind="logs", profile_id="default-v1")

    assert discovered.equation is not None
    assert discovered.equation.equation_family == "byte-constant-v1"

    program = compile_discovered_equation_program(discovered.equation)
    assert synthesize_reconstructive_bytes(program) == source


def test_discovery_finds_xor_step_sequence() -> None:
    source = bytearray(160)
    source[0] = 173
    step = 29
    for i in range(1, len(source)):
        source[i] = source[i - 1] ^ step

    discovered = discover_equations(bytes(source), domain_kind="text", profile_id="default-v1")
    assert discovered.equation is not None
    assert discovered.equation.equation_family in {
        "byte-xor-step-v1",
        "byte-affine-recursion-v1",
    }

    program = compile_discovered_equation_program(discovered.equation)
    assert synthesize_reconstructive_bytes(program) == bytes(source)


def test_discovery_braid_program_roundtrip_for_xor_step_sequence() -> None:
    source = bytearray(192)
    source[0] = 91
    step = 43
    for i in range(1, len(source)):
        source[i] = source[i - 1] ^ step

    discovered = discover_equations(bytes(source), domain_kind="text", profile_id="default-v1")
    assert discovered.equation is not None

    program = compile_discovered_braid_equation_program(discovered.equation)
    assert synthesize_reconstructive_bytes(program) == bytes(source)


def test_discovery_braid_program_roundtrip() -> None:
    source = bytes((17 * i + 9) % 256 for i in range(96))
    discovered = discover_equations(source, domain_kind="text", profile_id="default-v1")

    assert discovered.equation is not None
    program = compile_discovered_braid_equation_program(discovered.equation)

    assert program["reconstructive_program_type"] == "discovered-braid-equation-v1"
    assert synthesize_reconstructive_bytes(program) == source


def test_discovery_braid_program_is_more_compact_than_legacy_program() -> None:
    source = b"A" * 512
    discovered = discover_equations(source, domain_kind="text", profile_id="default-v1")

    assert discovered.equation is not None
    legacy_program = compile_discovered_equation_program(discovered.equation)
    braid_program = compile_discovered_braid_equation_program(discovered.equation)

    assert len(braid_program["reconstructive_program_payload"]) < len(
        legacy_program["reconstructive_program_payload"]
    )


def test_discovery_braid_program_rejects_legacy_keys() -> None:
    source = b"A" * 512
    discovered = discover_equations(source, domain_kind="text", profile_id="default-v1")

    assert discovered.equation is not None
    braid_program = compile_discovered_braid_equation_program(discovered.equation)

    payload = json.loads(braid_program["reconstructive_program_payload"])
    assert isinstance(payload, dict)
    legacy_payload = {
        "ec": payload["e"],
        "rl": payload["r"],
        "bg85": payload["g"],
    }

    legacy_program = {
        "reconstructive_program_type": "discovered-braid-equation-v1",
        "reconstructive_program_payload": json.dumps(legacy_payload, separators=(",", ":")),
    }
    with pytest.raises(Exception):
        synthesize_reconstructive_bytes(legacy_program)


def test_discovery_braid_program_sidechannel_packed_payload_supported() -> None:
    source = b"A" * 512
    discovered = discover_equations(source, domain_kind="text", profile_id="default-v1")
    assert discovered.equation is not None

    program = compile_discovered_braid_equation_program(discovered.equation)
    raw_payload = program["reconstructive_program_payload"]
    packed = "~sp85:" + base64.b85encode(zlib.compress(raw_payload.encode("utf-8"), level=9)).decode("ascii")
    wrapped_program = {
        "reconstructive_program_type": "discovered-braid-equation-v1",
        "reconstructive_program_payload": packed,
    }
    assert synthesize_reconstructive_bytes(wrapped_program) == source
