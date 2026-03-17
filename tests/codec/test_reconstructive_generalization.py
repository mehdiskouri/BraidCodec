"""Generalization and retrieval behavior tests for reconstructive discovery."""

from __future__ import annotations

import json
import zlib
from base64 import b85decode
from typing import TYPE_CHECKING

import pytest

from braidcodec.codec.encoder import encode
from braidcodec.codec.equation_discovery import (
    compile_discovered_equation_program,
    discover_equations,
)
from braidcodec.codec.schema import parse_reconstructive_payload_metadata
from braidcodec.crypto.keys import BraidKey, keygen

if TYPE_CHECKING:
    from pathlib import Path


def _make_key() -> BraidKey:
    return keygen(sector="TSR", n_strands=4)


def _payload_program_type(stream_metadata: dict[str, str]) -> str:
    payload = parse_reconstructive_payload_metadata(stream_metadata)
    return payload["reconstructive_program_type"]


def _audit_value(stream_metadata: dict[str, str], key: str) -> str | None:
    packed = stream_metadata.get("ra1", "")
    if not packed.startswith("~ra85:"):
        return None
    raw = zlib.decompress(b85decode(packed[len("~ra85:") :].encode("ascii")))
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        return None
    value = obj.get(key)
    return None if value is None else str(value)


_DISCOVERED_PROGRAM_TYPES = {"discovered-equation-v1", "discovered-braid-equation-v1"}


def test_discovery_required_rejects_non_discoverable_payload() -> None:
    key = _make_key()
    # Natural language text is unlikely to match deterministic byte-family discovery.
    data = (
        b"This is a free-form UTF-8 sample that should not match constant, "
        b"linear, or affine byte laws. "
        b"It exists to exercise required discovery mode behavior."
    )

    with pytest.raises(ValueError, match="discovery required"):
        encode(
            data,
            key,
            preprocessing_mode="reconstructive",
            reconstructive_domain="text",
            reconstructive_discovery="required",
        )


def test_discovery_enabled_falls_back_when_no_equation() -> None:
    key = _make_key()
    data = (
        b"Fallback behavior should remain correct when discovery cannot produce "
        b"an exact governing equation."
    )

    stream = encode(
        data,
        key,
        preprocessing_mode="reconstructive",
        reconstructive_domain="text",
        reconstructive_discovery="enabled",
        reconstructive_compact_audit_bundle=True,
    )

    assert _payload_program_type(stream.metadata) in {
        "latent-residual-v2",
        "latent-residual-v3",
        "sparse-corrective-v1",
        "token-delta-grammar-v1",
        "phrase-dictionary-v1",
        "repeat-text-v1",
    }
    assert _audit_value(stream.metadata, "equation_library_hit") == "0"


def test_equation_library_retrieval_sets_hit_marker(tmp_path: Path) -> None:
    key = _make_key()
    library_path = tmp_path / "equation-library.json"
    data = b"A" * 256

    first = encode(
        data,
        key,
        preprocessing_mode="reconstructive",
        reconstructive_domain="text",
        reconstructive_discovery="enabled",
        reconstructive_library_path=str(library_path),
        reconstructive_compact_audit_bundle=True,
    )
    assert _payload_program_type(first.metadata) in _DISCOVERED_PROGRAM_TYPES
    assert _audit_value(first.metadata, "equation_library_hit") == "0"

    second = encode(
        data,
        key,
        preprocessing_mode="reconstructive",
        reconstructive_domain="text",
        reconstructive_discovery="enabled",
        reconstructive_library_path=str(library_path),
        reconstructive_compact_audit_bundle=True,
    )
    assert _payload_program_type(second.metadata) in _DISCOVERED_PROGRAM_TYPES
    assert _audit_value(second.metadata, "equation_library_hit") == "1"
    assert _audit_value(second.metadata, "equation_library_hit_mode") == "exact"


def test_equation_library_compatible_retrieval_sets_hit_marker(tmp_path: Path) -> None:
    key = _make_key()
    library_path = tmp_path / "equation-library-compatible.json"
    data = b"A" * 256

    discovered = discover_equations(data, domain_kind="text", profile_id="default-v1")
    assert discovered.equation is not None
    program = compile_discovered_equation_program(discovered.equation)

    library_payload = {
        "mismatched-signature-key": {
            "domain_kind": "text",
            "reconstructive_program_type": program["reconstructive_program_type"],
            "reconstructive_program_payload": program["reconstructive_program_payload"],
            "symbolic_hash": discovered.equation.symbolic_hash,
        }
    }
    library_path.write_text(json.dumps(library_payload, sort_keys=True, separators=(",", ":")))

    stream = encode(
        data,
        key,
        preprocessing_mode="reconstructive",
        reconstructive_domain="text",
        reconstructive_discovery="enabled",
        reconstructive_library_path=str(library_path),
        reconstructive_compact_audit_bundle=True,
    )

    assert _payload_program_type(stream.metadata) in _DISCOVERED_PROGRAM_TYPES
    assert _audit_value(stream.metadata, "equation_library_hit") == "1"
    assert _audit_value(stream.metadata, "equation_library_hit_mode") == "compatible"


def test_strict_gate_profile_rejects_when_metrics_below_floor() -> None:
    key = _make_key()
    data = ("Cafe\u0301\nlog line\n" * 80).encode("utf-8")

    with pytest.raises(ValueError, match="strict gate failed"):
        encode(
            data,
            key,
            preprocessing_mode="reconstructive",
            reconstructive_domain="text",
            reconstructive_strict_gate_profile="strict-v1",
        )
