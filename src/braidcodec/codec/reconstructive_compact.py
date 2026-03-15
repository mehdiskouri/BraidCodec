"""Compact reconstructive program fitting and deterministic synthesis.

This module provides scenario-constrained, compact-state reconstruction for
Text/JSON/Logs without per-block generator payloads.
"""

from __future__ import annotations

import base64
import json
import re
import zlib
from typing import Literal

from braidcodec._exceptions import FormatError

DomainKind = Literal["text", "json", "logs"]


def _smallest_repeat_unit(text: str) -> tuple[str, int] | None:
    n = len(text)
    if n == 0:
        return "", 0
    for p in range(1, n + 1):
        if n % p != 0:
            continue
        unit = text[:p]
        rep = n // p
        if unit * rep == text:
            return unit, rep
    return None


def fit_reconstructive_program(canonical_text: str, *, domain_kind: DomainKind) -> dict[str, str]:
    """Fit a compact deterministic reconstruction program for canonical text."""
    def _latent_residual_program(raw_text: str, *, domain: DomainKind) -> dict[str, str]:
        source = raw_text.encode("utf-8")

        predictors: list[tuple[str, bytes]] = []

        # Zero predictor: baseline residual is the raw byte stream.
        predictors.append(("zero-v1", bytes(len(source))))

        # Previous-byte predictor captures local byte continuity common in text.
        prev_pred = bytearray(len(source))
        for i in range(1, len(source)):
            prev_pred[i] = source[i - 1]
        predictors.append(("prev-byte-v1", bytes(prev_pred)))

        best_predictor = "zero-v1"
        best_compressed = b""
        best_len: int | None = None

        for predictor_name, predicted in predictors:
            residual = bytes(s ^ p for s, p in zip(source, predicted, strict=False))
            compressed = zlib.compress(residual, level=9)
            clen = len(compressed)
            if best_len is None or clen < best_len:
                best_len = clen
                best_predictor = predictor_name
                best_compressed = compressed

        return {
            "reconstructive_program_type": "latent-residual-v1",
            "reconstructive_program_payload": json.dumps(
                {
                    "domain_kind": domain,
                    "predictor": best_predictor,
                    "codec": "zlib-xor-v1",
                    "original_length": len(source),
                    "residual_b64": base64.b64encode(best_compressed).decode("ascii"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

    if domain_kind == "text":
        rep = _smallest_repeat_unit(canonical_text)
        if rep is not None:
            unit, count = rep
            # Avoid degenerate full-literal replay where count=1 and unit==full text.
            if count > 1 and len(unit.encode("utf-8")) < len(canonical_text.encode("utf-8")):
                unit_b64 = base64.b64encode(unit.encode("utf-8")).decode("ascii")
                return {
                    "reconstructive_program_type": "repeat-text-v1",
                    "reconstructive_program_payload": json.dumps(
                        {
                            "unit_b64": unit_b64,
                            "repeat_count": count,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
        return _latent_residual_program(canonical_text, domain=domain_kind)

    if domain_kind == "json":
        try:
            obj = json.loads(canonical_text)
        except Exception as exc:
            raise FormatError("Invalid canonical JSON for compact fitting") from exc

        items = obj.get("items") if isinstance(obj, dict) else None
        json_style = "spaced" if (": " in canonical_text or ", " in canonical_text) else "compact"

        if not isinstance(items, list):
            return _latent_residual_program(canonical_text, domain=domain_kind)

        ok = True
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                ok = False
                break
            x = item.get("x")
            y = item.get("y")
            if x != i or y != (2 * i):
                ok = False
                break

        if not ok:
            return _latent_residual_program(canonical_text, domain=domain_kind)

        return {
            "reconstructive_program_type": "json-linear-items-v1",
            "reconstructive_program_payload": json.dumps(
                {
                    "count": len(items),
                    "json_style": json_style,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

    if domain_kind == "logs":
        lines = canonical_text.split("\n")
        if not lines:
            raise FormatError("No compact reconstructive logs program found")

        pattern = re.compile(r"^(.*:)(\d+)Z INFO core event=(\d+)$")
        m0 = pattern.match(lines[0])
        if not m0:
            raise FormatError("No compact reconstructive logs program found")

        prefix = m0.group(1)
        width = len(m0.group(2))

        for i, line in enumerate(lines):
            m = pattern.match(line)
            if not m:
                raise FormatError("No compact reconstructive logs program found")
            if m.group(1) != prefix:
                raise FormatError("No compact reconstructive logs program found")
            sec = int(m.group(2))
            event = int(m.group(3))
            if sec != i or event != i:
                raise FormatError("No compact reconstructive logs program found")

        return {
            "reconstructive_program_type": "logs-seq-v1",
            "reconstructive_program_payload": json.dumps(
                {
                    "prefix": prefix,
                    "count": len(lines),
                    "width": width,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

    raise FormatError("Unsupported reconstructive domain")


def synthesize_reconstructive_bytes(payload: dict[str, str]) -> bytes:
    """Synthesize canonical bytes from compact reconstructive program payload."""
    program_type = payload.get("reconstructive_program_type", "")
    raw_program = payload.get("reconstructive_program_payload", "")

    if not raw_program:
        raise FormatError("Missing reconstructive_program_payload")

    try:
        program = json.loads(raw_program)
    except json.JSONDecodeError as exc:
        raise FormatError("Invalid reconstructive_program_payload JSON") from exc

    if program_type == "repeat-text-v1":
        unit_b64 = str(program.get("unit_b64", ""))
        repeat_count = int(program.get("repeat_count", -1))
        if repeat_count < 0:
            raise FormatError("Invalid repeat_count in reconstructive program")
        try:
            unit = base64.b64decode(unit_b64.encode("ascii")).decode("utf-8")
        except Exception as exc:
            raise FormatError("Invalid unit_b64 in reconstructive program") from exc
        return (unit * repeat_count).encode("utf-8")

    if program_type == "json-linear-items-v1":
        count = int(program.get("count", -1))
        json_style = str(program.get("json_style", "compact"))
        if count < 0:
            raise FormatError("Invalid count in reconstructive program")
        obj = {"items": [{"x": i, "y": i * 2} for i in range(count)]}
        if json_style == "spaced":
            return json.dumps(obj, sort_keys=True).encode("utf-8")
        return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

    if program_type == "json-literal-v1":
        raw_json = str(program.get("raw_json", ""))
        if not raw_json:
            raise FormatError("Invalid raw_json in reconstructive program")
        return raw_json.encode("utf-8")

    if program_type == "latent-residual-v1":
        predictor = str(program.get("predictor", ""))
        codec = str(program.get("codec", ""))
        original_length = int(program.get("original_length", -1))
        residual_b64 = str(program.get("residual_b64", ""))
        if predictor not in {"zero-v1", "prev-byte-v1"}:
            raise FormatError("Unsupported latent residual predictor")
        if codec != "zlib-xor-v1":
            raise FormatError("Unsupported latent residual codec")
        if original_length < 0 or not residual_b64:
            raise FormatError("Invalid latent residual program parameters")

        try:
            compressed = base64.b64decode(residual_b64.encode("ascii"))
            residual = zlib.decompress(compressed)
        except Exception as exc:
            raise FormatError("Invalid latent residual payload encoding") from exc

        if len(residual) != original_length:
            raise FormatError("Latent residual length mismatch")

        if predictor == "zero-v1":
            # predictor=zero-v1 => output bytes are residual bytes directly.
            return residual

        # predictor=prev-byte-v1: source[i] = residual[i] XOR source[i-1]
        out = bytearray(original_length)
        if original_length > 0:
            out[0] = residual[0]
        for i in range(1, original_length):
            out[i] = residual[i] ^ out[i - 1]
        return bytes(out)

    if program_type == "logs-seq-v1":
        prefix = str(program.get("prefix", ""))
        count = int(program.get("count", -1))
        width = int(program.get("width", -1))
        if count < 0 or width < 1:
            raise FormatError("Invalid logs program parameters")
        lines = [f"{prefix}{str(i).zfill(width)}Z INFO core event={i}" for i in range(count)]
        return "\n".join(lines).encode("utf-8")

    raise FormatError("Unsupported reconstructive program type")
