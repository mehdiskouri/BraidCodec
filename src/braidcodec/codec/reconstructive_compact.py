"""Compact reconstructive program fitting and deterministic synthesis.

This module provides scenario-constrained, compact-state reconstruction for
Text/JSON/Logs without per-block generator payloads.
"""

from __future__ import annotations

import base64
import bz2
import json
import lzma
import re
import zlib
from typing import Literal

from braidcodec._exceptions import FormatError

DomainKind = Literal["text", "json", "logs"]


def _compress_zlib(data: bytes) -> bytes:
    return zlib.compress(data, level=9)


def _compress_bz2(data: bytes) -> bytes:
    return bz2.compress(data, compresslevel=9)


def _compress_lzma(data: bytes) -> bytes:
    return lzma.compress(data, preset=9)


def _compress_by_codec(codec_name: str, data: bytes) -> bytes:
    if codec_name == "zlib-xor-v1":
        return _compress_zlib(data)
    if codec_name == "bz2-xor-v1":
        return _compress_bz2(data)
    if codec_name == "lzma-xor-v1":
        return _compress_lzma(data)
    raise FormatError("Unsupported latent residual codec")


def _decompress_by_codec(codec_name: str, data: bytes) -> bytes:
    if codec_name == "zlib-xor-v1":
        return zlib.decompress(data)
    if codec_name == "bz2-xor-v1":
        return bz2.decompress(data)
    if codec_name == "lzma-xor-v1":
        return lzma.decompress(data)
    raise FormatError("Unsupported latent residual codec")


def _build_spectral_predictor(
    length: int,
    *,
    coupling_density: float,
    coupling_spectral_radius: float,
    coupling_nnz: int,
) -> tuple[bytes, dict[str, int]]:
    """Build deterministic byte predictor stream from coupling features."""
    q_density = round(max(0.0, min(coupling_density, 1.0)) * 65535.0)
    q_radius = round(max(0.0, min(coupling_spectral_radius, 1_000_000.0)))
    q_nnz = round(max(float(coupling_nnz), 0.0) ** 0.5)

    seed = (q_density ^ ((q_radius * 131) & 0xFF) ^ (length & 0xFF) ^ (q_nnz & 0xFF)) & 0xFF
    a = ((q_density % 127) * 2 + 1) & 0xFF
    if a == 0:
        a = 1
    b = ((q_radius % 251) + 1 + (q_nnz % 29)) & 0xFF
    if b == 0:
        b = 1

    out = bytearray(length)
    if length > 0:
        out[0] = seed
    for i in range(1, length):
        out[i] = (a * out[i - 1] + b + (i & 0xFF)) & 0xFF

    return bytes(out), {"seed": seed, "a": a, "b": b, "nnz_q": q_nnz}


def _fit_best_segment_codec(
    source: bytes,
    *,
    coupling_density: float,
    coupling_spectral_radius: float,
    coupling_nnz: int,
) -> tuple[str, str, bytes, dict[str, int]]:
    """Return best (predictor, codec, compressed_residual, predictor_params)."""
    predictors: list[tuple[str, bytes, dict[str, int]]] = []

    predictors.append(("zero-v1", bytes(len(source)), {}))

    prev_pred = bytearray(len(source))
    for i in range(1, len(source)):
        prev_pred[i] = source[i - 1]
    predictors.append(("prev-byte-v1", bytes(prev_pred), {}))

    spectral_pred, spectral_params = _build_spectral_predictor(
        len(source),
        coupling_density=coupling_density,
        coupling_spectral_radius=coupling_spectral_radius,
        coupling_nnz=coupling_nnz,
    )
    predictors.append(("spectral-byte-v1", spectral_pred, spectral_params))

    codecs = ["zlib-xor-v1", "bz2-xor-v1", "lzma-xor-v1"]
    nnz_signal = coupling_nnz / max(len(source), 1)
    if coupling_density > 0.08 or coupling_spectral_radius > 25.0 or nnz_signal > 8.0:
        codecs = ["bz2-xor-v1", "lzma-xor-v1", "zlib-xor-v1"]

    best_predictor = "zero-v1"
    best_codec = "zlib-xor-v1"
    best_compressed = b""
    best_predictor_params: dict[str, int] = {}
    best_len: int | None = None

    for predictor_name, predicted, predictor_params in predictors:
        residual = bytes(s ^ p for s, p in zip(source, predicted, strict=False))
        for codec_name in codecs:
            compressed = _compress_by_codec(codec_name, residual)
            clen = len(compressed)
            if best_len is None or clen < best_len:
                best_len = clen
                best_predictor = predictor_name
                best_codec = codec_name
                best_compressed = compressed
                best_predictor_params = predictor_params

    return best_predictor, best_codec, best_compressed, best_predictor_params


def _decode_residual_with_predictor(
    *,
    predictor: str,
    residual: bytes,
    original_length: int,
    predictor_params: dict[str, object] | None = None,
) -> bytes:
    if predictor == "zero-v1":
        return residual

    if predictor == "spectral-byte-v1":
        params_obj = predictor_params or {}
        try:
            raw_seed = params_obj.get("seed", -1)
            raw_a = params_obj.get("a", -1)
            raw_b = params_obj.get("b", -1)
            if not isinstance(raw_seed, int | str):
                raise TypeError
            if not isinstance(raw_a, int | str):
                raise TypeError
            if not isinstance(raw_b, int | str):
                raise TypeError
            seed = int(raw_seed)
            a = int(raw_a)
            b = int(raw_b)
        except (TypeError, ValueError) as exc:
            raise FormatError("Invalid spectral predictor params") from exc
        if not (0 <= seed <= 255 and 0 <= a <= 255 and 0 <= b <= 255):
            raise FormatError("Invalid spectral predictor params")

        pred = bytearray(original_length)
        if original_length > 0:
            pred[0] = seed
        for i in range(1, original_length):
            pred[i] = (a * pred[i - 1] + b + (i & 0xFF)) & 0xFF
        return bytes(r ^ p for r, p in zip(residual, pred, strict=False))

    if predictor == "prev-byte-v1":
        out_buf = bytearray(original_length)
        if original_length > 0:
            out_buf[0] = residual[0]
        for i in range(1, original_length):
            out_buf[i] = residual[i] ^ out_buf[i - 1]
        return bytes(out_buf)

    raise FormatError("Unsupported latent residual predictor")


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


def fit_reconstructive_program(
    canonical_text: str,
    *,
    domain_kind: DomainKind,
    coupling_density: float | None = None,
    coupling_spectral_radius: float | None = None,
    coupling_nnz: int | None = None,
) -> dict[str, str]:
    """Fit a compact deterministic reconstruction program for canonical text."""
    def _latent_residual_program(raw_text: str, *, domain: DomainKind) -> dict[str, str]:
        source = raw_text.encode("utf-8")
        c_density = float(coupling_density or 0.0)
        c_radius = float(coupling_spectral_radius or 0.0)
        c_nnz = int(coupling_nnz or 0)

        (
            best_predictor,
            best_codec,
            best_compressed,
            best_predictor_params,
        ) = _fit_best_segment_codec(
            source,
            coupling_density=c_density,
            coupling_spectral_radius=c_radius,
            coupling_nnz=c_nnz,
        )

        payload_v2 = {
            "domain_kind": domain,
            "predictor": best_predictor,
            "codec": best_codec,
            "original_length": len(source),
            "predictor_params": best_predictor_params,
            "residual_b64": base64.b64encode(best_compressed).decode("ascii"),
        }

        # Segment-pack v3: allow heterogeneous predictor/codec per chunk.
        segments: list[dict[str, object]] = []
        segment_size = 32768 if len(source) >= 65536 else 16384
        for start in range(0, len(source), segment_size):
            chunk = source[start : start + segment_size]
            cp = c_density * (1.0 + ((start // max(segment_size, 1)) % 3) * 0.05)
            cr = c_radius * (1.0 + ((start // max(segment_size, 1)) % 2) * 0.03)
            cn = max(0, round(c_nnz * (len(chunk) / max(len(source), 1))))
            predictor, codec, compressed, params = _fit_best_segment_codec(
                chunk,
                coupling_density=cp,
                coupling_spectral_radius=cr,
                coupling_nnz=cn,
            )
            segments.append(
                {
                    "offset": start,
                    "length": len(chunk),
                    "predictor": predictor,
                    "codec": codec,
                    "predictor_params": params,
                    "residual_b64": base64.b64encode(compressed).decode("ascii"),
                }
            )

        payload_v3 = {
            "domain_kind": domain,
            "segment_size": segment_size,
            "original_length": len(source),
            "segments": segments,
        }

        payload_v2_json = json.dumps(payload_v2, sort_keys=True, separators=(",", ":"))
        payload_v3_json = json.dumps(payload_v3, sort_keys=True, separators=(",", ":"))

        if len(payload_v3_json) < len(payload_v2_json):
            return {
                "reconstructive_program_type": "latent-residual-v3",
                "reconstructive_program_payload": payload_v3_json,
            }

        return {
            "reconstructive_program_type": "latent-residual-v2",
            "reconstructive_program_payload": payload_v2_json,
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

    if program_type in {"latent-residual-v1", "latent-residual-v2"}:
        predictor = str(program.get("predictor", ""))
        codec = str(program.get("codec", ""))
        original_length = int(program.get("original_length", -1))
        residual_b64 = str(program.get("residual_b64", ""))
        if predictor not in {"zero-v1", "prev-byte-v1", "spectral-byte-v1"}:
            raise FormatError("Unsupported latent residual predictor")
        if codec not in {"zlib-xor-v1", "bz2-xor-v1", "lzma-xor-v1"}:
            raise FormatError("Unsupported latent residual codec")
        if original_length < 0 or not residual_b64:
            raise FormatError("Invalid latent residual program parameters")

        try:
            compressed = base64.b64decode(residual_b64.encode("ascii"))
            residual = _decompress_by_codec(codec, compressed)
        except Exception as exc:
            raise FormatError("Invalid latent residual payload encoding") from exc

        if len(residual) != original_length:
            raise FormatError("Latent residual length mismatch")

        params_obj_raw = program.get("predictor_params", {})
        params_obj = params_obj_raw if isinstance(params_obj_raw, dict) else {}
        return _decode_residual_with_predictor(
            predictor=predictor,
            residual=residual,
            original_length=original_length,
            predictor_params=params_obj,
        )

    if program_type == "latent-residual-v3":
        original_length = int(program.get("original_length", -1))
        segments_obj = program.get("segments", [])
        if original_length < 0 or not isinstance(segments_obj, list):
            raise FormatError("Invalid latent residual v3 parameters")

        out = bytearray(original_length)
        cursor = 0
        for segment in segments_obj:
            if not isinstance(segment, dict):
                raise FormatError("Invalid latent residual v3 segment")
            offset = int(segment.get("offset", -1))
            length = int(segment.get("length", -1))
            predictor = str(segment.get("predictor", ""))
            codec = str(segment.get("codec", ""))
            residual_b64 = str(segment.get("residual_b64", ""))
            params_obj_raw = segment.get("predictor_params", {})
            params_obj = params_obj_raw if isinstance(params_obj_raw, dict) else {}

            if offset != cursor or length < 0:
                raise FormatError("Invalid latent residual v3 segment layout")
            if not residual_b64:
                raise FormatError("Invalid latent residual v3 segment payload")

            try:
                compressed = base64.b64decode(residual_b64.encode("ascii"))
                residual = _decompress_by_codec(codec, compressed)
            except Exception as exc:
                raise FormatError("Invalid latent residual v3 segment encoding") from exc

            if len(residual) != length:
                raise FormatError("Latent residual v3 segment length mismatch")

            decoded = _decode_residual_with_predictor(
                predictor=predictor,
                residual=residual,
                original_length=length,
                predictor_params=params_obj,
            )
            out[offset : offset + length] = decoded
            cursor += length

        if cursor != original_length:
            raise FormatError("Latent residual v3 total length mismatch")
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
