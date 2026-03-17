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
from collections import Counter
from typing import TYPE_CHECKING, Literal

import msgpack

from braidcodec._exceptions import FormatError
from braidcodec.codec.braid_program_codec import parse_discovered_braid_program

if TYPE_CHECKING:
    from collections.abc import Mapping

DomainKind = Literal["text", "json", "logs"]
_PACK_PREFIX = "~mp85:"
_SIDEBAND_PACK_PREFIX = "~sp85:"
_PACK_BIN_COMPRESSED_PREFIX = b"mpz1:"
_PACK_BIN_RAW_PREFIX = b"mp1:"
_PREDICTOR_ENCODE: dict[str, str] = {"zero-v1": "z", "prev-byte-v1": "p", "spectral-byte-v1": "s"}
_PREDICTOR_DECODE: dict[str, str] = {v: k for k, v in _PREDICTOR_ENCODE.items()}
_CODEC_ENCODE: dict[str, str] = {
    "raw-xor-v1": "r",
    "zlib-xor-v1": "z",
    "bz2-xor-v1": "b",
    "lzma-xor-v1": "l",
    "lzma2raw-xor-v1": "x",
}
_CODEC_DECODE: dict[str, str] = {v: k for k, v in _CODEC_ENCODE.items()}
_DOMAIN_ENCODE: dict[str, str] = {"text": "t", "json": "j", "logs": "l"}
_DOMAIN_DECODE: dict[str, str] = {v: k for k, v in _DOMAIN_ENCODE.items()}


def _compress_zlib(data: bytes) -> bytes:
    return zlib.compress(data, level=9)


def _compress_bz2(data: bytes) -> bytes:
    return bz2.compress(data, compresslevel=9)


def _compress_lzma(data: bytes) -> bytes:
    return lzma.compress(data, preset=9)


_LZMA2RAW_FILTERS = [
    {
        "id": lzma.FILTER_LZMA2,
        "dict_size": 1 << 20,
        "lc": 4,
        "lp": 0,
        "pb": 0,
        "mode": lzma.MODE_NORMAL,
        "mf": lzma.MF_BT4,
        "nice_len": 273,
    }
]


def _compress_lzma2raw(data: bytes) -> bytes:
    return lzma.compress(data, format=lzma.FORMAT_RAW, filters=_LZMA2RAW_FILTERS)


def _compress_by_codec(codec_name: str, data: bytes) -> bytes:
    if codec_name == "raw-xor-v1":
        return data
    if codec_name == "zlib-xor-v1":
        return _compress_zlib(data)
    if codec_name == "bz2-xor-v1":
        return _compress_bz2(data)
    if codec_name == "lzma-xor-v1":
        return _compress_lzma(data)
    if codec_name == "lzma2raw-xor-v1":
        return _compress_lzma2raw(data)
    raise FormatError("Unsupported latent residual codec")


def _decompress_by_codec(codec_name: str, data: bytes) -> bytes:
    if codec_name == "raw-xor-v1":
        return data
    if codec_name == "zlib-xor-v1":
        return zlib.decompress(data)
    if codec_name == "bz2-xor-v1":
        return bz2.decompress(data)
    if codec_name == "lzma-xor-v1":
        return lzma.decompress(data)
    if codec_name == "lzma2raw-xor-v1":
        return lzma.decompress(data, format=lzma.FORMAT_RAW, filters=_LZMA2RAW_FILTERS)
    raise FormatError("Unsupported latent residual codec")


def _morton_key_1d(index: int, lane: int = 0) -> int:
    """Compute compact 1D Morton-like key for deterministic segment shaping."""
    x = max(index, 0)
    out = 0
    bit = 0
    while x > 0:
        out |= (x & 1) << (2 * bit)
        x >>= 1
        bit += 1
    return (out << 3) | (max(lane, 0) & 0x7)


def _encode_residual_blob(data: bytes) -> str:
    """Encode residual bytes with lower-overhead ASCII envelope."""
    return base64.b85encode(data).decode("ascii")


def _decode_residual_blob(program: dict[str, object]) -> bytes:
    """Decode residual bytes from payload (binary or base85 encoding)."""
    raw_bin = program.get("residual_bytes", program.get("rb"))
    if isinstance(raw_bin, bytes):
        return raw_bin
    if isinstance(raw_bin, bytearray):
        return bytes(raw_bin)

    raw_b85 = program.get("residual_b85", program.get("r85"))
    if isinstance(raw_b85, str) and raw_b85:
        try:
            return base64.b85decode(raw_b85.encode("ascii"))
        except Exception as exc:
            raise FormatError("Invalid latent residual base85 payload") from exc

    raise FormatError("Missing latent residual payload bytes")


def _serialize_program_payload(program: dict[str, object]) -> str:
    """Serialize reconstructive program payload with compact binary fallback."""
    json_payload = ""
    try:
        json_payload = json.dumps(program, sort_keys=True, separators=(",", ":"))
    except TypeError:
        # Program contains binary fields; packed path is required.
        json_payload = ""
    packed = msgpack.packb(program, use_bin_type=True)
    packed_blob = base64.b85encode(zlib.compress(packed, level=9)).decode("ascii")
    packed_payload = _PACK_PREFIX + packed_blob
    if not json_payload:
        return packed_payload
    return packed_payload if len(packed_payload) < len(json_payload) else json_payload


def serialize_reconstructive_program_sidechannel(program: dict[str, object]) -> bytes:
    """Serialize reconstructive program sidechannel for compact transport.

    The binary envelope avoids base85 overhead for large latent residual payloads.
    """
    packed = bytes(msgpack.packb(program, use_bin_type=True))
    compressed = bytes(zlib.compress(packed, level=9))
    if len(compressed) + len(_PACK_BIN_COMPRESSED_PREFIX) < (
        len(packed) + len(_PACK_BIN_RAW_PREFIX)
    ):
        return _PACK_BIN_COMPRESSED_PREFIX + compressed
    return _PACK_BIN_RAW_PREFIX + packed


def parse_reconstructive_program_payload(raw_payload: str | bytes) -> dict[str, object]:
    """Parse reconstructive program payload from JSON or packed base85 blob."""
    if isinstance(raw_payload, bytes):
        if raw_payload.startswith(_PACK_BIN_COMPRESSED_PREFIX):
            body = raw_payload[len(_PACK_BIN_COMPRESSED_PREFIX) :]
            try:
                decoded = msgpack.unpackb(zlib.decompress(body), raw=False)
            except Exception as exc:
                raise FormatError("Invalid compressed binary reconstructive payload") from exc
            if not isinstance(decoded, dict):
                raise FormatError("Binary reconstructive payload must decode to object")
            return {str(k): v for k, v in decoded.items()}

        if raw_payload.startswith(_PACK_BIN_RAW_PREFIX):
            body = raw_payload[len(_PACK_BIN_RAW_PREFIX) :]
            try:
                decoded = msgpack.unpackb(body, raw=False)
            except Exception as exc:
                raise FormatError("Invalid binary reconstructive payload") from exc
            if not isinstance(decoded, dict):
                raise FormatError("Binary reconstructive payload must decode to object")
            return {str(k): v for k, v in decoded.items()}

        # Legacy or non-prefixed payloads may still be UTF-8 text.
        try:
            raw_payload = raw_payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FormatError("Unsupported binary reconstructive payload encoding") from exc

    if raw_payload.startswith(_SIDEBAND_PACK_PREFIX):
        encoded = raw_payload[len(_SIDEBAND_PACK_PREFIX) :]
        try:
            raw_payload = zlib.decompress(base64.b85decode(encoded.encode("ascii"))).decode(
                "utf-8"
            )
        except Exception as exc:
            raise FormatError("Invalid packed sidechannel reconstructive program payload") from exc

    if raw_payload.startswith(_PACK_PREFIX):
        encoded = raw_payload[len(_PACK_PREFIX) :]
        try:
            packed = base64.b85decode(encoded.encode("ascii"))
            decoded = msgpack.unpackb(zlib.decompress(packed), raw=False)
        except Exception as exc:
            raise FormatError("Invalid packed reconstructive program payload") from exc
        if not isinstance(decoded, dict):
            raise FormatError("Packed reconstructive program payload must decode to object")
        return {str(k): v for k, v in decoded.items()}

    try:
        decoded_json = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise FormatError("Invalid reconstructive_program_payload JSON") from exc
    if not isinstance(decoded_json, dict):
        raise FormatError("reconstructive_program_payload must decode to object")
    return {str(k): v for k, v in decoded_json.items()}


def _coerce_int_field(value: object, *, field_name: str) -> int:
    """Coerce JSON/msgpack field to int with explicit validation."""
    if not isinstance(value, int | str):
        raise FormatError(f"Invalid {field_name} in reconstructive program")
    try:
        return int(value)
    except ValueError as exc:
        raise FormatError(f"Invalid {field_name} in reconstructive program") from exc


def _build_spectral_predictor(
    length: int,
    *,
    coupling_density: float,
    coupling_spectral_radius: float,
    coupling_nnz: int,
    morton_key: int,
) -> tuple[bytes, dict[str, int]]:
    """Build deterministic byte predictor stream from coupling features."""
    q_density = round(max(0.0, min(coupling_density, 1.0)) * 65535.0)
    q_radius = round(max(0.0, min(coupling_spectral_radius, 1_000_000.0)))
    q_nnz = round(max(float(coupling_nnz), 0.0) ** 0.5)
    q_morton = int(morton_key) & 0xFFFF

    seed = (
        q_density
        ^ ((q_radius * 131) & 0xFF)
        ^ (length & 0xFF)
        ^ (q_nnz & 0xFF)
        ^ (q_morton & 0xFF)
    ) & 0xFF
    a = (((q_density + (q_morton % 31)) % 127) * 2 + 1) & 0xFF
    if a == 0:
        a = 1
    b = ((q_radius % 251) + 1 + (q_nnz % 29) + ((q_morton >> 3) % 17)) & 0xFF
    if b == 0:
        b = 1

    out = bytearray(length)
    if length > 0:
        out[0] = seed
    for i in range(1, length):
        out[i] = (a * out[i - 1] + b + (i & 0xFF)) & 0xFF

    return bytes(out), {"seed": seed, "a": a, "b": b, "nnz_q": q_nnz, "morton_q": q_morton}


def _fit_best_segment_codec(
    source: bytes,
    *,
    coupling_density: float,
    coupling_spectral_radius: float,
    coupling_nnz: int,
    morton_key: int,
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
        morton_key=morton_key,
    )
    predictors.append(("spectral-byte-v1", spectral_pred, spectral_params))

    codecs = ["raw-xor-v1", "zlib-xor-v1", "bz2-xor-v1", "lzma-xor-v1", "lzma2raw-xor-v1"]
    nnz_signal = coupling_nnz / max(len(source), 1)
    if coupling_density > 0.08 or coupling_spectral_radius > 25.0 or nnz_signal > 8.0:
        codecs = ["bz2-xor-v1", "lzma2raw-xor-v1", "lzma-xor-v1", "zlib-xor-v1"]
    if (morton_key & 1) == 1:
        codecs = [codecs[1], codecs[0], codecs[2]]

    best_predictor = "zero-v1"
    best_codec = "zlib-xor-v1"
    best_compressed = b""
    best_predictor_params: dict[str, int] = {}
    best_wire_len: int | None = None

    for predictor_name, predicted, predictor_params in predictors:
        residual = bytes(s ^ p for s, p in zip(source, predicted, strict=False))
        for codec_name in codecs:
            compressed = _compress_by_codec(codec_name, residual)
            candidate_payload: dict[str, object] = {
                "o": 0,
                "l": len(source),
                "p": _PREDICTOR_ENCODE.get(predictor_name, predictor_name),
                "c": _CODEC_ENCODE.get(codec_name, codec_name),
                "r85": _encode_residual_blob(compressed),
            }
            if predictor_params:
                candidate_payload["pp"] = predictor_params
            wire_len = len(json.dumps(candidate_payload, sort_keys=True, separators=(",", ":")))

            if best_wire_len is None or wire_len < best_wire_len:
                best_wire_len = wire_len
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


def _tokenize_text_pieces(text: str) -> list[str]:
    """Split text into reversible token pieces (words/whitespace/punctuation)."""
    if not text:
        return []
    return re.findall(r"\w+|\s+|[^\w\s]", text)


def _fit_token_delta_program(text: str) -> dict[str, str] | None:
    """Build token-delta program for recurring token transitions.

    Returns None when token vocabulary does not provide useful reuse.
    """
    pieces = _tokenize_text_pieces(text)
    if len(pieces) < 8:
        return None

    token_counts = Counter(
        p for p in pieces if p and p[0].isalnum() and len(p) >= 3 and p.strip() == p
    )
    vocab = [t for t, c in token_counts.most_common(96) if c >= 3]
    if len(vocab) < 8:
        return None

    vocab_index = {token: i for i, token in enumerate(vocab)}
    delta_stream: list[int | None] = []
    literals: list[str] = []
    prev_idx = 0

    for piece in pieces:
        idx = vocab_index.get(piece)
        if idx is None:
            delta_stream.append(None)
            literals.append(piece)
            continue
        delta_stream.append(idx - prev_idx)
        prev_idx = idx

    payload = {
        "v": vocab,
        "d": delta_stream,
        "l": literals,
    }
    return {
        "reconstructive_program_type": "token-delta-grammar-v1",
        "reconstructive_program_payload": _serialize_program_payload(payload),
    }


def _synthesize_token_delta_program(program: dict[str, object]) -> bytes:
    """Decode token-delta grammar payload back to original text bytes."""
    raw_vocab = program.get("v", [])
    raw_deltas = program.get("d", [])
    raw_literals = program.get("l", [])
    if (
        not isinstance(raw_vocab, list)
        or not isinstance(raw_deltas, list)
        or not isinstance(raw_literals, list)
    ):
        raise FormatError("Invalid token-delta grammar payload")

    vocab = [str(v) for v in raw_vocab]
    literals = [str(v) for v in raw_literals]
    literal_cursor = 0
    prev_idx = 0
    out: list[str] = []

    for item in raw_deltas:
        if item is None:
            if literal_cursor >= len(literals):
                raise FormatError("token-delta literal stream underflow")
            out.append(literals[literal_cursor])
            literal_cursor += 1
            continue

        if not isinstance(item, int | str):
            raise FormatError("Invalid token-delta delta value")
        delta = int(item)
        idx = prev_idx + delta
        if idx < 0 or idx >= len(vocab):
            raise FormatError("token-delta index out of range")
        out.append(vocab[idx])
        prev_idx = idx

    if literal_cursor != len(literals):
        raise FormatError("token-delta literal stream overflow")
    return "".join(out).encode("utf-8")


def _fit_phrase_dictionary_program(text: str) -> dict[str, str] | None:
    """Build a phrase-dictionary recurrence program for repeated n-grams."""
    tokens = re.findall(r"\S+\s*", text)
    if len(tokens) < 16:
        return None

    counts: Counter[tuple[str, ...]] = Counter()
    for n in (2, 3, 4):
        for i in range(0, len(tokens) - n + 1):
            phrase = tuple(tokens[i : i + n])
            phrase_bytes = "".join(phrase).encode("utf-8")
            if len(phrase_bytes) >= 12:
                counts[phrase] += 1

    phrases = [p for p, c in counts.most_common(48) if c >= 3]
    if not phrases:
        return None

    # Keep longer and more frequent phrases first for greedy packing.
    phrases = sorted(
        phrases,
        key=lambda p: (-(len(p)), -counts[p], "".join(p)),
    )
    phrase_strings = ["".join(p) for p in phrases]

    marks: list[int] = []
    literals: list[str] = []
    i = 0
    while i < len(tokens):
        best_idx = -1
        best_len = 0
        for p_idx, phrase in enumerate(phrases):
            plen = len(phrase)
            if plen <= best_len:
                continue
            if i + plen <= len(tokens) and tuple(tokens[i : i + plen]) == phrase:
                best_idx = p_idx
                best_len = plen
        if best_idx >= 0:
            marks.append(best_idx)
            i += best_len
            continue

        marks.append(-1)
        literals.append(tokens[i])
        i += 1

    payload = {
        "p": phrase_strings,
        "m": marks,
        "l": literals,
    }
    return {
        "reconstructive_program_type": "phrase-dictionary-v1",
        "reconstructive_program_payload": _serialize_program_payload(payload),
    }


def _synthesize_phrase_dictionary_program(program: dict[str, object]) -> bytes:
    """Decode phrase dictionary payload back to original text bytes."""
    raw_phrases = program.get("p", [])
    raw_marks = program.get("m", [])
    raw_literals = program.get("l", [])
    if (
        not isinstance(raw_phrases, list)
        or not isinstance(raw_marks, list)
        or not isinstance(raw_literals, list)
    ):
        raise FormatError("Invalid phrase dictionary payload")

    phrases = [str(v) for v in raw_phrases]
    literals = [str(v) for v in raw_literals]
    literal_cursor = 0
    out: list[str] = []

    for marker in raw_marks:
        if not isinstance(marker, int | str):
            raise FormatError("Invalid phrase dictionary marker")
        idx = int(marker)
        if idx == -1:
            if literal_cursor >= len(literals):
                raise FormatError("phrase dictionary literal stream underflow")
            out.append(literals[literal_cursor])
            literal_cursor += 1
            continue
        if idx < 0 or idx >= len(phrases):
            raise FormatError("phrase dictionary index out of range")
        out.append(phrases[idx])

    if literal_cursor != len(literals):
        raise FormatError("phrase dictionary literal stream overflow")
    return "".join(out).encode("utf-8")


def _fit_sparse_corrective_program(
    *,
    source_text: str,
    base_program: dict[str, str],
) -> dict[str, str] | None:
    """Build sparse corrective residual over a generative base program.

    The generated payload stores sparse XOR patches against the base output.
    """
    source = source_text.encode("utf-8")
    base = synthesize_reconstructive_bytes(base_program)
    if len(base) != len(source):
        return None

    mismatch = [i for i, (a, b) in enumerate(zip(source, base, strict=False)) if a != b]
    if not mismatch:
        return None

    # Sparse residuals only: skip if mismatch density is too high.
    if len(mismatch) / max(len(source), 1) > 0.35:
        return None

    patches: list[dict[str, object]] = []
    start = mismatch[0]
    prev = start
    for idx in mismatch[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        chunk = bytes(source[i] ^ base[i] for i in range(start, prev + 1))
        patches.append({"o": start, "x85": _encode_residual_blob(chunk)})
        start = idx
        prev = idx
    chunk = bytes(source[i] ^ base[i] for i in range(start, prev + 1))
    patches.append({"o": start, "x85": _encode_residual_blob(chunk)})

    payload = {
        "n": len(source),
        "bt": base_program.get("reconstructive_program_type", ""),
        "bp": base_program.get("reconstructive_program_payload", ""),
        "px": patches,
    }
    return {
        "reconstructive_program_type": "sparse-corrective-v1",
        "reconstructive_program_payload": _serialize_program_payload(payload),
    }


def _synthesize_sparse_corrective_program(program: dict[str, object]) -> bytes:
    """Decode sparse corrective payload by applying XOR patches to base output."""
    raw_len = program.get("n", -1)
    base_type = str(program.get("bt", ""))
    base_payload = str(program.get("bp", ""))
    patches_obj = program.get("px", [])
    if not isinstance(raw_len, int | str):
        raise FormatError("Invalid sparse corrective payload length")
    target_len = int(raw_len)
    if target_len < 0:
        raise FormatError("Invalid sparse corrective payload length")
    if not base_type or not base_payload:
        raise FormatError("Invalid sparse corrective base program")
    if not isinstance(patches_obj, list):
        raise FormatError("Invalid sparse corrective patch payload")

    base_bytes = synthesize_reconstructive_bytes(
        {
            "reconstructive_program_type": base_type,
            "reconstructive_program_payload": base_payload,
        }
    )
    if len(base_bytes) != target_len:
        raise FormatError("Sparse corrective base length mismatch")

    out = bytearray(base_bytes)
    for patch in patches_obj:
        if not isinstance(patch, dict):
            raise FormatError("Invalid sparse corrective patch entry")
        offset = _coerce_int_field(patch.get("o", -1), field_name="patch offset")
        xor_blob = patch.get("x85", "")
        if not isinstance(xor_blob, str) or not xor_blob:
            raise FormatError("Invalid sparse corrective patch data")
        try:
            delta = base64.b85decode(xor_blob.encode("ascii"))
        except Exception as exc:
            raise FormatError("Invalid sparse corrective patch encoding") from exc
        end = offset + len(delta)
        if offset < 0 or end > len(out):
            raise FormatError("Sparse corrective patch out of bounds")
        for i, value in enumerate(delta):
            out[offset + i] ^= value

    return bytes(out)


def _select_smallest_exact_program(
    *,
    source_text: str,
    candidates: list[dict[str, str]],
) -> dict[str, str]:
    """Select smallest candidate that reconstructs source exactly."""
    source_bytes = source_text.encode("utf-8")
    best: dict[str, str] | None = None
    best_len: int | None = None

    for candidate in candidates:
        try:
            reconstructed = synthesize_reconstructive_bytes(candidate)
        except Exception:
            continue
        if reconstructed != source_bytes:
            continue
        payload_len = len(candidate.get("reconstructive_program_payload", "").encode("utf-8"))
        if best is None or best_len is None or payload_len < best_len:
            best = candidate
            best_len = payload_len

    if best is None:
        raise FormatError("No exact reconstructive program candidate available")
    return best


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
            morton_key=0,
        )

        payload_v2: dict[str, object] = {
            "d": _DOMAIN_ENCODE.get(domain, domain),
            "p": _PREDICTOR_ENCODE.get(best_predictor, best_predictor),
            "c": _CODEC_ENCODE.get(best_codec, best_codec),
            "n": len(source),
            "rb": best_compressed,
        }
        if best_predictor_params:
            payload_v2["pp"] = best_predictor_params

        payload_v2_legacy: dict[str, object] = {
            "d": _DOMAIN_ENCODE.get(domain, domain),
            "p": _PREDICTOR_ENCODE.get(best_predictor, best_predictor),
            "c": _CODEC_ENCODE.get(best_codec, best_codec),
            "n": len(source),
            "r85": _encode_residual_blob(best_compressed),
        }
        if best_predictor_params:
            payload_v2_legacy["pp"] = best_predictor_params

        payload_v2_serialized_bin = _serialize_program_payload(payload_v2)
        payload_v2_serialized_legacy = _serialize_program_payload(payload_v2_legacy)
        payload_v2_serialized = (
            payload_v2_serialized_bin
            if len(payload_v2_serialized_bin) <= len(payload_v2_serialized_legacy)
            else payload_v2_serialized_legacy
        )

        # Segment-pack v3: evaluate multiple chunk granularities and pick the
        # smallest final serialized payload.
        segment_candidates: list[int]
        if len(source) <= 2048:
            segment_candidates = [len(source)]
        else:
            segment_candidates = [s for s in (2048, 4096, 8192, 16384, 32768) if s < len(source)]
            segment_candidates.append(len(source))

        best_v3_payload: dict[str, object] | None = None
        best_v3_serialized = ""

        for segment_size in segment_candidates:
            segments: list[dict[str, object]] = []
            segments_legacy: list[dict[str, object]] = []
            for start in range(0, len(source), segment_size):
                segment_index = start // max(segment_size, 1)
                morton_key = _morton_key_1d(segment_index, lane=(c_nnz & 0x7))
                chunk = source[start : start + segment_size]
                cp = c_density * (1.0 + (morton_key & 0x3) * 0.03)
                cr = c_radius * (1.0 + ((morton_key >> 2) & 0x3) * 0.02)
                cn = max(0, round(c_nnz * (len(chunk) / max(len(source), 1))))
                cn = max(0, round(cn * (1.0 + (((morton_key & 0x7) - 3) * 0.02))))
                predictor, codec, compressed, params = _fit_best_segment_codec(
                    chunk,
                    coupling_density=cp,
                    coupling_spectral_radius=cr,
                    coupling_nnz=cn,
                    morton_key=morton_key,
                )
                segment: dict[str, object] = {
                    "o": start,
                    "l": len(chunk),
                    "p": _PREDICTOR_ENCODE.get(predictor, predictor),
                    "c": _CODEC_ENCODE.get(codec, codec),
                    "rb": compressed,
                }
                segment_legacy: dict[str, object] = {
                    "o": start,
                    "l": len(chunk),
                    "p": _PREDICTOR_ENCODE.get(predictor, predictor),
                    "c": _CODEC_ENCODE.get(codec, codec),
                    "r85": _encode_residual_blob(compressed),
                }
                if params:
                    segment["pp"] = params
                    segment_legacy["pp"] = params
                segments.append(segment)
                segments_legacy.append(segment_legacy)

            payload_v3: dict[str, object] = {
                "d": _DOMAIN_ENCODE.get(domain, domain),
                "ss": segment_size,
                "n": len(source),
                "s": segments,
            }
            payload_v3_legacy: dict[str, object] = {
                "d": _DOMAIN_ENCODE.get(domain, domain),
                "ss": segment_size,
                "n": len(source),
                "s": segments_legacy,
            }

            serialized_bin = _serialize_program_payload(payload_v3)
            serialized_legacy = _serialize_program_payload(payload_v3_legacy)
            if len(serialized_bin) <= len(serialized_legacy):
                serialized = serialized_bin
                selected_payload = payload_v3
            else:
                serialized = serialized_legacy
                selected_payload = payload_v3_legacy

            if not best_v3_serialized or len(serialized) < len(best_v3_serialized):
                best_v3_serialized = serialized
                best_v3_payload = selected_payload

        if best_v3_payload is not None and len(best_v3_serialized) < len(payload_v2_serialized):
            return {
                "reconstructive_program_type": "latent-residual-v3",
                "reconstructive_program_payload": best_v3_serialized,
            }

        return {
            "reconstructive_program_type": "latent-residual-v2",
            "reconstructive_program_payload": payload_v2_serialized,
        }

    if domain_kind == "text":
        candidates: list[dict[str, str]] = []

        rep = _smallest_repeat_unit(canonical_text)
        if rep is not None:
            unit, count = rep
            # Avoid degenerate full-literal replay where count=1 and unit==full text.
            if count > 1 and len(unit.encode("utf-8")) < len(canonical_text.encode("utf-8")):
                unit_b64 = base64.b64encode(unit.encode("utf-8")).decode("ascii")
                candidates.append(
                    {
                        "reconstructive_program_type": "repeat-text-v1",
                        "reconstructive_program_payload": _serialize_program_payload(
                            {
                                "unit_b64": unit_b64,
                                "repeat_count": count,
                            }
                        ),
                    }
                )

        token_delta_candidate = _fit_token_delta_program(canonical_text)
        if token_delta_candidate is not None:
            candidates.append(token_delta_candidate)

        phrase_candidate = _fit_phrase_dictionary_program(canonical_text)
        if phrase_candidate is not None:
            candidates.append(phrase_candidate)

        latent_candidate = _latent_residual_program(canonical_text, domain=domain_kind)
        candidates.append(latent_candidate)

        sparse_candidates: list[dict[str, str]] = []
        for base_candidate in candidates:
            sparse = _fit_sparse_corrective_program(
                source_text=canonical_text,
                base_program=base_candidate,
            )
            if sparse is not None:
                sparse_candidates.append(sparse)
        candidates.extend(sparse_candidates)

        return _select_smallest_exact_program(source_text=canonical_text, candidates=candidates)

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
            "reconstructive_program_payload": _serialize_program_payload(
                {
                    "count": len(items),
                    "json_style": json_style,
                }
            ),
        }

    if domain_kind == "logs":
        lines = canonical_text.split("\n")
        if not lines:
            return _latent_residual_program(canonical_text, domain=domain_kind)

        pattern = re.compile(r"^(.*:)(\d+)Z INFO core event=(\d+)$")
        m0 = pattern.match(lines[0])
        if not m0:
            return _latent_residual_program(canonical_text, domain=domain_kind)

        prefix = m0.group(1)
        width = len(m0.group(2))

        for i, line in enumerate(lines):
            m = pattern.match(line)
            if not m:
                return _latent_residual_program(canonical_text, domain=domain_kind)
            if m.group(1) != prefix:
                return _latent_residual_program(canonical_text, domain=domain_kind)
            sec = int(m.group(2))
            event = int(m.group(3))
            if sec != i or event != i:
                return _latent_residual_program(canonical_text, domain=domain_kind)

        return {
            "reconstructive_program_type": "logs-seq-v1",
            "reconstructive_program_payload": _serialize_program_payload(
                {
                    "prefix": prefix,
                    "count": len(lines),
                    "width": width,
                }
            ),
        }

    raise FormatError("Unsupported reconstructive domain")


def _synthesize_discovered_equation_bytes(
    *,
    equation_family: str,
    coefficients: list[int],
    initial_state: list[int],
    rollout_length: int,
) -> bytes:
    """Synthesize bytes for discovered equation families."""
    out = bytearray(rollout_length)
    if rollout_length == 0:
        return b""
    if not initial_state:
        raise FormatError("Discovered equation initial_state must be non-empty")

    x0 = initial_state[0] & 0xFF
    out[0] = x0

    if equation_family == "byte-constant-v1":
        for i in range(1, rollout_length):
            out[i] = x0
        return bytes(out)

    if equation_family == "byte-linear-mod-v1":
        if len(coefficients) != 1:
            raise FormatError("Invalid byte-linear-mod-v1 coefficient count")
        step = coefficients[0] & 0xFF
        for i in range(1, rollout_length):
            out[i] = (out[i - 1] + step) & 0xFF
        return bytes(out)

    if equation_family == "byte-xor-step-v1":
        if len(coefficients) != 1:
            raise FormatError("Invalid byte-xor-step-v1 coefficient count")
        step = coefficients[0] & 0xFF
        for i in range(1, rollout_length):
            out[i] = out[i - 1] ^ step
        return bytes(out)

    if equation_family == "byte-affine-recursion-v1":
        if len(coefficients) != 2:
            raise FormatError("Invalid byte-affine-recursion-v1 coefficient count")
        a = coefficients[0] & 0xFF
        b = coefficients[1] & 0xFF
        for i in range(1, rollout_length):
            out[i] = (a * out[i - 1] + b) & 0xFF
        return bytes(out)

    raise FormatError("Unsupported discovered equation family")


def synthesize_reconstructive_bytes(payload: Mapping[str, str | bytes]) -> bytes:
    """Synthesize canonical bytes from compact reconstructive program payload."""
    program_type = payload.get("reconstructive_program_type", "")
    raw_program_obj = payload.get("reconstructive_program_payload", "")
    raw_program = raw_program_obj if isinstance(raw_program_obj, str | bytes) else ""

    if not raw_program:
        raise FormatError("Missing reconstructive_program_payload")

    program = parse_reconstructive_program_payload(raw_program)

    if program_type == "discovered-equation-v1":
        equation_family = str(program.get("equation_family", ""))
        coeffs_obj = program.get("coefficients", [])
        initial_obj = program.get("initial_state", [])
        rollout_length = _coerce_int_field(
            program.get("rollout_length", -1), field_name="rollout_length"
        )

        if not isinstance(coeffs_obj, list) or not isinstance(initial_obj, list):
            raise FormatError("Invalid discovered equation payload")
        if rollout_length < 0:
            raise FormatError("Invalid discovered equation rollout length")

        try:
            coefficients = [int(v) for v in coeffs_obj]
            initial_state = [int(v) for v in initial_obj]
        except (TypeError, ValueError) as exc:
            raise FormatError("Invalid discovered equation numeric payload") from exc

        return _synthesize_discovered_equation_bytes(
            equation_family=equation_family,
            coefficients=coefficients,
            initial_state=initial_state,
            rollout_length=rollout_length,
        )

    if program_type == "discovered-braid-equation-v1":
        parsed = parse_discovered_braid_program(program)
        return _synthesize_discovered_equation_bytes(
            equation_family=parsed.equation_family,
            coefficients=parsed.coefficients,
            initial_state=parsed.initial_state,
            rollout_length=parsed.rollout_length,
        )

    if program_type == "repeat-text-v1":
        unit_b64 = str(program.get("unit_b64", ""))
        repeat_count = _coerce_int_field(
            program.get("repeat_count", -1),
            field_name="repeat_count",
        )
        if repeat_count < 0:
            raise FormatError("Invalid repeat_count in reconstructive program")
        try:
            unit = base64.b64decode(unit_b64.encode("ascii")).decode("utf-8")
        except Exception as exc:
            raise FormatError("Invalid unit_b64 in reconstructive program") from exc
        return (unit * repeat_count).encode("utf-8")

    if program_type == "token-delta-grammar-v1":
        return _synthesize_token_delta_program(program)

    if program_type == "phrase-dictionary-v1":
        return _synthesize_phrase_dictionary_program(program)

    if program_type == "sparse-corrective-v1":
        return _synthesize_sparse_corrective_program(program)

    if program_type == "json-linear-items-v1":
        count = _coerce_int_field(program.get("count", -1), field_name="count")
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
        predictor = str(program.get("predictor", program.get("p", "")))
        codec = str(program.get("codec", program.get("c", "")))
        predictor = _PREDICTOR_DECODE.get(predictor, predictor)
        codec = _CODEC_DECODE.get(codec, codec)
        original_length = _coerce_int_field(
            program.get("original_length", program.get("n", -1)),
            field_name="original_length",
        )
        if predictor not in {"zero-v1", "prev-byte-v1", "spectral-byte-v1"}:
            raise FormatError("Unsupported latent residual predictor")
        if codec not in {
            "raw-xor-v1",
            "zlib-xor-v1",
            "bz2-xor-v1",
            "lzma-xor-v1",
            "lzma2raw-xor-v1",
        }:
            raise FormatError("Unsupported latent residual codec")
        if original_length < 0:
            raise FormatError("Invalid latent residual program parameters")

        try:
            compressed = _decode_residual_blob(program)
            residual = _decompress_by_codec(codec, compressed)
        except Exception as exc:
            raise FormatError("Invalid latent residual payload encoding") from exc

        if len(residual) != original_length:
            raise FormatError("Latent residual length mismatch")

        params_obj_raw = program.get("predictor_params", program.get("pp", {}))
        params_obj = params_obj_raw if isinstance(params_obj_raw, dict) else {}
        return _decode_residual_with_predictor(
            predictor=predictor,
            residual=residual,
            original_length=original_length,
            predictor_params=params_obj,
        )

    if program_type == "latent-residual-v3":
        original_length = _coerce_int_field(
            program.get("original_length", program.get("n", -1)),
            field_name="original_length",
        )
        segments_obj = program.get("segments", program.get("s", []))
        if original_length < 0 or not isinstance(segments_obj, list):
            raise FormatError("Invalid latent residual v3 parameters")

        out = bytearray(original_length)
        cursor = 0
        for segment in segments_obj:
            if not isinstance(segment, dict):
                raise FormatError("Invalid latent residual v3 segment")
            raw_offset = segment.get("offset", segment.get("o", -1))
            raw_length = segment.get("length", segment.get("l", -1))
            if not isinstance(raw_offset, int | str) or not isinstance(raw_length, int | str):
                raise FormatError("Invalid latent residual v3 segment layout")
            offset = _coerce_int_field(raw_offset, field_name="segment offset")
            length = _coerce_int_field(raw_length, field_name="segment length")
            predictor = str(segment.get("predictor", segment.get("p", "")))
            codec = str(segment.get("codec", segment.get("c", "")))
            predictor = _PREDICTOR_DECODE.get(predictor, predictor)
            codec = _CODEC_DECODE.get(codec, codec)
            params_obj_raw = segment.get("predictor_params", segment.get("pp", {}))
            params_obj = params_obj_raw if isinstance(params_obj_raw, dict) else {}

            if offset != cursor or length < 0:
                raise FormatError("Invalid latent residual v3 segment layout")

            try:
                compressed = _decode_residual_blob(segment)
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
        count = _coerce_int_field(program.get("count", -1), field_name="count")
        width = _coerce_int_field(program.get("width", -1), field_name="width")
        if count < 0 or width < 1:
            raise FormatError("Invalid logs program parameters")
        lines = [f"{prefix}{str(i).zfill(width)}Z INFO core event={i}" for i in range(count)]
        return "\n".join(lines).encode("utf-8")

    raise FormatError("Unsupported reconstructive program type")
