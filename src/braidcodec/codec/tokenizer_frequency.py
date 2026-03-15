"""Deterministic frequency-domain tokenizer for reconstructive mode.

Phase B introduces a canonical tokenizer for Text/JSON/Logs with a fixed
frequency-bin lattice and stable metadata hashes. This module does not yet
perform manifold fitting; it defines the deterministic front-end contract.
"""

from __future__ import annotations

import json
import re
import struct
import unicodedata
from dataclasses import dataclass
from typing import Literal

import blake3

from braidcodec._exceptions import FormatError
from braidcodec.codec.oscillator_normalization import (
    NORMALIZATION_PROFILE_ID,
    normalization_profile_hash,
)

DomainKind = Literal["text", "json", "logs"]

TOKENIZER_ID = "frequency-tokenizer"
TOKENIZER_VERSION = "v1"
BIN_COUNT = 128
FREQ_LOW_HZ = 110.0
FREQ_HIGH_HZ = 7040.0


@dataclass(frozen=True, slots=True)
class FrequencyToken:
    """One deterministic token projected to a frequency bin."""

    domain: DomainKind
    token_class: str
    value: str
    bin_index: int
    frequency_hz: float
    phoneme_tag: str | None = None


@dataclass(frozen=True, slots=True)
class TokenizationResult:
    """Tokenizer output payload for reconstructive mode pre-processing."""

    domain: DomainKind
    canonical_text: str
    tokens: tuple[FrequencyToken, ...]
    metadata: dict[str, str]


def _build_bin_table(
    *,
    count: int = BIN_COUNT,
    low_hz: float = FREQ_LOW_HZ,
    high_hz: float = FREQ_HIGH_HZ,
) -> tuple[float, ...]:
    if count < 2:
        raise ValueError("count must be >= 2")
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("frequency bounds must satisfy 0 < low < high")

    ratio = high_hz / low_hz
    table = [low_hz * (ratio ** (i / (count - 1))) for i in range(count)]
    return tuple(table)


_BIN_TABLE = _build_bin_table()
_BIN_TABLE_HASH = blake3.blake3(
    b"".join(struct.pack(">d", x) for x in _BIN_TABLE)
).hexdigest()


def _phoneme_tag(value: str) -> str | None:
    letters = [c for c in value.lower() if c.isalpha()]
    if not letters:
        return None
    vowels = set("aeiou")
    if all(ch in vowels for ch in letters):
        return "vowel"
    if all(ch not in vowels for ch in letters):
        return "consonant"
    return "mixed"


def _bin_index(domain: DomainKind, token_class: str, value: str, phoneme: str | None) -> int:
    payload = f"{TOKENIZER_VERSION}|{domain}|{token_class}|{phoneme or '-'}|{value}".encode()
    digest = blake3.blake3(payload).digest(length=8)
    return int.from_bytes(digest, byteorder="big", signed=False) % BIN_COUNT


def _split_text_tokens(value: str) -> list[tuple[str, str]]:
    parts = re.findall(r"\w+|\s+|[^\w\s]", value, flags=re.UNICODE)
    tokens: list[tuple[str, str]] = []
    for part in parts:
        if part.isspace():
            tokens.append(("whitespace", part))
        elif part.isalnum() or part.replace("_", "").isalnum():
            tokens.append(("word", part))
        else:
            tokens.append(("punct", part))
    return tokens


def _scan_json_tokens(value: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    n = len(value)
    while i < n:
        ch = value[i]
        if ch in "{}[]:,":
            tokens.append(("json-struct", ch))
            i += 1
            continue

        if ch == '"':
            j = i + 1
            escaped = False
            while j < n:
                c = value[j]
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    break
                j += 1
            if j >= n:
                raise FormatError("Malformed canonical JSON string token")
            literal = value[i : j + 1]
            tokens.append(("json-string", literal))
            i = j + 1
            continue

        if ch == "-" or ch.isdigit():
            j = i + 1
            while j < n and value[j] in "0123456789eE+.-":
                j += 1
            tokens.append(("json-number", value[i:j]))
            i = j
            continue

        if value.startswith("true", i):
            tokens.append(("json-literal", "true"))
            i += 4
            continue
        if value.startswith("false", i):
            tokens.append(("json-literal", "false"))
            i += 5
            continue
        if value.startswith("null", i):
            tokens.append(("json-literal", "null"))
            i += 4
            continue

        raise FormatError("Malformed canonical JSON token stream", offset=i)

    return tokens


def _split_log_tokens(value: str) -> list[tuple[str, str]]:
    parts = re.findall(
        r"\d{4}-\d{2}-\d{2}[T ][0-9:.+-Z]*|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_.-]*|\d+|\s+|[^\w\s]",
        value,
        flags=re.UNICODE,
    )
    tokens: list[tuple[str, str]] = []
    for part in parts:
        if part.isspace():
            tokens.append(("log-whitespace", part))
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ][0-9:.+-Z]*", part):
            tokens.append(("log-timestamp", part))
        elif part.startswith("[") and part.endswith("]"):
            tokens.append(("log-bracket", part))
        elif part.isdigit():
            tokens.append(("log-number", part))
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", part):
            tokens.append(("log-ident", part))
        else:
            tokens.append(("log-punct", part))
    return tokens


def _canonicalize_text(raw: str | bytes) -> str:
    text: str
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FormatError("Text domain requires UTF-8 input", reason=str(exc)) from exc
    else:
        text = str(raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def _canonicalize_json(raw: str | bytes) -> str:
    raw_text: str
    if isinstance(raw, bytes):
        try:
            raw_text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FormatError("JSON domain requires UTF-8 input", reason=str(exc)) from exc
    else:
        raw_text = str(raw)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise FormatError("Invalid JSON input for tokenizer", reason=str(exc)) from exc

    return json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonicalize_logs(raw: str | bytes) -> str:
    text = _canonicalize_text(raw)
    # Keep semantic whitespace in log messages; only normalize line endings.
    return text


def _token_entries(domain: DomainKind, canonical_text: str) -> list[tuple[str, str]]:
    if domain == "text":
        return _split_text_tokens(canonical_text)
    if domain == "json":
        return _scan_json_tokens(canonical_text)
    if domain == "logs":
        return _split_log_tokens(canonical_text)
    raise FormatError("Unsupported tokenizer domain", domain=domain)


def _vocab_hash(entries: list[tuple[str, str]]) -> str:
    uniq = sorted({f"{token_class}:{value}" for token_class, value in entries})
    payload = "\n".join(uniq).encode("utf-8")
    return blake3.blake3(payload).hexdigest()


class FrequencyTokenizerV1:
    """Deterministic tokenizer with fixed frequency bin mapping."""

    tokenizer_id = TOKENIZER_ID
    tokenizer_version = TOKENIZER_VERSION
    bin_count = BIN_COUNT
    normalization_profile_id = NORMALIZATION_PROFILE_ID

    def tokenize(self, raw: str | bytes, *, domain: DomainKind) -> TokenizationResult:
        if domain == "text":
            canonical = _canonicalize_text(raw)
        elif domain == "json":
            canonical = _canonicalize_json(raw)
        elif domain == "logs":
            canonical = _canonicalize_logs(raw)
        else:
            raise FormatError("Unsupported tokenizer domain", domain=domain)

        entries = _token_entries(domain, canonical)
        tokens: list[FrequencyToken] = []
        for token_class, value in entries:
            phoneme = _phoneme_tag(value) if domain == "text" and token_class == "word" else None
            idx = _bin_index(domain, token_class, value, phoneme)
            tokens.append(
                FrequencyToken(
                    domain=domain,
                    token_class=token_class,
                    value=value,
                    bin_index=idx,
                    frequency_hz=_BIN_TABLE[idx],
                    phoneme_tag=phoneme,
                )
            )

        metadata = {
            "tokenizer_id": TOKENIZER_ID,
            "tokenizer_version": TOKENIZER_VERSION,
            "domain_kind": domain,
            "bin_count": str(BIN_COUNT),
            "bin_table_hash": _BIN_TABLE_HASH,
            "vocab_hash": _vocab_hash(entries),
            "normalization_profile_id": NORMALIZATION_PROFILE_ID,
            "normalization_profile_hash": normalization_profile_hash(),
        }

        return TokenizationResult(
            domain=domain,
            canonical_text=canonical,
            tokens=tuple(tokens),
            metadata=metadata,
        )


def reconstructive_metadata_keys() -> frozenset[str]:
    """Required metadata keys emitted by FrequencyTokenizerV1."""
    return frozenset(
        {
            "tokenizer_id",
            "tokenizer_version",
            "domain_kind",
            "bin_count",
            "bin_table_hash",
            "vocab_hash",
            "normalization_profile_id",
            "normalization_profile_hash",
        }
    )
