"""Persistent candidate equation library for reconstructive discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class CandidateLibraryEntry:
    """A reusable discovery artifact bound to a semantic signature."""

    signature: str
    domain_kind: str
    reconstructive_program_type: str
    reconstructive_program_payload: str
    symbolic_hash: str


class CandidateLibraryStore:
    """Deterministic JSON-backed retrieval/promotion store."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @classmethod
    def from_optional_path(cls, path: str | None) -> CandidateLibraryStore | None:
        if path is None or not path.strip():
            return None
        return cls(Path(path))

    def _load(self) -> dict[str, dict[str, str]]:
        if not self._path.exists():
            return {}
        raw = self._path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        parsed_obj = json.loads(raw)
        if not isinstance(parsed_obj, dict):
            return {}
        parsed = cast("dict[Any, Any]", parsed_obj)
        out: dict[str, dict[str, str]] = {}
        for key, value in parsed.items():
            if isinstance(key, str) and isinstance(value, dict):
                value_map = cast("dict[Any, Any]", value)
                normalized: dict[str, str] = {}
                for raw_k, raw_v in value_map.items():
                    normalized[str(raw_k)] = str(raw_v)
                out[key] = normalized
        return out

    def _save(self, payload: dict[str, dict[str, str]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    def retrieve(self, signature: str) -> CandidateLibraryEntry | None:
        items = self._load()
        raw = items.get(signature)
        if raw is None:
            return None

        required = {
            "domain_kind",
            "reconstructive_program_type",
            "reconstructive_program_payload",
            "symbolic_hash",
        }
        if any(k not in raw for k in required):
            return None

        return CandidateLibraryEntry(
            signature=signature,
            domain_kind=raw["domain_kind"],
            reconstructive_program_type=raw["reconstructive_program_type"],
            reconstructive_program_payload=raw["reconstructive_program_payload"],
            symbolic_hash=raw["symbolic_hash"],
        )

    def retrieve_compatible(
        self,
        *,
        domain_kind: str,
        exclude_signature: str,
        limit: int = 32,
    ) -> list[CandidateLibraryEntry]:
        """Return deterministic domain-compatible candidates excluding exact key."""
        items = self._load()
        out: list[CandidateLibraryEntry] = []
        for signature in sorted(items.keys()):
            if signature == exclude_signature:
                continue
            raw = items[signature]
            required = {
                "domain_kind",
                "reconstructive_program_type",
                "reconstructive_program_payload",
                "symbolic_hash",
            }
            if any(k not in raw for k in required):
                continue
            if raw["domain_kind"] != domain_kind:
                continue
            out.append(
                CandidateLibraryEntry(
                    signature=signature,
                    domain_kind=raw["domain_kind"],
                    reconstructive_program_type=raw["reconstructive_program_type"],
                    reconstructive_program_payload=raw["reconstructive_program_payload"],
                    symbolic_hash=raw["symbolic_hash"],
                )
            )
            if len(out) >= max(limit, 1):
                break
        return out

    def promote(self, entry: CandidateLibraryEntry) -> None:
        items = self._load()
        items[entry.signature] = {
            "domain_kind": entry.domain_kind,
            "reconstructive_program_type": entry.reconstructive_program_type,
            "reconstructive_program_payload": entry.reconstructive_program_payload,
            "symbolic_hash": entry.symbolic_hash,
        }
        self._save(items)


def compute_semantic_signature(parts: dict[str, Any]) -> str:
    """Compute deterministic signature key for library retrieval."""
    import blake3

    serial = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return blake3.blake3(serial.encode("utf-8")).hexdigest()
