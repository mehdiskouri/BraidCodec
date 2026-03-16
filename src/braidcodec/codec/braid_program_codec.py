"""Braid-backed serialization helpers for discovered reconstructive equations.

This module provides a deterministic, reversible encoding from discovered
byte-equation parameters into a braid-generator word.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from braidcodec._exceptions import FormatError
from braidcodec.algebra.braid_equations import create_braid_equation

BRAID_PROGRAM_N_STRANDS = 5
_U8_SYMBOLS: tuple[int, ...] = (1, -1, 2, -2, 3, -3, 4, -4)
_DIGIT_BY_SYMBOL: dict[int, int] = {g: i for i, g in enumerate(_U8_SYMBOLS)}
_SYMBOL_BY_DIGIT: tuple[int, ...] = _U8_SYMBOLS
_U8_WORD_LEN = 3
_FAMILY_TO_CODE: dict[str, int] = {
    "byte-constant-v1": 1,
    "byte-linear-mod-v1": 2,
    "byte-affine-recursion-v1": 3,
    "byte-xor-step-v1": 4,
}
_CODE_TO_FAMILY: dict[int, str] = {v: k for k, v in _FAMILY_TO_CODE.items()}


@dataclass(frozen=True)
class DiscoveredEquationFields:
    equation_family: str
    coefficients: list[int]
    initial_state: list[int]
    rollout_length: int


def _u8_to_word(value: int) -> list[int]:
    if not (0 <= value <= 255):
        raise FormatError("Byte value out of range for braid serialization", value=value)
    d0 = value % 8
    d1 = (value // 8) % 8
    d2 = (value // 64) % 8
    return [_U8_SYMBOLS[d0], _U8_SYMBOLS[d1], _U8_SYMBOLS[d2]]


def _word_to_u8(word: list[int]) -> int:
    if len(word) != _U8_WORD_LEN:
        raise FormatError("Invalid braid byte-word length", length=len(word))
    try:
        d0 = _DIGIT_BY_SYMBOL[word[0]]
        d1 = _DIGIT_BY_SYMBOL[word[1]]
        d2 = _DIGIT_BY_SYMBOL[word[2]]
    except KeyError as exc:
        raise FormatError("Invalid braid generator in byte-word") from exc
    return d0 + (8 * d1) + (64 * d2)


def _counts_for_family(equation_family: str) -> tuple[int, int]:
    if equation_family == "byte-constant-v1":
        return 1, 1
    if equation_family == "byte-linear-mod-v1":
        return 1, 1
    if equation_family == "byte-xor-step-v1":
        return 1, 1
    if equation_family == "byte-affine-recursion-v1":
        return 2, 1
    raise FormatError("Unsupported discovered equation family", equation_family=equation_family)


def _pack_values_to_generators(values: list[int]) -> list[int]:
    generators: list[int] = []
    for value in values:
        generators.extend(_u8_to_word(value & 0xFF))
    return generators


def _unpack_values_from_generators(generators: list[int], expected_count: int) -> list[int]:
    expected_len = expected_count * _U8_WORD_LEN
    if len(generators) != expected_len:
        raise FormatError(
            "Invalid braid generator count for discovered equation payload",
            expected=expected_len,
            actual=len(generators),
        )

    out: list[int] = []
    for i in range(0, len(generators), _U8_WORD_LEN):
        out.append(_word_to_u8(generators[i : i + _U8_WORD_LEN]))
    return out


def _pack_generators_to_b85(generators: list[int]) -> str:
    """Pack generator symbols into base85 over 4-bit digits for compact payloads."""
    digits: list[int] = []
    for gen in generators:
        if gen not in _DIGIT_BY_SYMBOL:
            raise FormatError("Unsupported generator in compact braid payload", generator=gen)
        digits.append(_DIGIT_BY_SYMBOL[gen])

    packed = bytearray()
    i = 0
    while i < len(digits):
        hi = digits[i]
        lo = digits[i + 1] if (i + 1) < len(digits) else 0xF
        packed.append((hi << 4) | lo)
        i += 2
    return base64.b85encode(bytes(packed)).decode("ascii")


def _unpack_generators_from_b85(blob: str, expected_count: int) -> list[int]:
    """Unpack compact base85 4-bit digit representation into generators."""
    try:
        raw = base64.b85decode(blob.encode("ascii"))
    except Exception as exc:
        raise FormatError("Invalid compact braid generator payload") from exc

    digits: list[int] = []
    for value in raw:
        digits.append((value >> 4) & 0xF)
        digits.append(value & 0xF)

    if len(digits) < expected_count:
        raise FormatError("Compact braid payload too short", expected=expected_count, actual=len(digits))

    used = digits[:expected_count]
    tail = digits[expected_count:]
    if any(d != 0xF for d in tail):
        raise FormatError("Compact braid payload contains invalid tail bits")

    if any(d > 7 for d in used):
        raise FormatError("Compact braid payload contains invalid digit")

    return [_SYMBOL_BY_DIGIT[d] for d in used]


def compile_discovered_braid_program(
    *,
    equation_family: str,
    coefficients: tuple[int, ...],
    initial_state: tuple[int, ...],
    rollout_length: int,
    symbolic_hash: str,
    symbolic_form: str,
) -> dict[str, object]:
    """Compile discovered equation fields into braid-backed payload fields."""
    _ = symbolic_hash
    _ = symbolic_form
    coeff_count, init_count = _counts_for_family(equation_family)
    if len(coefficients) != coeff_count:
        raise FormatError("Coefficient count mismatch for discovered equation family")
    if len(initial_state) != init_count:
        raise FormatError("Initial-state count mismatch for discovered equation family")
    if rollout_length < 0:
        raise FormatError("Invalid discovered equation rollout length")

    encoded_values = [*(c & 0xFF for c in coefficients), *(s & 0xFF for s in initial_state)]
    braid_generators = _pack_values_to_generators(encoded_values)

    # Validate this is a syntactically valid braid equation representation.
    create_braid_equation(braid_generators, BRAID_PROGRAM_N_STRANDS, sector="Identity")

    family_code = _FAMILY_TO_CODE[equation_family]
    return {
        "e": family_code,
        "r": rollout_length,
        "g": _pack_generators_to_b85(braid_generators),
    }


def parse_discovered_braid_program(program: dict[str, object]) -> DiscoveredEquationFields:
    """Decode braid-backed discovered-equation payload into equation fields."""
    family_code_obj = program.get("e")
    if family_code_obj is None:
        raise FormatError("Missing discovered braid equation family code")
    try:
        equation_family = _CODE_TO_FAMILY[int(family_code_obj)]
    except Exception as exc:
        raise FormatError("Unsupported discovered braid equation code") from exc

    rollout_raw = program.get("r", -1)
    n_strands_raw = program.get("ns", BRAID_PROGRAM_N_STRANDS)
    sector = str(program.get("bs", "Identity"))
    generators_obj = program.get("braid_generators", [])
    generators_b85 = program.get("g", "")

    try:
        rollout_length = int(rollout_raw)
        n_strands = int(n_strands_raw)
    except (TypeError, ValueError) as exc:
        raise FormatError("Invalid discovered braid numeric payload") from exc

    if rollout_length < 0:
        raise FormatError("Invalid discovered braid rollout length")
    if n_strands != BRAID_PROGRAM_N_STRANDS:
        raise FormatError("Unsupported discovered braid strand count", n_strands=n_strands)
    if sector != "Identity":
        raise FormatError("Unsupported discovered braid sector", sector=sector)
    coeff_count, init_count = _counts_for_family(equation_family)
    expected_gen_count = (coeff_count + init_count) * _U8_WORD_LEN

    generators: list[int]
    if isinstance(generators_b85, str) and generators_b85:
        generators = _unpack_generators_from_b85(generators_b85, expected_count=expected_gen_count)
    else:
        if not isinstance(generators_obj, list):
            raise FormatError("Invalid discovered braid generators payload")
        try:
            generators = [int(g) for g in generators_obj]
        except (TypeError, ValueError) as exc:
            raise FormatError("Invalid discovered braid generators payload") from exc

    create_braid_equation(generators, BRAID_PROGRAM_N_STRANDS, sector="Identity")

    values = _unpack_values_from_generators(generators, coeff_count + init_count)
    coefficients = values[:coeff_count]
    initial_state = values[coeff_count:]

    return DiscoveredEquationFields(
        equation_family=equation_family,
        coefficients=coefficients,
        initial_state=initial_state,
        rollout_length=rollout_length,
    )
