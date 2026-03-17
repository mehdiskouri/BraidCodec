"""Tests for reconstructive commitment side-channel binding behavior."""

from __future__ import annotations

import pytest

from braidcodec._exceptions import FormatError
from braidcodec.codec.encoder import encode
from braidcodec.codec.schema import (
    validate_reconstructive_compact_transport_metadata,
)
from braidcodec.crypto.keys import BraidKey, keygen


def _make_key() -> BraidKey:
    return keygen(sector="TSR", n_strands=4)


def test_commitment_v2_binds_program_sidechannel() -> None:
    key = _make_key()
    data = ("not-a-simple-progression " * 20).encode("utf-8")
    stream = encode(data, key, preprocessing_mode="reconstructive", reconstructive_domain="text")

    # Ensure side-channel payload path is exercised for compact reconstructive streams.
    assert "rpb" in stream.metadata
    validate_reconstructive_compact_transport_metadata(stream.metadata, stream.blocks)

    tampered = dict(stream.metadata)
    sidechannel = tampered["rpb"]
    tampered["rpb"] = sidechannel + (b"A" if isinstance(sidechannel, bytes) else "A")

    with pytest.raises(FormatError, match="commitment v3 mismatch"):
        validate_reconstructive_compact_transport_metadata(tampered, stream.blocks)


def test_lean_mode_omits_compact_commitment_field() -> None:
    key = _make_key()
    data = b'{"msg":"legacy-compat"}'
    stream = encode(
        data,
        key,
        preprocessing_mode="reconstructive",
        reconstructive_domain="json",
        reconstructive_compact_transport="lean",
    )

    assert "rc" not in stream.metadata
    assert "rc3" not in stream.metadata
    validate_reconstructive_compact_transport_metadata(stream.metadata, stream.blocks)
