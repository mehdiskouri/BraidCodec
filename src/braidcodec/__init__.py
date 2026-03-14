"""BraidCodec — Topological data codec using braid group equations."""

from braidcodec._version import __version__
from braidcodec.codec.decoder import decode
from braidcodec.codec.encoder import encode
from braidcodec.codec.schema import EncodedBlock, EncodedStream
from braidcodec.crypto.integrity import VerificationResult, verify
from braidcodec.crypto.keys import BraidKey, key_from_bytes, key_to_bytes, keygen

__all__ = [
    "BraidKey",
    "EncodedBlock",
    "EncodedStream",
    "VerificationResult",
    "__version__",
    "decode",
    "encode",
    "key_from_bytes",
    "key_to_bytes",
    "keygen",
    "verify",
]
