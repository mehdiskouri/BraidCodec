"""BraidCodec — Topological data codec using braid group equations."""

from braidcodec._version import __version__
from braidcodec.codec.encoder import encode
from braidcodec.codec.schema import EncodedBlock, EncodedStream
from braidcodec.crypto.keys import BraidKey, key_from_bytes, key_to_bytes, keygen

__all__ = [
    "BraidKey",
    "EncodedBlock",
    "EncodedStream",
    "__version__",
    "encode",
    "key_from_bytes",
    "key_to_bytes",
    "keygen",
]
