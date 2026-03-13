"""BraidCodec — Topological data codec using braid group equations."""

from braidcodec._version import __version__
from braidcodec.codec.schema import EncodedBlock, EncodedStream

__all__ = ["EncodedBlock", "EncodedStream", "__version__"]
