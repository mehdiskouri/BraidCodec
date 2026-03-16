"""Demonstrate BraidCodec's integrity verification catching tampering."""

from __future__ import annotations

import braidcodec
from braidcodec.codec.schema import EncodedStream


def main() -> None:
    # ── Setup: encode some data ───────────────────────────────────
    key = braidcodec.keygen(sector="TSR", n_strands=4)
    data = b"Integrity demonstration payload"
    stream = braidcodec.encode(data, key, generators_per_block=8)

    print("=== Original stream ===")
    result = braidcodec.verify(stream, key)
    print(f"Valid: {result.valid}")
    for d in result.details:
        print(f"  {d}")

    # ── Tamper 1: corrupt checksum ────────────────────────────────
    print("\n=== Tamper: corrupt checksum ===")
    bad_checksum_stream = EncodedStream(
        blocks=stream.blocks,
        n_strands=stream.n_strands,
        sector=stream.sector,
        total_bytes=stream.total_bytes,
        checksum=b"\x00" * 32,  # wrong checksum
        version=stream.version,
        timestamp=stream.timestamp,
        metadata=stream.metadata,
    )
    result = braidcodec.verify(bad_checksum_stream, key)
    print(f"Valid: {result.valid}")
    print(f"Checksum passed: {result.checksum_passed}")
    for d in result.details:
        print(f"  {d}")

    # ── Tamper 2: wrong key ───────────────────────────────────────
    print("\n=== Tamper: wrong key ===")
    wrong_key = braidcodec.keygen(sector="Ising", n_strands=4)
    result = braidcodec.verify(stream, wrong_key)
    print(f"Valid: {result.valid}")
    for d in result.details:
        print(f"  {d}")

    # ── Tamper 3: modify block writhe ─────────────────────────────
    print("\n=== Tamper: modify stored writhe ===")
    from dataclasses import replace

    block0 = stream.blocks[0]
    bad_block = replace(block0, writhe=block0.writhe + 99)
    bad_writhe_stream = EncodedStream(
        blocks=(bad_block, *stream.blocks[1:]),
        n_strands=stream.n_strands,
        sector=stream.sector,
        total_bytes=stream.total_bytes,
        checksum=stream.checksum,
        version=stream.version,
        timestamp=stream.timestamp,
        metadata=stream.metadata,
    )
    result = braidcodec.verify(bad_writhe_stream, key)
    print(f"Valid: {result.valid}")
    print(f"Writhe passed: {result.writhe_passed}")
    for d in result.details:
        print(f"  {d}")

    print("\n✓ All tampering scenarios detected correctly")


if __name__ == "__main__":
    main()
