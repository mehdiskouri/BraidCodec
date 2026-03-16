"""Minimal BraidCodec example: keygen → encode → decode → verify."""

import braidcodec


def main() -> None:
    # Generate a topological key
    key = braidcodec.keygen(sector="TSR", n_strands=4)
    print(f"Key: sector={key.sector}, n_strands={key.n_strands}, id={key.key_id[:8]}…")

    # Encode data into braid equations
    data = b"Hello, topology!"
    encoded = braidcodec.encode(data, key, generators_per_block=8)
    print(f"Encoded: {len(encoded.blocks)} blocks, checksum={encoded.checksum.hex()[:16]}…")

    # Decode back to original bytes
    decoded = braidcodec.decode(encoded, key)
    assert decoded == data
    print(f"Decoded: {decoded!r}")

    # Verify integrity (5-channel verification)
    result = braidcodec.verify(encoded, key)
    assert result.valid
    print(f"Verified: valid={result.valid}")
    for detail in result.details:
        print(f"  {detail}")


if __name__ == "__main__":
    main()
