"""Encode a file to .brdc format and save the key."""

from __future__ import annotations

import argparse
from pathlib import Path

import braidcodec


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a file with BraidCodec")
    parser.add_argument("input", type=Path, help="Input file to encode")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .brdc file (default: INPUT.brdc)",
    )
    parser.add_argument("-k", "--key-file", type=Path, help="Key file (default: INPUT.key)")
    parser.add_argument("--sector", default="TSR", help="Anyon sector (default: TSR)")
    parser.add_argument("--n-strands", type=int, default=4, help="Number of strands (default: 4)")
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output or input_path.with_suffix(".brdc")
    key_path: Path = args.key_file or input_path.with_suffix(".key")

    # Generate key
    key = braidcodec.keygen(sector=args.sector, n_strands=args.n_strands)
    key_path.write_bytes(braidcodec.key_to_bytes(key))
    print(f"Key saved to {key_path} (id={key.key_id[:8]}…)")

    # Read and encode
    data = input_path.read_bytes()
    encoded = braidcodec.encode(data, key, generators_per_block=8)
    output_path.write_bytes(encoded.to_bytes())
    print(f"Encoded {len(data)} bytes → {output_path} ({len(encoded.blocks)} blocks)")

    # Verify round-trip
    loaded = braidcodec.EncodedStream.from_bytes(output_path.read_bytes())
    decoded = braidcodec.decode(loaded, key)
    assert decoded == data
    print("Round-trip verified ✓")


if __name__ == "__main__":
    main()
