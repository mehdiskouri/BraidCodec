"""BraidCodec CLI — command-line interface.

Six commands: ``keygen``, ``encode``, ``decode``, ``verify``, ``inspect``,
``benchmark``.  Exit codes: 0 (OK), 1 (integrity), 2 (key mismatch),
3 (format), 4 (I/O).
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from braidcodec._exceptions import (
    BraidCodecError,
    FormatError,
    IntegrityError,
    KeyMismatchError,
)
from braidcodec.codec.compressor import compress, compression_ratio
from braidcodec.codec.decoder import decode
from braidcodec.codec.encoder import encode
from braidcodec.codec.schema import EncodedStream
from braidcodec.crypto.integrity import verify
from braidcodec.crypto.keys import (
    BraidKey,
    key_from_bytes,
    key_to_bytes,
)
from braidcodec.crypto.keys import (
    keygen as _keygen,
)

# ── Exit codes ────────────────────────────────────────────────────────────

EXIT_OK: int = 0
EXIT_INTEGRITY: int = 1
EXIT_KEY_MISMATCH: int = 2
EXIT_FORMAT: int = 3
EXIT_IO: int = 4


# ── Helpers ───────────────────────────────────────────────────────────────


def _echo(ctx: click.Context, msg: str, *, err: bool = False) -> None:
    """Print respecting --quiet flag."""
    if not ctx.obj.get("quiet", False):
        click.echo(msg, err=err)


def _verbose(ctx: click.Context) -> bool:
    return bool(ctx.obj.get("verbose", False))


def _handle_error(ctx: click.Context, exc: Exception) -> int:
    """Map exception to exit code and print message."""
    if isinstance(exc, KeyMismatchError):
        _echo(ctx, f"Error: {exc}", err=True)
        return EXIT_KEY_MISMATCH
    if isinstance(exc, IntegrityError):
        _echo(ctx, f"Error: {exc}", err=True)
        return EXIT_INTEGRITY
    if isinstance(exc, FormatError):
        _echo(ctx, f"Error: {exc}", err=True)
        return EXIT_FORMAT
    if isinstance(exc, OSError):
        _echo(ctx, f"I/O error: {exc}", err=True)
        return EXIT_IO
    if isinstance(exc, BraidCodecError):
        _echo(ctx, f"Error: {exc}", err=True)
        return EXIT_INTEGRITY
    raise exc


def _load_key(path: Path) -> BraidKey:
    """Read a key file."""
    return key_from_bytes(path.read_bytes())


def _load_stream(path: Path) -> EncodedStream:
    """Read and deserialize an encoded file."""
    return EncodedStream.from_bytes(path.read_bytes())


# ── CLI group ─────────────────────────────────────────────────────────────


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-error output.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """BraidCodec — topological data codec."""
    if verbose and quiet:
        raise click.UsageError("--verbose and --quiet are mutually exclusive.")
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


# ── keygen ────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--sector", default="TSR", show_default=True, help="Anyon sector.")
@click.option("--strands", default=4, show_default=True, type=int, help="Number of strands.")
@click.option(
    "-o", "--output", "output_path", required=True, type=click.Path(), help="Key output file."
)
@click.pass_context
def keygen(ctx: click.Context, sector: str, strands: int, output_path: str) -> None:
    """Generate a BraidKey and write it to a file."""
    try:
        key = _keygen(sector=sector, n_strands=strands)
        Path(output_path).write_bytes(key_to_bytes(key))
        _echo(ctx, f"Key written to {output_path} (id: {key.key_id})")
    except Exception as exc:
        ctx.exit(_handle_error(ctx, exc))


# ── encode ────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output", "output_path", required=True, type=click.Path(), help="Encoded output file."
)
@click.option("--key", "key_path", type=click.Path(exists=True), default=None, help="Key file.")
@click.option("--sector", default="TSR", show_default=True, help="Anyon sector (if no key).")
@click.option("--strands", default=4, show_default=True, type=int, help="Strands (if no key).")
@click.option(
    "--compression-level",
    "comp_level",
    default=0,
    show_default=True,
    type=click.IntRange(0, 2),
    help="Compression level (0-2).",
)
@click.option("--workers", default=None, type=int, help="Parallel workers.")
@click.pass_context
def encode_cmd(
    ctx: click.Context,
    input_file: str,
    output_path: str,
    key_path: str | None,
    sector: str,
    strands: int,
    comp_level: int,
    workers: int | None,
) -> None:
    """Encode a file into BraidCodec format."""
    try:
        if key_path:
            key = _load_key(Path(key_path))
        else:
            key = _keygen(sector=sector, n_strands=strands)
            click.echo(f"Auto-generated key id: {key.key_id}", err=True)

        data = Path(input_file).read_bytes()
        stream = encode(data, key, max_workers=workers)

        if comp_level > 0:
            stream = compress(stream, key, level=comp_level)

        Path(output_path).write_bytes(stream.to_bytes())
        _echo(ctx, f"Encoded {len(data)} bytes → {output_path}")
    except Exception as exc:
        ctx.exit(_handle_error(ctx, exc))


# ── decode ────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output", "output_path", required=True, type=click.Path(), help="Decoded output file."
)
@click.option("--key", "key_path", required=True, type=click.Path(exists=True), help="Key file.")
@click.pass_context
def decode_cmd(
    ctx: click.Context,
    input_file: str,
    output_path: str,
    key_path: str,
) -> None:
    """Decode a BraidCodec file back to original data."""
    try:
        key = _load_key(Path(key_path))
        stream = _load_stream(Path(input_file))
        result = decode(stream, key)
        Path(output_path).write_bytes(result)
        _echo(ctx, f"Decoded {len(result)} bytes → {output_path}")
    except Exception as exc:
        ctx.exit(_handle_error(ctx, exc))


# ── verify ────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("encoded_file", type=click.Path(exists=True))
@click.option("--key", "key_path", required=True, type=click.Path(exists=True), help="Key file.")
@click.option("--fermion-check", is_flag=True, help="Enable fermion occupation check.")
@click.pass_context
def verify_cmd(
    ctx: click.Context,
    encoded_file: str,
    key_path: str,
    fermion_check: bool,
) -> None:
    """Verify integrity of an encoded file."""
    try:
        key = _load_key(Path(key_path))
        stream = _load_stream(Path(encoded_file))
        result = verify(stream, key, fermion_check=fermion_check)

        _echo(ctx, f"Structural : {'PASS' if result.structural_passed else 'FAIL'}")
        _echo(ctx, f"Writhe     : {'PASS' if result.writhe_passed else 'FAIL'}")
        _echo(ctx, f"Invariant  : {'PASS' if result.invariant_passed else 'FAIL'}")
        if result.fermion_passed is not None:
            _echo(ctx, f"Fermion    : {'PASS' if result.fermion_passed else 'FAIL'}")
        _echo(ctx, f"Checksum   : {'PASS' if result.checksum_passed else 'FAIL'}")
        _echo(ctx, f"Overall    : {'VALID' if result.valid else 'INVALID'}")

        if result.failed_blocks:
            _echo(ctx, f"Failed blocks: {list(result.failed_blocks)}")
        if _verbose(ctx):
            for detail in result.details:
                _echo(ctx, f"  {detail}")

        if not result.valid:
            ctx.exit(EXIT_INTEGRITY)
    except Exception as exc:
        ctx.exit(_handle_error(ctx, exc))


# ── inspect ───────────────────────────────────────────────────────────────


@cli.command()
@click.argument("encoded_file", type=click.Path(exists=True))
@click.pass_context
def inspect_cmd(ctx: click.Context, encoded_file: str) -> None:
    """Inspect metadata of an encoded file (no key needed)."""
    try:
        stream = _load_stream(Path(encoded_file))
        _echo(ctx, f"version     : {stream.version}")
        _echo(ctx, f"sector      : {stream.sector}")
        _echo(ctx, f"n_strands   : {stream.n_strands}")
        _echo(ctx, f"block_count : {len(stream.blocks)}")
        _echo(ctx, f"total_bytes : {stream.total_bytes}")
        _echo(ctx, f"checksum    : {stream.checksum.hex()}")
        _echo(ctx, f"timestamp   : {stream.timestamp}")

        if _verbose(ctx):
            for block in stream.blocks:
                compressed = block.decode_generators is not None
                _echo(
                    ctx,
                    f"  block {block.block_index}: "
                    f"tier={block.invariant_tier}, "
                    f"generators={len(block.generators)}"
                    f"{', compressed' if compressed else ''}",
                )
    except Exception as exc:
        ctx.exit(_handle_error(ctx, exc))


# ── benchmark ─────────────────────────────────────────────────────────────


def _fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def _throughput(nbytes: int, seconds: float) -> str:
    if seconds == 0:
        return "∞"
    bps = nbytes / seconds
    if bps > 1e6:
        return f"{bps / 1e6:.1f} MB/s"
    if bps > 1e3:
        return f"{bps / 1e3:.1f} KB/s"
    return f"{bps:.0f} B/s"


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.pass_context
def benchmark(ctx: click.Context, input_file: str) -> None:
    """Benchmark encode/compress/decode/verify on a file."""
    try:
        data = Path(input_file).read_bytes()
        nbytes = len(data)
        key = _keygen(sector="TSR", n_strands=4)

        # Encode
        t0 = time.perf_counter()
        stream = encode(data, key, max_workers=1)
        t_encode = time.perf_counter() - t0

        # Compress
        t0 = time.perf_counter()
        compressed = compress(stream, key, level=1)
        t_compress = time.perf_counter() - t0
        ratio = compression_ratio(stream, compressed)

        # Decode
        t0 = time.perf_counter()
        decoded = decode(compressed, key)
        t_decode = time.perf_counter() - t0
        assert decoded == data

        # Verify
        t0 = time.perf_counter()
        vr = verify(compressed, key)
        t_verify = time.perf_counter() - t0

        _echo(ctx, f"{'Operation':<12} {'Time':>10} {'Throughput':>14}")
        _echo(ctx, f"{'─' * 12} {'─' * 10} {'─' * 14}")
        _echo(ctx, f"{'Encode':<12} {_fmt_time(t_encode):>10} {_throughput(nbytes, t_encode):>14}")
        _echo(
            ctx,
            f"{'Compress':<12} {_fmt_time(t_compress):>10} {_throughput(nbytes, t_compress):>14}",
        )
        _echo(ctx, f"{'Decode':<12} {_fmt_time(t_decode):>10} {_throughput(nbytes, t_decode):>14}")
        _echo(
            ctx,
            f"{'Verify':<12} {_fmt_time(t_verify):>10} {_throughput(nbytes, t_verify):>14}",
        )
        _echo(ctx, f"Compression ratio: {ratio:.3f}")
        _echo(ctx, f"Verification: {'VALID' if vr.valid else 'INVALID'}")
    except Exception as exc:
        ctx.exit(_handle_error(ctx, exc))
