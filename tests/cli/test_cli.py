"""Tests for the BraidCodec CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from braidcodec.cli.main import (
    EXIT_FORMAT,
    EXIT_KEY_MISMATCH,
    EXIT_OK,
    cli,
)
from braidcodec.codec.schema import EncodedStream, compute_reconstructive_commitment
from braidcodec.crypto.keys import key_from_bytes, key_to_bytes, keygen

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def key_file(tmp_path: Path) -> Path:
    key = keygen(sector="TSR", n_strands=4, theta_offset=1.0)
    p = tmp_path / "test.key"
    p.write_bytes(key_to_bytes(key))
    return p


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.bin"
    p.write_bytes(b"Hello, BraidCodec!")
    return p


@pytest.fixture()
def encoded_file(key_file: Path, sample_file: Path, tmp_path: Path, runner: CliRunner) -> Path:
    out = tmp_path / "encoded.brdc"
    result = runner.invoke(
        cli,
        ["encode", str(sample_file), "-o", str(out), "--key", str(key_file)],
    )
    assert result.exit_code == EXIT_OK, result.output
    return out


def _tamper_reconstructive_payload_contraction(path: Path) -> None:
    """Mutate reconstructive payload K_M params so contraction fails."""
    raw = path.read_bytes()
    if raw[:8] == b"\x89HDF\r\n\x1a\n" or path.suffix.lower() in {".h5", ".hdf5"}:
        stream = EncodedStream.from_hdf5_bytes(raw)
        is_hdf5 = True
    else:
        stream = EncodedStream.from_bytes(raw)
        is_hdf5 = False
    meta = dict(stream.metadata)
    payload_obj = json.loads(meta["reconstructive_payload_v1"])
    payload_obj["km_kappa"] = "0.8"
    payload_obj["km_eta"] = "0.4"
    payload = json.dumps(payload_obj, sort_keys=True, separators=(",", ":"))
    meta["km_kappa"] = "0.8"
    meta["km_eta"] = "0.4"
    meta["reconstructive_payload_v1"] = payload
    meta["reconstructive_commitment"] = compute_reconstructive_commitment(payload, stream.blocks)
    tampered = EncodedStream(
        blocks=stream.blocks,
        n_strands=stream.n_strands,
        sector=stream.sector,
        total_bytes=stream.total_bytes,
        checksum=stream.checksum,
        version=stream.version,
        timestamp=stream.timestamp,
        metadata=meta,
    )
    if is_hdf5:
        path.write_bytes(tampered.to_hdf5_bytes())
    else:
        path.write_bytes(tampered.to_bytes())


# ═════════════════════════════════════════════════════════════════════════
# Top-level group
# ═════════════════════════════════════════════════════════════════════════


class TestCliGroup:
    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == EXIT_OK

    def test_encode_decode_hdf5_extension(self, runner: CliRunner, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        in_file = tmp_path / "input.bin"
        in_file.write_bytes(b"hdf5 container path" * 4)

        key_file = tmp_path / "k.key"
        out_file = tmp_path / "out.h5"
        dec_file = tmp_path / "decoded.bin"

        r_key = runner.invoke(cli, ["keygen", "-o", str(key_file)])
        assert r_key.exit_code == EXIT_OK

        r_enc = runner.invoke(
            cli,
            [
                "encode",
                str(in_file),
                "-o",
                str(out_file),
                "--key",
                str(key_file),
            ],
        )
        assert r_enc.exit_code == EXIT_OK

        magic = out_file.read_bytes()[:8]
        assert magic == b"\x89HDF\r\n\x1a\n"

        r_dec = runner.invoke(
            cli,
            [
                "decode",
                str(out_file),
                "-o",
                str(dec_file),
                "--key",
                str(key_file),
            ],
        )
        assert r_dec.exit_code == EXIT_OK
        assert dec_file.read_bytes() == in_file.read_bytes()

    def test_verbose_quiet_conflict(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--verbose", "--quiet", "keygen", "-o", "/dev/null"])
        assert result.exit_code != EXIT_OK

    def test_commands_listed(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        for cmd in ("keygen", "encode", "decode", "verify", "inspect", "benchmark"):
            assert cmd in result.output


# ═════════════════════════════════════════════════════════════════════════
# keygen
# ═════════════════════════════════════════════════════════════════════════


class TestKeygen:
    def test_keygen_creates_file(self, runner: CliRunner, tmp_path: Path) -> None:
        out = tmp_path / "new.key"
        result = runner.invoke(cli, ["keygen", "-o", str(out)])
        assert result.exit_code == EXIT_OK
        assert out.exists()
        assert len(out.read_bytes()) == 50

    def test_keygen_roundtrips(self, runner: CliRunner, tmp_path: Path) -> None:
        out = tmp_path / "rt.key"
        result = runner.invoke(
            cli, ["keygen", "--sector", "Ising", "--strands", "3", "-o", str(out)]
        )
        assert result.exit_code == EXIT_OK
        key = key_from_bytes(out.read_bytes())
        assert key.sector == "Ising"
        assert key.n_strands == 3

    def test_keygen_verbose(self, runner: CliRunner, tmp_path: Path) -> None:
        out = tmp_path / "v.key"
        result = runner.invoke(cli, ["-v", "keygen", "-o", str(out)])
        assert result.exit_code == EXIT_OK
        assert "Key written" in result.output


# ═════════════════════════════════════════════════════════════════════════
# encode
# ═════════════════════════════════════════════════════════════════════════


class TestEncode:
    def test_encode_with_key(
        self, runner: CliRunner, key_file: Path, sample_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "enc.brdc"
        result = runner.invoke(
            cli, ["encode", str(sample_file), "-o", str(out), "--key", str(key_file)]
        )
        assert result.exit_code == EXIT_OK
        assert out.exists()
        assert out.stat().st_size > 0

    def test_encode_auto_key(self, runner: CliRunner, sample_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "enc2.brdc"
        result = runner.invoke(cli, ["encode", str(sample_file), "-o", str(out)])
        assert result.exit_code == EXIT_OK
        assert "Auto-generated key id" in result.output

    def test_encode_with_compression(
        self, runner: CliRunner, key_file: Path, sample_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "enc_c.brdc"
        result = runner.invoke(
            cli,
            [
                "encode",
                str(sample_file),
                "-o",
                str(out),
                "--key",
                str(key_file),
                "--compression-level",
                "1",
            ],
        )
        assert result.exit_code == EXIT_OK
        assert out.exists()

    def test_encode_reconstructive_with_domain(
        self, runner: CliRunner, key_file: Path, sample_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "enc_reconstructive.h5"
        result = runner.invoke(
            cli,
            [
                "encode",
                str(sample_file),
                "-o",
                str(out),
                "--key",
                str(key_file),
                "--preprocessing-mode",
                "reconstructive",
                "--reconstructive-domain",
                "text",
                "--container",
                "hdf5",
            ],
        )
        assert result.exit_code == EXIT_OK
        assert out.exists()
        assert out.read_bytes()[:8] == b"\x89HDF\r\n\x1a\n"

    def test_encode_reconstructive_auto_uses_wire_for_brdc_suffix(
        self, runner: CliRunner, key_file: Path, sample_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "enc_reconstructive.brdc"
        result = runner.invoke(
            cli,
            [
                "encode",
                str(sample_file),
                "-o",
                str(out),
                "--key",
                str(key_file),
                "--preprocessing-mode",
                "reconstructive",
                "--reconstructive-domain",
                "text",
            ],
        )
        assert result.exit_code == EXIT_OK
        assert out.exists()
        assert out.read_bytes()[:8] != b"\x89HDF\r\n\x1a\n"

    def test_encode_reconstructive_forced_hdf5_ignores_suffix(
        self, runner: CliRunner, key_file: Path, sample_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "enc_reconstructive.brdc"
        result = runner.invoke(
            cli,
            [
                "encode",
                str(sample_file),
                "-o",
                str(out),
                "--key",
                str(key_file),
                "--preprocessing-mode",
                "reconstructive",
                "--reconstructive-domain",
                "text",
                "--container",
                "hdf5",
            ],
        )
        assert result.exit_code == EXIT_OK
        assert out.exists()
        assert out.read_bytes()[:8] == b"\x89HDF\r\n\x1a\n"


# ═════════════════════════════════════════════════════════════════════════
# decode
# ═════════════════════════════════════════════════════════════════════════


class TestDecode:
    def test_decode_roundtrip(
        self, runner: CliRunner, key_file: Path, encoded_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "decoded.bin"
        result = runner.invoke(
            cli, ["decode", str(encoded_file), "-o", str(out), "--key", str(key_file)]
        )
        assert result.exit_code == EXIT_OK
        assert out.read_bytes() == b"Hello, BraidCodec!"

    def test_decode_wrong_key(self, runner: CliRunner, encoded_file: Path, tmp_path: Path) -> None:
        wrong_key_path = tmp_path / "wrong.key"
        wrong_key = keygen(sector="Ising", n_strands=4)
        wrong_key_path.write_bytes(key_to_bytes(wrong_key))
        out = tmp_path / "bad.bin"
        result = runner.invoke(
            cli, ["decode", str(encoded_file), "-o", str(out), "--key", str(wrong_key_path)]
        )
        assert result.exit_code == EXIT_KEY_MISMATCH

    def test_decode_failure_taxonomy_reconstructive_contraction(
        self, runner: CliRunner, key_file: Path, sample_file: Path, tmp_path: Path
    ) -> None:
        enc = tmp_path / "r2.h5"
        out = tmp_path / "out.bin"
        r = runner.invoke(
            cli,
            [
                "encode",
                str(sample_file),
                "-o",
                str(enc),
                "--key",
                str(key_file),
                "--preprocessing-mode",
                "reconstructive",
                "--reconstructive-domain",
                "text",
            ],
        )
        assert r.exit_code == EXIT_OK

        _tamper_reconstructive_payload_contraction(enc)
        dr = runner.invoke(cli, ["decode", str(enc), "-o", str(out), "--key", str(key_file)])
        assert dr.exit_code != EXIT_OK
        assert "Failure category:" in dr.output


# ═════════════════════════════════════════════════════════════════════════
# verify
# ═════════════════════════════════════════════════════════════════════════


class TestVerify:
    def test_verify_valid(self, runner: CliRunner, key_file: Path, encoded_file: Path) -> None:
        result = runner.invoke(cli, ["verify", str(encoded_file), "--key", str(key_file)])
        assert result.exit_code == EXIT_OK
        assert "VALID" in result.output

    def test_verify_wrong_key(self, runner: CliRunner, encoded_file: Path, tmp_path: Path) -> None:
        wrong_key_path = tmp_path / "wrong.key"
        wrong_key = keygen(sector="Ising", n_strands=4)
        wrong_key_path.write_bytes(key_to_bytes(wrong_key))
        result = runner.invoke(cli, ["verify", str(encoded_file), "--key", str(wrong_key_path)])
        # verify() returns VerificationResult with valid=False; exit 1 (INTEGRITY)
        assert result.exit_code != EXIT_OK

    def test_verify_verbose(self, runner: CliRunner, key_file: Path, encoded_file: Path) -> None:
        result = runner.invoke(cli, ["-v", "verify", str(encoded_file), "--key", str(key_file)])
        assert result.exit_code == EXIT_OK
        assert "PASS" in result.output

    def test_verify_diagnostics_flag(
        self,
        runner: CliRunner,
        key_file: Path,
        encoded_file: Path,
    ) -> None:
        result = runner.invoke(
            cli,
            ["verify", str(encoded_file), "--key", str(key_file), "--diagnostics"],
        )
        assert result.exit_code == EXIT_OK

    def test_verify_failure_taxonomy_reconstructive_contraction(
        self, runner: CliRunner, key_file: Path, sample_file: Path, tmp_path: Path
    ) -> None:
        enc = tmp_path / "r.h5"
        r = runner.invoke(
            cli,
            [
                "encode",
                str(sample_file),
                "-o",
                str(enc),
                "--key",
                str(key_file),
                "--preprocessing-mode",
                "reconstructive",
                "--reconstructive-domain",
                "text",
            ],
        )
        assert r.exit_code == EXIT_OK

        _tamper_reconstructive_payload_contraction(enc)
        vr = runner.invoke(cli, ["verify", str(enc), "--key", str(key_file)])
        assert vr.exit_code != EXIT_OK
        assert "Failure category:" in vr.output

    def test_verify_verbose_wrong_key(
        self, runner: CliRunner, encoded_file: Path, tmp_path: Path
    ) -> None:
        """Verbose verify with wrong key → shows details."""
        wrong_key_path = tmp_path / "wrong2.key"
        wrong_key = keygen(sector="Ising", n_strands=4)
        wrong_key_path.write_bytes(key_to_bytes(wrong_key))
        result = runner.invoke(
            cli, ["-v", "verify", str(encoded_file), "--key", str(wrong_key_path)]
        )
        assert result.exit_code != EXIT_OK
        assert "FAIL" in result.output


# ═════════════════════════════════════════════════════════════════════════
# inspect
# ═════════════════════════════════════════════════════════════════════════


class TestInspect:
    def test_inspect_fields(self, runner: CliRunner, encoded_file: Path) -> None:
        result = runner.invoke(cli, ["inspect", str(encoded_file)])
        assert result.exit_code == EXIT_OK
        for field in ("version", "sector", "n_strands", "block_count", "total_bytes", "checksum"):
            assert field in result.output

    def test_inspect_verbose(self, runner: CliRunner, encoded_file: Path) -> None:
        result = runner.invoke(cli, ["-v", "inspect", str(encoded_file)])
        assert result.exit_code == EXIT_OK
        assert "block 0" in result.output

    def test_inspect_corrupt_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Corrupt .brdc → exit FORMAT."""
        bad = tmp_path / "bad.brdc"
        bad.write_bytes(b"XXXX")
        result = runner.invoke(cli, ["inspect", str(bad)])
        assert result.exit_code == EXIT_FORMAT


# ═════════════════════════════════════════════════════════════════════════
# benchmark
# ═════════════════════════════════════════════════════════════════════════


class TestBenchmark:
    def test_benchmark_runs(self, runner: CliRunner, sample_file: Path) -> None:
        result = runner.invoke(cli, ["benchmark", str(sample_file)])
        assert result.exit_code == EXIT_OK
        for header in ("Operation", "Time", "Throughput"):
            assert header in result.output
        assert "Compression ratio" in result.output


# ═════════════════════════════════════════════════════════════════════════
# End-to-end pipeline
# ═════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    def test_full_pipeline(self, runner: CliRunner, tmp_path: Path) -> None:
        """keygen → encode → inspect → verify → decode → compare."""
        key_path = tmp_path / "e2e.key"
        input_path = tmp_path / "input.bin"
        enc_path = tmp_path / "encoded.brdc"
        dec_path = tmp_path / "decoded.bin"

        data = b"End-to-end test data for BraidCodec CLI"
        input_path.write_bytes(data)

        # keygen
        r = runner.invoke(
            cli, ["keygen", "--sector", "TSR", "--strands", "4", "-o", str(key_path)]
        )
        assert r.exit_code == EXIT_OK

        # encode
        r = runner.invoke(
            cli,
            [
                "encode",
                str(input_path),
                "-o",
                str(enc_path),
                "--key",
                str(key_path),
                "--compression-level",
                "1",
            ],
        )
        assert r.exit_code == EXIT_OK

        # inspect
        r = runner.invoke(cli, ["inspect", str(enc_path)])
        assert r.exit_code == EXIT_OK
        assert "TSR" in r.output

        # verify
        r = runner.invoke(cli, ["verify", str(enc_path), "--key", str(key_path)])
        assert r.exit_code == EXIT_OK

        # decode
        r = runner.invoke(
            cli, ["decode", str(enc_path), "-o", str(dec_path), "--key", str(key_path)]
        )
        assert r.exit_code == EXIT_OK
        assert dec_path.read_bytes() == data

    def test_corrupt_file_format_error(self, runner: CliRunner, tmp_path: Path) -> None:
        """Truncated file → exit 3 (FORMAT)."""
        bad = tmp_path / "bad.brdc"
        bad.write_bytes(b"BRDC\x00\x01garbage")
        key_path = tmp_path / "k.key"
        # Need a key file even though it will fail on format first
        key = keygen()
        key_path.write_bytes(key_to_bytes(key))
        r = runner.invoke(
            cli,
            ["decode", str(bad), "-o", str(tmp_path / "out"), "--key", str(key_path)],
        )
        assert r.exit_code == EXIT_FORMAT

    def test_missing_file_io_error(self, runner: CliRunner, tmp_path: Path) -> None:
        """Missing file → exit 4 (IO)."""
        r = runner.invoke(cli, ["inspect", str(tmp_path / "nonexistent.brdc")])
        # click validates exists=True and returns exit 2 (UsageError)
        assert r.exit_code != EXIT_OK

    def test_quiet_mode(self, runner: CliRunner, tmp_path: Path) -> None:
        key_path = tmp_path / "q.key"
        r = runner.invoke(cli, ["-q", "keygen", "-o", str(key_path)])
        assert r.exit_code == EXIT_OK
        assert r.output == ""
