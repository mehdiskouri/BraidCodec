"""Tests for the BraidCodec CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from braidcodec.cli.main import (
    EXIT_FORMAT,
    EXIT_KEY_MISMATCH,
    EXIT_OK,
    cli,
)
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


# ═════════════════════════════════════════════════════════════════════════
# Top-level group
# ═════════════════════════════════════════════════════════════════════════


class TestCliGroup:
    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == EXIT_OK
        assert "BraidCodec" in result.output

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
