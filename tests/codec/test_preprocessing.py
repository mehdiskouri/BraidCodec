"""Tests for topology preprocessing helpers."""

from __future__ import annotations

from braidcodec.codec.preprocessing import (
    build_chunk_profiles,
    compute_bfps,
    mean_estimated_cost,
    sort_chunk_indices_for_topology,
)


class TestComputeBfps:
    def test_deterministic_signature(self) -> None:
        chunk = b"abc123" * 3
        s1 = compute_bfps(chunk)
        s2 = compute_bfps(chunk)
        assert s1 == s2

    def test_empty_chunk_signature(self) -> None:
        sig = compute_bfps(b"")
        assert sig.hash32 == 0
        assert sig.density_fp == 0
        assert sig.centroid_fp == 0
        assert sig.variance_fp == 0


class TestChunkProfiles:
    def test_profile_count_matches_chunks(self) -> None:
        chunks = [
            (0, b"\x00\x01", 2),
            (1, b"\x10\x11", 2),
            (2, b"\xff\x00", 2),
        ]
        profiles = build_chunk_profiles(chunks, n_strands=4, generators_per_block=8)
        assert len(profiles) == len(chunks)

    def test_profile_determinism(self) -> None:
        chunks = [
            (0, b"chunk0", 6),
            (1, b"chunk1", 6),
            (2, b"chunk2", 6),
            (3, b"chunk3", 6),
        ]
        p1 = build_chunk_profiles(chunks, n_strands=4, generators_per_block=32)
        p2 = build_chunk_profiles(chunks, n_strands=4, generators_per_block=32)
        assert p1 == p2

    def test_sorted_indices_cover_all_blocks(self) -> None:
        chunks = [
            (0, b"aaaa", 4),
            (1, b"bbbb", 4),
            (2, b"cccc", 4),
            (3, b"dddd", 4),
        ]
        profiles = build_chunk_profiles(chunks, n_strands=5, generators_per_block=16)
        order = sort_chunk_indices_for_topology(profiles)
        assert sorted(order) == [0, 1, 2, 3]

    def test_mean_cost_positive_for_non_empty(self) -> None:
        chunks = [(0, b"abcdef", 6)]
        profiles = build_chunk_profiles(chunks, n_strands=6, generators_per_block=16)
        assert mean_estimated_cost(profiles) > 0
