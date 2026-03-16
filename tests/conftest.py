"""Shared test fixtures for BraidCodec."""

from __future__ import annotations

import pytest

from braidcodec import BraidKey, keygen
from braidcodec.algebra.braid_equations import BraidEquation
from braidcodec.algebra.fermion_bounds import FermionBounds, create_fermion_bounds


@pytest.fixture()
def small_braid() -> BraidEquation:
    """A short 3-strand TSR braid: σ₁ σ₂ σ₁."""
    return BraidEquation(3, [1, 2, 1], sector="TSR")


@pytest.fixture()
def identity_braid() -> BraidEquation:
    """An empty 2-strand braid (identity element)."""
    return BraidEquation(2, [])


@pytest.fixture(params=["Identity", "TSR", "Ising", "Fibonacci", "SU2k2"])
def all_sectors(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture yielding each valid sector name."""
    sector: str = request.param
    return sector


@pytest.fixture()
def random_braid() -> object:
    """Factory fixture: ``random_braid(n_strands, length, sector)``."""
    import random as _random

    def _factory(
        n_strands: int = 4,
        length: int = 10,
        sector: str = "TSR",
    ) -> BraidEquation:
        gens = [_random.choice([-1, 1]) * _random.randint(1, n_strands - 1) for _ in range(length)]
        return BraidEquation(n_strands, gens, sector=sector)

    return _factory


@pytest.fixture()
def sample_key() -> BraidKey:
    """A deterministic TSR key with fixed theta_offset."""
    return keygen(sector="TSR", n_strands=4, theta_offset=0.42)


@pytest.fixture()
def sample_data() -> bytes:
    """32-byte sample payload."""
    return b"BraidCodec topological data codec"


@pytest.fixture()
def fermion_bounds_5() -> FermionBounds:
    """5-site fermion bounds with sites 1 and 3 initially occupied."""
    return create_fermion_bounds(5, initial_occupation=[1, 3])
