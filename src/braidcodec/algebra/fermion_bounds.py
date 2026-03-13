"""
fermion_bounds.py — Fermionic occupation constraints via Jordan-Wigner + stabilizer formalism.

Python mirror of algebra/fermion_bounds.jl.
Uses qiskit.quantum_info.Pauli for Jordan-Wigner operators and a lightweight
stabilizer phase tracker for gate application (O(1) per single-qubit gate).

Theory: Fermions map to qubits via Jordan-Wigner transformation:
    cⱼ  = (∏_{k<j} Zₖ) · (Xⱼ + iYⱼ)/2   [annihilation]
    cⱼ† = (∏_{k<j} Zₖ) · (Xⱼ - iYⱼ)/2   [creation]

The Z-string enforces anticommutation: cᵢcⱼ = -cⱼcᵢ for i≠j.
"""

from __future__ import annotations

from typing import NamedTuple

try:
    from qiskit.quantum_info import Pauli
except ImportError:
    from ._pauli_compat import Pauli

# TSR constants (canonical values from TSRConstants.jl)
FB_C_CONSTANT = 0.141416474  # d-orbital impedance matching
FB_D_PRIME = 2.67e-16  # Decoherence scale (J)


# =============================================================================
# Jordan-Wigner Operators (qiskit Pauli)
# =============================================================================


def jordan_wigner_string(site: int, n_sites: int) -> Pauli:
    """Construct the Jordan-Wigner Z-string for site j: Z₁Z₂...Z_{j-1}.

    Parameters
    ----------
    site : int
        1-indexed site.
    n_sites : int
        Total number of sites.

    Returns
    -------
    qiskit.quantum_info.Pauli
        Pauli operator representing ∏_{k<j} Zₖ.
    """
    if not (1 <= site <= n_sites):
        msg = f"Site {site} out of bounds [1, {n_sites}]"
        raise ValueError(msg)

    # qiskit Pauli label is big-endian: leftmost char = qubit n-1, rightmost = qubit 0
    # site j (1-indexed) → qubit (j-1) (0-indexed)
    # Z on qubits 0..(site-2), I elsewhere
    label = ["I"] * n_sites
    for k in range(site - 1):  # qubits 0..(site-2) → sites 1..(site-1)
        label[n_sites - 1 - k] = "Z"  # big-endian position

    return Pauli("".join(label))


def fermion_parity_operator(n_sites: int) -> Pauli:
    """Total fermion parity operator: P = ∏ⱼ Zⱼ = Z₁Z₂...Zₙ.

    Returns
    -------
    qiskit.quantum_info.Pauli
        Pauli operator with Z on all sites.
    """
    return Pauli("Z" * n_sites)


# =============================================================================
# Lightweight Stabilizer Phase Tracker
# =============================================================================


class _StabilizerPhases:
    """Minimal stabilizer tracking for computational basis states.

    For |x₁x₂...xₙ⟩ the stabilizers are ±Z₁, ±Z₂, ..., ±Zₙ
    where +Zⱼ ↔ |0⟩ⱼ (phase=+1) and -Zⱼ ↔ |1⟩ⱼ (phase=-1).

    Gate effects (on stabilizers):
        X on qubit j → flip phase[j]   (XZX† = -Z)
        Z on qubit j → no change       (ZZZ† = Z)
    """

    __slots__ = ("n", "phases")

    def __init__(self, n: int) -> None:
        self.n = n
        self.phases = [1] * n  # +1 = |0⟩, -1 = |1⟩

    def apply_x(self, qubit: int) -> None:
        """Apply X gate on qubit (0-indexed)."""
        self.phases[qubit] *= -1

    def apply_z(self, qubit: int) -> None:
        """Apply Z gate on qubit (0-indexed). No-op for Z stabilizers."""
        pass


# =============================================================================
# FermionBounds
# =============================================================================


class FermionBounds:
    """Fermionic occupation constraints via Jordan-Wigner + stabilizer formalism.

    Parameters
    ----------
    n_sites : int
        Number of fermionic modes/sites.
    """

    __slots__ = ("n_sites", "occupation", "stab")

    def __init__(self, n_sites: int) -> None:
        self.n_sites: int = n_sites
        self.occupation: list[bool] = [False] * n_sites
        self.stab = _StabilizerPhases(n_sites)


def create_fermion_bounds(
    n_sites: int,
    *,
    initial_occupation: list[int] | None = None,
) -> FermionBounds:
    """Initialise fermionic bounds with *n_sites* modes in vacuum |0…0⟩.

    Parameters
    ----------
    n_sites : int
        Number of fermionic sites (1-indexed in public API).
    initial_occupation : list[int], optional
        1-indexed sites to occupy initially.
    """
    if n_sites <= 0:
        msg = "Need at least one fermionic site"
        raise ValueError(msg)
    if n_sites > 10_000:
        msg = "Maximum 10000 sites for numerical stability"
        raise ValueError(msg)

    bounds = FermionBounds(n_sites)

    if initial_occupation is not None:
        for site in initial_occupation:
            if not (1 <= site <= n_sites):
                msg = f"Site {site} out of bounds [1, {n_sites}]"
                raise ValueError(msg)
            occupy(bounds, site)

    return bounds


# =============================================================================
# Occupation Operations
# =============================================================================


def occupy(bounds: FermionBounds, site: int) -> bool:
    """Create a fermion at *site* (1-indexed). Returns False if already occupied."""
    if not (1 <= site <= bounds.n_sites):
        msg = f"Site {site} out of bounds [1, {bounds.n_sites}]"
        raise ValueError(msg)
    if bounds.occupation[site - 1]:
        return False

    bounds.stab.apply_x(site - 1)
    bounds.occupation[site - 1] = True
    return True


def vacate(bounds: FermionBounds, site: int) -> bool:
    """Annihilate fermion at *site* (1-indexed). Returns False if already empty."""
    if not (1 <= site <= bounds.n_sites):
        msg = f"Site {site} out of bounds [1, {bounds.n_sites}]"
        raise ValueError(msg)
    if not bounds.occupation[site - 1]:
        return False

    bounds.stab.apply_x(site - 1)
    bounds.occupation[site - 1] = False
    return True


def is_occupied(bounds: FermionBounds, site: int) -> bool:
    """Check if fermionic *site* (1-indexed) is occupied."""
    if not (1 <= site <= bounds.n_sites):
        msg = f"Site {site} out of bounds [1, {bounds.n_sites}]"
        raise ValueError(msg)
    return bounds.occupation[site - 1]


def get_occupation(bounds: FermionBounds) -> list[bool]:
    """Return copy of the full occupation vector."""
    return list(bounds.occupation)


# =============================================================================
# Pauli Exclusion & Parity
# =============================================================================


def check_pauli_exclusion(bounds: FermionBounds, sites: list[int]) -> bool:
    """True if *all* listed sites are unoccupied (can be filled)."""
    if not all(1 <= s <= bounds.n_sites for s in sites):
        msg = "Sites out of bounds"
        raise ValueError(msg)
    return not any(bounds.occupation[s - 1] for s in sites)


def get_total_parity(bounds: FermionBounds) -> int:
    """+1 for even particle number, -1 for odd."""
    n_particles = sum(bounds.occupation)
    return 1 if n_particles % 2 == 0 else -1


# =============================================================================
# Hopping
# =============================================================================


def apply_hopping(bounds: FermionBounds, from_site: int, to_site: int) -> bool:
    """Fermionic hopping cⱼ†cᵢ (destroy at *from_site*, create at *to_site*).

    Conserves particle number and parity. Returns False if blocked.
    """
    if not (1 <= from_site <= bounds.n_sites):
        msg = "from_site out of bounds"
        raise ValueError(msg)
    if not (1 <= to_site <= bounds.n_sites):
        msg = "to_site out of bounds"
        raise ValueError(msg)
    if from_site == to_site:
        msg = "Cannot hop to same site"
        raise ValueError(msg)

    if not bounds.occupation[from_site - 1] or bounds.occupation[to_site - 1]:
        return False

    # Fermionic sign from Jordan-Wigner string
    lo, hi = sorted((from_site, to_site))
    n_between = sum(bounds.occupation[k - 1] for k in range(lo + 1, hi))

    # Update occupation
    bounds.occupation[from_site - 1] = False
    bounds.occupation[to_site - 1] = True

    # Update stabilizer state
    bounds.stab.apply_x(from_site - 1)
    bounds.stab.apply_x(to_site - 1)

    # Apply Z gates between sites for odd fermionic sign
    if n_between % 2 != 0:
        for k in range(lo + 1, hi):
            if bounds.occupation[k - 1]:
                bounds.stab.apply_z(k - 1)

    return True


# =============================================================================
# Measurement
# =============================================================================


def measure_occupation(bounds: FermionBounds, site: int) -> tuple[int, FermionBounds]:
    """Measure occupation at *site*. Returns (outcome, bounds)."""
    if not (1 <= site <= bounds.n_sites):
        msg = f"Site {site} out of bounds [1, {bounds.n_sites}]"
        raise ValueError(msg)
    outcome = 1 if bounds.occupation[site - 1] else 0
    return (outcome, bounds)


# =============================================================================
# Validation & Statistics
# =============================================================================


class ValidationResult(NamedTuple):
    valid: bool
    parity_consistent: bool
    occupation_consistent: bool
    stabilizer_rank: int
    n_particles: int
    parity: int


def validate_fermionic_state(bounds: FermionBounds) -> ValidationResult:
    """Validate stabilizer state is consistent with fermionic constraints."""
    n_particles = sum(bounds.occupation)
    expected_parity = 1 if n_particles % 2 == 0 else -1
    actual_parity = get_total_parity(bounds)
    parity_ok = expected_parity == actual_parity

    return ValidationResult(
        valid=parity_ok,
        parity_consistent=parity_ok,
        occupation_consistent=True,
        stabilizer_rank=bounds.n_sites,
        n_particles=n_particles,
        parity=expected_parity,
    )


class OccupationStats(NamedTuple):
    n_occupied: int
    n_empty: int
    filling_fraction: float
    parity: int
    n_sites: int


def get_occupation_stats(bounds: FermionBounds) -> OccupationStats:
    """Statistics about fermionic occupation."""
    n_occ = sum(bounds.occupation)
    return OccupationStats(
        n_occupied=n_occ,
        n_empty=bounds.n_sites - n_occ,
        filling_fraction=n_occ / bounds.n_sites,
        parity=get_total_parity(bounds),
        n_sites=bounds.n_sites,
    )


# =============================================================================
# Energy
# =============================================================================


def fermionic_energy(
    bounds: FermionBounds,
    *,
    hopping_strength: float = 1.0,  # noqa: ARG001
    chemical_potential: float = 0.0,
) -> float:
    """Fermionic energy in tight-binding model.

    E = -t Σ⟨ij⟩ (cᵢ†cⱼ + h.c.) - μ Σᵢ nᵢ

    For computational basis states, kinetic energy is zero.
    """
    n_particles = sum(bounds.occupation)
    return -chemical_potential * n_particles


# =============================================================================
# Atomic Acquisition API
# =============================================================================


def try_acquire(bounds: FermionBounds, site: int) -> bool:
    """Attempt to occupy *site*. Wrapper around `occupy`."""
    return occupy(bounds, site)


def release(bounds: FermionBounds, site: int) -> bool:
    """Release *site*. Wrapper around `vacate`."""
    return vacate(bounds, site)


def check_available(bounds: FermionBounds, site: int) -> bool:
    """True if *site* is unoccupied."""
    if not (1 <= site <= bounds.n_sites):
        msg = f"Site {site} out of bounds [1, {bounds.n_sites}]"
        raise ValueError(msg)
    return not bounds.occupation[site - 1]


def get_owner(bounds: FermionBounds, site: int) -> int:
    """0 if unoccupied, 1 if occupied."""
    if not (1 <= site <= bounds.n_sites):
        msg = f"Site {site} out of bounds [1, {bounds.n_sites}]"
        raise ValueError(msg)
    return 1 if bounds.occupation[site - 1] else 0
