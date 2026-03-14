"""
test_fermion_bounds.py — Rigorous testing for fermion_bounds.py.
Mirror of algebra/tests/test_fermion_bounds.jl.
"""

import time

import pytest

try:
    from qiskit.quantum_info import Pauli  # type: ignore[import-untyped]
except ImportError:
    from braidcodec.algebra._pauli_compat import Pauli  # type: ignore[assignment]

from braidcodec.algebra.fermion_bounds import (
    FB_C_CONSTANT,
    FB_D_PRIME,
    FermionBounds,
    apply_hopping,
    check_available,
    check_pauli_exclusion,
    create_fermion_bounds,
    fermion_parity_operator,
    fermionic_energy,
    get_occupation,
    get_occupation_stats,
    get_owner,
    get_total_parity,
    is_occupied,
    jordan_wigner_string,
    measure_occupation,
    occupy,
    release,
    try_acquire,
    vacate,
    validate_fermionic_state,
)


# ========================================================================
# 2.1 Load Testing
# ========================================================================
class TestLoad:
    def test_class_exists(self):
        assert FermionBounds is not None

    def test_factory_exists(self):
        assert create_fermion_bounds is not None

    def test_constants(self):
        assert pytest.approx(0.141416474, abs=1e-8) == FB_C_CONSTANT
        assert pytest.approx(2.67e-16, abs=1e-18) == FB_D_PRIME

    def test_exports(self):
        import braidcodec.algebra.fermion_bounds as mod

        for name in [
            "occupy",
            "vacate",
            "is_occupied",
            "get_occupation",
            "check_pauli_exclusion",
            "get_total_parity",
            "get_occupation_stats",
            "validate_fermionic_state",
            "apply_hopping",
            "measure_occupation",
            "jordan_wigner_string",
            "fermion_parity_operator",
            "try_acquire",
            "release",
            "check_available",
        ]:
            assert hasattr(mod, name)


# ========================================================================
# 2.2 Stabilizer State Initialization
# ========================================================================
class TestInitialization:
    def test_vacuum_state(self):
        bounds = create_fermion_bounds(10)
        assert bounds.n_sites == 10
        assert not any(bounds.occupation)

    def test_vacuum_parity(self):
        bounds = create_fermion_bounds(10)
        assert get_total_parity(bounds) == 1

    def test_initial_stats(self):
        bounds = create_fermion_bounds(10)
        stats = get_occupation_stats(bounds)
        assert stats.n_occupied == 0
        assert stats.n_empty == 10
        assert stats.filling_fraction == pytest.approx(0.0)
        assert stats.n_sites == 10

    def test_initial_occupation(self):
        bounds = create_fermion_bounds(10, initial_occupation=[1, 3, 5])
        assert sum(bounds.occupation) == 3
        assert bounds.occupation[0]  # site 1
        assert bounds.occupation[2]  # site 3
        assert bounds.occupation[4]  # site 5
        assert not bounds.occupation[1]  # site 2
        assert not bounds.occupation[3]  # site 4

    def test_odd_parity(self):
        bounds = create_fermion_bounds(10, initial_occupation=[1, 3, 5])
        assert get_total_parity(bounds) == -1

    def test_validate_initial(self):
        bounds = create_fermion_bounds(10, initial_occupation=[1, 3, 5])
        v = validate_fermionic_state(bounds)
        assert v.valid
        assert v.parity_consistent
        assert v.occupation_consistent
        assert v.n_particles == 3
        assert v.parity == -1

    def test_even_occupation(self):
        bounds = create_fermion_bounds(10, initial_occupation=[2, 4, 6, 8])
        assert sum(bounds.occupation) == 4
        assert get_total_parity(bounds) == 1

    def test_single_site(self):
        bounds = create_fermion_bounds(1)
        assert bounds.n_sites == 1
        assert not bounds.occupation[0]


# ========================================================================
# 2.3 Occupation Operations
# ========================================================================
class TestOccupation:
    def test_occupy_empty(self):
        bounds = create_fermion_bounds(10)
        assert occupy(bounds, 5)
        assert is_occupied(bounds, 5)

    def test_pauli_exclusion(self):
        bounds = create_fermion_bounds(10)
        occupy(bounds, 5)
        assert not occupy(bounds, 5)
        assert sum(bounds.occupation) == 1

    def test_vacate_occupied(self):
        bounds = create_fermion_bounds(10)
        occupy(bounds, 5)
        assert vacate(bounds, 5)
        assert not is_occupied(bounds, 5)

    def test_vacate_empty(self):
        bounds = create_fermion_bounds(10)
        assert not vacate(bounds, 5)

    def test_multiple_occupations(self):
        bounds = create_fermion_bounds(10)
        for site in [1, 3, 7, 9]:
            assert occupy(bounds, site)
        assert sum(bounds.occupation) == 4

    def test_occupation_vector(self):
        bounds = create_fermion_bounds(10)
        for site in [1, 3, 7, 9]:
            occupy(bounds, site)
        occ = get_occupation(bounds)
        assert occ[0] and occ[2] and occ[6] and occ[8]
        assert not occ[1] and not occ[3] and not occ[4]

    def test_parity_tracking(self):
        bounds = create_fermion_bounds(10)
        for site in [1, 3, 7, 9]:
            occupy(bounds, site)
        assert get_total_parity(bounds) == 1  # 4 = even

    def test_pauli_exclusion_check(self):
        bounds = create_fermion_bounds(10)
        for site in [1, 3, 7, 9]:
            occupy(bounds, site)
        assert check_pauli_exclusion(bounds, [2, 4])
        assert not check_pauli_exclusion(bounds, [1, 2])
        assert not check_pauli_exclusion(bounds, [1, 3])

    def test_sequential_operations(self):
        bounds = create_fermion_bounds(5)
        assert occupy(bounds, 1)
        assert occupy(bounds, 2)
        assert occupy(bounds, 3)
        assert sum(bounds.occupation) == 3
        assert vacate(bounds, 2)
        assert sum(bounds.occupation) == 2
        assert not is_occupied(bounds, 2)
        assert is_occupied(bounds, 1)
        assert is_occupied(bounds, 3)

    def test_validate_after_ops(self):
        bounds = create_fermion_bounds(10)
        for site in [1, 3, 7, 9]:
            occupy(bounds, site)
        v = validate_fermionic_state(bounds)
        assert v.valid
        assert v.n_particles == 4


# ========================================================================
# 2.4 Jordan-Wigner String
# ========================================================================
class TestJordanWigner:
    def test_jw_site1(self):
        jw = jordan_wigner_string(1, 5)
        assert isinstance(jw, Pauli)

    def test_jw_site3(self):
        jw = jordan_wigner_string(3, 5)
        assert isinstance(jw, Pauli)

    @pytest.mark.parametrize("site", range(1, 11))
    def test_jw_all_sites(self, site):
        jw = jordan_wigner_string(site, 10)
        assert isinstance(jw, Pauli)

    def test_parity_operator(self):
        p = fermion_parity_operator(5)
        assert isinstance(p, Pauli)

    @pytest.mark.parametrize("n", range(1, 11))
    def test_parity_operator_sizes(self, n):
        p = fermion_parity_operator(n)
        assert isinstance(p, Pauli)

    def test_jw_anticommutation_occupation(self):
        # Occupy 2 then 4 vs 4 then 2 → same occupation pattern
        b1 = create_fermion_bounds(5)
        occupy(b1, 2)
        occupy(b1, 4)

        b2 = create_fermion_bounds(5)
        occupy(b2, 4)
        occupy(b2, 2)

        s1 = get_occupation_stats(b1)
        s2 = get_occupation_stats(b2)
        assert s1.n_occupied == s2.n_occupied
        assert b1.occupation == b2.occupation


# ========================================================================
# 2.5 Hopping Operations
# ========================================================================
class TestHopping:
    def test_basic_hop(self):
        bounds = create_fermion_bounds(10)
        occupy(bounds, 3)
        assert apply_hopping(bounds, 3, 7)
        assert not is_occupied(bounds, 3)
        assert is_occupied(bounds, 7)

    def test_hop_from_empty(self):
        bounds = create_fermion_bounds(10)
        occupy(bounds, 3)
        apply_hopping(bounds, 3, 7)
        assert not apply_hopping(bounds, 3, 5)

    def test_hop_to_occupied(self):
        bounds = create_fermion_bounds(10)
        occupy(bounds, 3)
        apply_hopping(bounds, 3, 7)
        occupy(bounds, 5)
        assert not apply_hopping(bounds, 7, 5)

    def test_particle_conservation(self):
        bounds = create_fermion_bounds(10, initial_occupation=[2, 5, 8])
        n0 = sum(bounds.occupation)
        apply_hopping(bounds, 5, 6)
        assert sum(bounds.occupation) == n0

    def test_parity_conservation(self):
        bounds = create_fermion_bounds(10, initial_occupation=[2, 5, 8])
        apply_hopping(bounds, 5, 6)
        p_before = get_total_parity(bounds)
        apply_hopping(bounds, 6, 9)
        assert get_total_parity(bounds) == p_before

    def test_multiple_hops(self):
        bounds = create_fermion_bounds(10)
        occupy(bounds, 1)
        assert apply_hopping(bounds, 1, 2)
        assert apply_hopping(bounds, 2, 3)
        assert apply_hopping(bounds, 3, 4)
        assert is_occupied(bounds, 4)
        assert not is_occupied(bounds, 1)
        assert sum(bounds.occupation) == 1

    def test_hop_with_intermediate(self):
        bounds = create_fermion_bounds(10)
        occupy(bounds, 2)
        occupy(bounds, 5)
        occupy(bounds, 8)
        assert apply_hopping(bounds, 2, 3)

    def test_hop_same_site_raises(self):
        bounds = create_fermion_bounds(5)
        occupy(bounds, 3)
        with pytest.raises(ValueError):
            apply_hopping(bounds, 3, 3)

    def test_hop_across_many(self):
        bounds = create_fermion_bounds(20)
        occupy(bounds, 1)
        assert apply_hopping(bounds, 1, 20)
        assert is_occupied(bounds, 20)
        assert not is_occupied(bounds, 1)

    def test_validate_after_hop(self):
        bounds = create_fermion_bounds(10, initial_occupation=[1, 5, 9])
        apply_hopping(bounds, 5, 6)
        v = validate_fermionic_state(bounds)
        assert v.valid


# ========================================================================
# 2.6 Measurement and Validation
# ========================================================================
class TestMeasurement:
    def test_measure_occupied(self):
        bounds = create_fermion_bounds(10, initial_occupation=[2, 5, 7])
        outcome, _ = measure_occupation(bounds, 5)
        assert outcome == 1

    def test_measure_empty(self):
        bounds = create_fermion_bounds(10, initial_occupation=[2, 5, 7])
        outcome, _ = measure_occupation(bounds, 3)
        assert outcome == 0

    def test_measure_all_sites(self):
        bounds = create_fermion_bounds(10, initial_occupation=[2, 5, 7])
        for site in range(1, 11):
            outcome, _ = measure_occupation(bounds, site)
            assert outcome in (0, 1)
            assert outcome == (1 if is_occupied(bounds, site) else 0)

    def test_validation(self):
        bounds = create_fermion_bounds(10, initial_occupation=[2, 5, 7])
        v = validate_fermionic_state(bounds)
        assert v.valid
        assert v.parity_consistent
        assert v.occupation_consistent
        assert v.n_particles == 3

    def test_validate_after_occupy(self):
        bounds = create_fermion_bounds(10, initial_occupation=[2, 5, 7])
        occupy(bounds, 9)
        v = validate_fermionic_state(bounds)
        assert v.valid
        assert v.n_particles == 4

    def test_stats(self):
        bounds = create_fermion_bounds(10, initial_occupation=[2, 5, 7])
        occupy(bounds, 9)
        stats = get_occupation_stats(bounds)
        assert stats.n_occupied == 4
        assert stats.n_empty == 6
        assert stats.filling_fraction == pytest.approx(0.4)
        assert stats.parity == 1

    def test_validate_empty(self):
        bounds = create_fermion_bounds(5)
        v = validate_fermionic_state(bounds)
        assert v.valid
        assert v.n_particles == 0
        assert v.parity == 1

    def test_validate_full(self):
        bounds = create_fermion_bounds(5, initial_occupation=[1, 2, 3, 4, 5])
        v = validate_fermionic_state(bounds)
        assert v.valid
        assert v.n_particles == 5
        assert v.parity == -1


# ========================================================================
# 2.7 TSR Physics
# ========================================================================
class TestTSRPhysics:
    def test_energy(self):
        bounds = create_fermion_bounds(10, initial_occupation=[1, 3, 5])
        E = fermionic_energy(bounds, hopping_strength=1.0, chemical_potential=0.5)
        assert pytest.approx(-1.5) == E

    def test_energy_different_params(self):
        bounds = create_fermion_bounds(10, initial_occupation=[1, 3, 5])
        E1 = fermionic_energy(bounds, chemical_potential=0.5)
        E2 = fermionic_energy(bounds, chemical_potential=1.0)
        assert E1 != E2

    def test_energy_scales_with_particles(self):
        b1 = create_fermion_bounds(10, initial_occupation=[1, 2, 3, 4, 5])
        b2 = create_fermion_bounds(10, initial_occupation=[1, 3, 5])
        E1 = fermionic_energy(b1, chemical_potential=1.0)
        E2 = fermionic_energy(b2, chemical_potential=1.0)
        assert E1 != E2

    def test_try_acquire(self):
        bounds = create_fermion_bounds(5)
        assert try_acquire(bounds, 3)
        assert not try_acquire(bounds, 3)

    def test_release(self):
        bounds = create_fermion_bounds(5)
        try_acquire(bounds, 3)
        assert release(bounds, 3)
        assert not release(bounds, 3)

    def test_check_available(self):
        bounds = create_fermion_bounds(5)
        assert check_available(bounds, 3)
        occupy(bounds, 3)
        assert not check_available(bounds, 3)

    def test_get_owner(self):
        bounds = create_fermion_bounds(5)
        occupy(bounds, 4)
        assert get_owner(bounds, 4) == 1
        assert get_owner(bounds, 2) == 0

    def test_acquire_release_cycles(self):
        bounds = create_fermion_bounds(10)
        for _ in range(5):
            assert try_acquire(bounds, 5)
            assert release(bounds, 5)
        assert not is_occupied(bounds, 5)


# ========================================================================
# 2.8 Edge Cases and Performance
# ========================================================================
class TestEdgeCases:
    def test_single_site_ops(self):
        bounds = create_fermion_bounds(1)
        assert occupy(bounds, 1)
        assert not occupy(bounds, 1)
        assert vacate(bounds, 1)
        assert not vacate(bounds, 1)

    def test_max_sites_exceeded(self):
        with pytest.raises(ValueError):
            create_fermion_bounds(10001)

    def test_large_system(self):
        bounds = create_fermion_bounds(1000)
        assert bounds.n_sites == 1000

    def test_fill_many(self):
        bounds = create_fermion_bounds(1000)
        for i in range(1, 51):
            occupy(bounds, i)
        assert sum(bounds.occupation) == 50

    def test_validate_large(self):
        bounds = create_fermion_bounds(1000)
        for i in range(1, 51):
            occupy(bounds, i)
        v = validate_fermionic_state(bounds)
        assert v.valid
        assert v.n_particles == 50

    def test_perf_occupy(self):
        bounds = create_fermion_bounds(1000)
        t0 = time.perf_counter()
        for i in range(1, 101):
            occupy(bounds, i)
        assert time.perf_counter() - t0 < 0.5

    def test_perf_hopping(self):
        bounds = create_fermion_bounds(1000)
        for i in range(1, 101):
            occupy(bounds, i)
        t0 = time.perf_counter()
        for i in range(1, 100):
            apply_hopping(bounds, i, i + 1)
        assert time.perf_counter() - t0 < 1.0

    def test_invalid_site_zero(self):
        bounds = create_fermion_bounds(10)
        with pytest.raises(ValueError):
            occupy(bounds, 0)

    def test_invalid_site_too_high(self):
        bounds = create_fermion_bounds(10)
        with pytest.raises(ValueError):
            occupy(bounds, 11)

    def test_invalid_is_occupied(self):
        bounds = create_fermion_bounds(10)
        with pytest.raises(ValueError):
            is_occupied(bounds, 0)
        with pytest.raises(ValueError):
            is_occupied(bounds, 11)

    def test_boundary_sites(self):
        bounds = create_fermion_bounds(10)
        assert occupy(bounds, 1)
        assert occupy(bounds, 10)
        assert is_occupied(bounds, 1)
        assert is_occupied(bounds, 10)

    def test_hopping_boundaries(self):
        bounds = create_fermion_bounds(10)
        occupy(bounds, 1)
        occupy(bounds, 10)
        assert apply_hopping(bounds, 1, 2)
        assert apply_hopping(bounds, 10, 9)
        assert is_occupied(bounds, 2)
        assert is_occupied(bounds, 9)

    def test_half_filling_stats(self):
        bounds = create_fermion_bounds(100)
        for i in range(1, 100, 2):
            occupy(bounds, i)
        stats = get_occupation_stats(bounds)
        assert stats.n_occupied == 50
        assert stats.filling_fraction == pytest.approx(0.5)

    def test_repeated_cycles(self):
        bounds = create_fermion_bounds(50)
        for _ in range(10):
            for i in range(1, 26):
                occupy(bounds, i)
            for i in range(1, 26):
                vacate(bounds, i)
        assert sum(bounds.occupation) == 0
        v = validate_fermionic_state(bounds)
        assert v.valid


# ========================================================================
# 2.9 Advanced Fermionic Physics
# ========================================================================
class TestAdvancedPhysics:
    def test_parity_sectors(self):
        b_even = create_fermion_bounds(10, initial_occupation=[1, 2, 3, 4])
        b_odd = create_fermion_bounds(10, initial_occupation=[1, 2, 3])
        assert get_total_parity(b_even) == 1
        assert get_total_parity(b_odd) == -1

    def test_parity_preserved_hopping(self):
        bounds = create_fermion_bounds(10, initial_occupation=[1, 2, 3, 4])
        p0 = get_total_parity(bounds)
        apply_hopping(bounds, 1, 5)
        apply_hopping(bounds, 2, 6)
        assert get_total_parity(bounds) == p0

    def test_particle_number_conservation(self):
        bounds = create_fermion_bounds(10, initial_occupation=[1, 2, 3])
        n0 = sum(bounds.occupation)
        apply_hopping(bounds, 1, 7)
        apply_hopping(bounds, 2, 8)
        apply_hopping(bounds, 3, 9)
        assert sum(bounds.occupation) == n0

    def test_pauli_exclusion_multiple(self):
        bounds = create_fermion_bounds(20)
        sites = [1, 3, 5, 7, 9]
        assert check_pauli_exclusion(bounds, sites)
        occupy(bounds, 5)
        assert not check_pauli_exclusion(bounds, sites)

    def test_complex_hopping_wave(self):
        bounds = create_fermion_bounds(20, initial_occupation=[1, 5, 10, 15, 20])
        assert apply_hopping(bounds, 1, 2)
        assert apply_hopping(bounds, 2, 3)
        assert apply_hopping(bounds, 3, 4)
        assert is_occupied(bounds, 4)
        assert sum(bounds.occupation) == 5

    @pytest.mark.parametrize("n_fill", [0, 25, 50, 75, 100])
    def test_filling_fractions(self, n_fill):
        bounds = create_fermion_bounds(100)
        for i in range(1, n_fill + 1):
            occupy(bounds, i)
        stats = get_occupation_stats(bounds)
        assert stats.filling_fraction == pytest.approx(n_fill / 100)


# ========================================================================
# 2.10 TSR Constants Integration
# ========================================================================
class TestTSRIntegration:
    def test_constants_positive(self):
        assert FB_C_CONSTANT > 0
        assert FB_D_PRIME > 0

    def test_energy_scales_with_mu(self):
        bounds = create_fermion_bounds(10, initial_occupation=[1, 2, 3])
        E1 = fermionic_energy(bounds, chemical_potential=0.1)
        E2 = fermionic_energy(bounds, chemical_potential=0.2)
        assert E1 != E2

    def test_hopping_strength_irrelevant(self):
        bounds = create_fermion_bounds(10, initial_occupation=[1, 2, 3])
        E_low = fermionic_energy(bounds, hopping_strength=0.5, chemical_potential=0.1)
        E_high = fermionic_energy(bounds, hopping_strength=2.0, chemical_potential=0.1)
        assert E_low == E_high

    def test_zero_chemical_potential(self):
        bounds = create_fermion_bounds(10, initial_occupation=[1, 2, 3])
        assert fermionic_energy(bounds, chemical_potential=0.0) == 0.0


class TestCoverageGaps:
    """Cover error branches missed by the main test suite."""

    def test_jordan_wigner_out_of_bounds(self):
        with pytest.raises(ValueError, match="out of bounds"):
            jordan_wigner_string(0, 5)
        with pytest.raises(ValueError, match="out of bounds"):
            jordan_wigner_string(6, 5)

    def test_create_fermion_bounds_zero_sites(self):
        with pytest.raises(ValueError, match="at least one"):
            create_fermion_bounds(0)

    def test_create_fermion_bounds_too_many_sites(self):
        with pytest.raises(ValueError, match="10000"):
            create_fermion_bounds(10_001)

    def test_create_fermion_bounds_bad_initial(self):
        with pytest.raises(ValueError, match="out of bounds"):
            create_fermion_bounds(5, initial_occupation=[0])
        with pytest.raises(ValueError, match="out of bounds"):
            create_fermion_bounds(5, initial_occupation=[6])

    def test_check_pauli_exclusion_out_of_bounds(self):
        bounds = create_fermion_bounds(5)
        with pytest.raises(ValueError, match="out of bounds"):
            check_pauli_exclusion(bounds, [0])
        with pytest.raises(ValueError, match="out of bounds"):
            check_pauli_exclusion(bounds, [6])

    def test_apply_hopping_out_of_bounds(self):
        bounds = create_fermion_bounds(5, initial_occupation=[1])
        with pytest.raises(ValueError, match="from_site out of bounds"):
            apply_hopping(bounds, 0, 2)
        with pytest.raises(ValueError, match="to_site out of bounds"):
            apply_hopping(bounds, 1, 0)
        with pytest.raises(ValueError, match="same site"):
            apply_hopping(bounds, 1, 1)

    def test_measure_occupation_out_of_bounds(self):
        bounds = create_fermion_bounds(5)
        with pytest.raises(ValueError, match="out of bounds"):
            measure_occupation(bounds, 0)
        with pytest.raises(ValueError, match="out of bounds"):
            measure_occupation(bounds, 6)

    def test_check_available_out_of_bounds(self):
        bounds = create_fermion_bounds(5)
        with pytest.raises(ValueError, match="out of bounds"):
            check_available(bounds, 0)

    def test_get_owner_out_of_bounds(self):
        bounds = create_fermion_bounds(5)
        with pytest.raises(ValueError, match="out of bounds"):
            get_owner(bounds, 0)


class TestPauliCompat:
    """Cover _pauli_compat.py branches directly."""

    def test_invalid_label(self):
        with pytest.raises(Exception, match="(?i)pauli.*label|invalid"):
            Pauli("ABC")

    def test_to_label(self):
        p = Pauli("XYZIZ")
        assert p.to_label() == "XYZIZ"

    def test_repr(self):
        p = Pauli("XYZ")
        assert repr(p) == "Pauli('XYZ')"

    def test_eq_same(self):
        assert Pauli("XY") == Pauli("XY")

    def test_eq_different(self):
        assert Pauli("XY") != Pauli("YX")

    def test_eq_non_pauli(self):
        assert Pauli("X") != "X"

    def test_hash(self):
        assert hash(Pauli("XY")) == hash(Pauli("XY"))
        assert hash(Pauli("XY")) != hash(Pauli("YX"))
