"""
test_yang_baxter.py — Rigorous testing for yang_baxter.py.
Mirror of algebra/tests/test_yang_baxter.jl.
"""

import time

import numpy as np
import pytest
from numpy.testing import assert_allclose

from braidcodec.algebra.braid_equations import (
    contract_braid_tensor,
    create_braid_equation,
    jones_polynomial,
    simplify_braid,
)
from braidcodec.algebra.yang_baxter import (
    check_yb_equivalence_catlab,
    compute_braid_complexity,
    create_braid_diagram,
    validate_braid_category,
    verify_yang_baxter_morphism,
)


# ========================================================================
# 3.1 Load Testing
# ========================================================================
class TestLoad:
    def test_exports(self):
        import braidcodec.algebra.yang_baxter as mod

        for name in [
            "verify_yang_baxter_morphism",
            "validate_braid_category",
            "create_braid_diagram",
            "check_yb_equivalence_catlab",
            "compute_braid_complexity",
        ]:
            assert hasattr(mod, name)


# ========================================================================
# 3.2 Categorical Diagram Construction
# ========================================================================
class TestBraidDiagrams:
    def test_single_generator(self):
        diag = create_braid_diagram([1], 3)
        assert isinstance(diag, list)
        assert len(diag) == 1

    def test_multiple_generators(self):
        diag = create_braid_diagram([1, 2], 3)
        assert len(diag) == 2

    def test_inverse_generators(self):
        diag = create_braid_diagram([-1], 3)
        assert isinstance(diag, list)
        assert len(diag) == 1

    def test_complex_diagram(self):
        diag = create_braid_diagram([1, 2, 1], 3)
        assert len(diag) == 3

    @pytest.mark.parametrize("n", range(2, 7))
    def test_various_strand_counts(self, n):
        for i in range(1, n):
            diag = create_braid_diagram([i], n)
            assert len(diag) == 1

    def test_empty_diagram(self):
        diag = create_braid_diagram([], 3)
        assert len(diag) == 0

    def test_long_sequence(self):
        diag = create_braid_diagram(list(range(1, 6)), 6)
        assert len(diag) == 5

    def test_mixed_positive_negative(self):
        diag = create_braid_diagram([1, -2, 3, -1], 5)
        assert len(diag) == 4

    def test_diagram_entries_are_lists(self):
        diag = create_braid_diagram([1, 2], 3)
        assert all(isinstance(d, list) for d in diag)

    def test_repeated_generators(self):
        diag = create_braid_diagram([1, 1, 1], 3)
        assert len(diag) == 3


# ========================================================================
# 3.3 Yang-Baxter Morphism Verification
# ========================================================================
class TestYBMorphism:
    def test_yang_baxter_equation(self):
        b_lhs = create_braid_equation([1, 2, 1], 3, sector="TSR")
        b_rhs = create_braid_equation([2, 1, 2], 3, sector="TSR")
        assert verify_yang_baxter_morphism(b_lhs, b_rhs)

    @pytest.mark.parametrize("sector", ["TSR", "Ising", "Fibonacci", "SU2k2"])
    def test_yb_all_sectors(self, sector):
        b1 = create_braid_equation([1, 2, 1], 3, sector=sector)
        b2 = create_braid_equation([2, 1, 2], 3, sector=sector)
        assert verify_yang_baxter_morphism(b1, b2)

    def test_non_equivalent_braids(self):
        b1 = create_braid_equation([1, 2], 3, sector="TSR")
        b2 = create_braid_equation([2, 1], 3, sector="TSR")
        assert not verify_yang_baxter_morphism(b1, b2)

    def test_identity_braids_equivalent(self):
        b1 = create_braid_equation([], 3, sector="TSR")
        b2 = create_braid_equation([], 3, sector="TSR")
        assert verify_yang_baxter_morphism(b1, b2)

    def test_forward_and_inverse_not_equivalent(self):
        b_fwd = create_braid_equation([1], 3, sector="TSR")
        b_inv = create_braid_equation([-1], 3, sector="TSR")
        assert not verify_yang_baxter_morphism(b_fwd, b_inv)

    def test_yb_4_strands(self):
        b_lhs = create_braid_equation([1, 2, 1], 4, sector="TSR")
        b_rhs = create_braid_equation([2, 1, 2], 4, sector="TSR")
        assert verify_yang_baxter_morphism(b_lhs, b_rhs)

    def test_yb_different_positions(self):
        b_lhs = create_braid_equation([2, 3, 2], 4, sector="TSR")
        b_rhs = create_braid_equation([3, 2, 3], 4, sector="TSR")
        assert verify_yang_baxter_morphism(b_lhs, b_rhs)

    def test_simplified_equivalence(self):
        b_complex = create_braid_equation([1, -1, 2, 1], 3, sector="TSR")
        b_simple = create_braid_equation([2, 1], 3, sector="TSR")
        assert verify_yang_baxter_morphism(b_complex, b_simple)

    def test_different_strand_dimensions(self):
        b3 = create_braid_equation([1, 2, 1], 3, sector="TSR")
        b4 = create_braid_equation([1, 2, 1], 4, sector="TSR")
        m3 = contract_braid_tensor(b3)
        m4 = contract_braid_tensor(b4)
        assert m3.shape != m4.shape  # 8×8 vs 16×16

    def test_non_adjacent_commute(self):
        b_13 = create_braid_equation([1, 3], 4, sector="TSR")
        b_31 = create_braid_equation([3, 1], 4, sector="TSR")
        assert verify_yang_baxter_morphism(b_13, b_31)

    def test_identical_large_braid(self):
        b1 = create_braid_equation([1, 2, 3, 4], 5, sector="TSR")
        b2 = create_braid_equation([1, 2, 3, 4], 5, sector="TSR")
        assert verify_yang_baxter_morphism(b1, b2)


# ========================================================================
# 3.4 Category Axiom Validation
# ========================================================================
class TestCategoryAxioms:
    def test_valid_braid(self):
        braid = create_braid_equation([1, 2, 1], 3, sector="TSR")
        assert validate_braid_category(braid)

    def test_empty_braid(self):
        braid = create_braid_equation([], 5, sector="TSR")
        assert validate_braid_category(braid)

    @pytest.mark.parametrize("i", range(1, 5))
    def test_single_generator(self, i):
        braid = create_braid_equation([i], 5, sector="TSR")
        assert validate_braid_category(braid)

    def test_inverse_generators(self):
        braid = create_braid_equation([-1, -2, -3], 4, sector="TSR")
        assert validate_braid_category(braid)

    def test_complex_braid(self):
        braid = create_braid_equation([1, 2, 3, 1, 2, 1], 4, sector="TSR")
        assert validate_braid_category(braid)

    def test_yb_pattern(self):
        braid = create_braid_equation([1, 2, 1], 3, sector="TSR")
        assert validate_braid_category(braid)

    def test_multiple_yb_patterns(self):
        braid = create_braid_equation([1, 2, 1, 2, 3, 2], 4, sector="TSR")
        assert validate_braid_category(braid)

    def test_large_strand_count(self):
        braid = create_braid_equation([1, 3, 5, 7], 8, sector="TSR")
        assert validate_braid_category(braid)

    @pytest.mark.parametrize("sector", ["TSR", "Ising", "Fibonacci", "SU2k2"])
    def test_all_sectors(self, sector):
        braid = create_braid_equation([1, 2, 1], 3, sector=sector)
        assert validate_braid_category(braid)

    def test_composition_validation(self):
        braid = create_braid_equation([1, 2], 3, sector="TSR")
        assert validate_braid_category(braid)
        braid_id = create_braid_equation([1, -1], 3, sector="TSR")
        assert validate_braid_category(braid_id)


# ========================================================================
# 3.5 Catlab Equivalence Checking
# ========================================================================
class TestCatlabEquivalence:
    def test_simplified_braids_equivalent(self):
        b1 = create_braid_equation([1, -1, 2], 3, sector="TSR")
        b2 = create_braid_equation([2], 3, sector="TSR")
        assert check_yb_equivalence_catlab(b1, b2)

    def test_yb_equivalence(self):
        b1 = create_braid_equation([1, 2, 1], 3, sector="TSR")
        b2 = create_braid_equation([2, 1, 2], 3, sector="TSR")
        assert check_yb_equivalence_catlab(b1, b2)

    def test_non_equivalent(self):
        b1 = create_braid_equation([1], 3, sector="TSR")
        b2 = create_braid_equation([2], 3, sector="TSR")
        assert not check_yb_equivalence_catlab(b1, b2)

    def test_identity_equivalence(self):
        b1 = create_braid_equation([], 3, sector="TSR")
        b2 = create_braid_equation([1, -1], 3, sector="TSR")
        assert check_yb_equivalence_catlab(b1, b2)

    def test_multiple_cancellations(self):
        b1 = create_braid_equation([1, 2, -2, -1], 3, sector="TSR")
        b2 = create_braid_equation([], 3, sector="TSR")
        assert check_yb_equivalence_catlab(b1, b2)

    def test_complex_equivalence(self):
        b1 = create_braid_equation([1, 2, 1, -1, -2], 3, sector="TSR")
        b2 = create_braid_equation([1], 3, sector="TSR")
        assert check_yb_equivalence_catlab(b1, b2)

    def test_different_sectors_not_equivalent(self):
        b_tsr = create_braid_equation([1, 2, 1], 3, sector="TSR")
        b_ising = create_braid_equation([1, 2, 1], 3, sector="Ising")
        assert not check_yb_equivalence_catlab(b_tsr, b_ising)

    def test_strand_count_mismatch_raises(self):
        b3 = create_braid_equation([1, 2], 3, sector="TSR")
        b4 = create_braid_equation([1, 2], 4, sector="TSR")
        with pytest.raises(ValueError):
            check_yb_equivalence_catlab(b3, b4)


# ========================================================================
# 3.6 Complexity Metrics
# ========================================================================
class TestBraidComplexity:
    def test_empty_braid(self):
        c = compute_braid_complexity(create_braid_equation([], 3, sector="TSR"))
        assert c.length == 0
        assert c.unique_generators == 0
        assert c.inverse_count == 0
        assert c.max_generator == 0

    def test_single_generator(self):
        c = compute_braid_complexity(create_braid_equation([1], 3, sector="TSR"))
        assert c.length == 1
        assert c.unique_generators == 1
        assert c.inverse_count == 0
        assert c.max_generator == 1

    def test_multiple_generators(self):
        c = compute_braid_complexity(create_braid_equation([1, 2, 1, 2], 3, sector="TSR"))
        assert c.length == 4
        assert c.unique_generators == 2
        assert c.inverse_count == 0
        assert c.max_generator == 2

    def test_with_inverse_generators(self):
        c = compute_braid_complexity(create_braid_equation([1, -2, 3, -1], 4, sector="TSR"))
        assert c.length == 4
        assert c.unique_generators == 3
        assert c.inverse_count == 2
        assert c.max_generator == 3

    def test_complex_braid(self):
        c = compute_braid_complexity(create_braid_equation([1, 2, 3, 4, 3, 2, 1], 5, sector="TSR"))
        assert c.length == 7
        assert c.unique_generators == 4
        assert c.max_generator == 4

    def test_all_negative(self):
        c = compute_braid_complexity(create_braid_equation([-1, -2, -3], 4, sector="TSR"))
        assert c.inverse_count == 3
        assert c.length == 3


# ========================================================================
# 3.7 Edge Cases
# ========================================================================
class TestEdgeCases:
    def test_min_strand_count(self):
        braid = create_braid_equation([1], 2, sector="TSR")
        assert validate_braid_category(braid)
        diag = create_braid_diagram([1], 2)
        assert len(diag) == 1

    def test_max_generator_index(self):
        braid = create_braid_equation([9], 10, sector="TSR")
        assert validate_braid_category(braid)

    def test_very_long_braid(self):
        long_gens = list(range(1, 10)) + list(range(9, 0, -1))
        braid = create_braid_equation(long_gens, 10, sector="TSR")
        assert validate_braid_category(braid)
        c = compute_braid_complexity(braid)
        assert c.length == 18

    def test_alternating_generators(self):
        braid = create_braid_equation([1, -1, 1, -1, 1], 3, sector="TSR")
        assert validate_braid_category(braid)

    def test_all_same_generator(self):
        braid = create_braid_equation([2, 2, 2, 2], 5, sector="TSR")
        assert validate_braid_category(braid)

    def test_sparse_non_adjacent_commute(self):
        braid1 = create_braid_equation([1, 3, 5, 7], 8, sector="TSR")
        assert validate_braid_category(braid1)
        braid2 = create_braid_equation([7, 5, 3, 1], 8, sector="TSR")
        assert verify_yang_baxter_morphism(braid1, braid2)

    def test_diagram_edge_cases(self):
        assert len(create_braid_diagram([], 5)) == 0
        assert len(create_braid_diagram([1], 2)) == 1

    def test_complexity_edge(self):
        c = compute_braid_complexity(create_braid_equation([], 3, sector="TSR"))
        assert c.max_generator == 0

    def test_yb_verify_identity_vs_non_identity(self):
        b_id = create_braid_equation([], 3, sector="TSR")
        b_any = create_braid_equation([1, 2], 3, sector="TSR")
        assert not verify_yang_baxter_morphism(b_id, b_any)


# ========================================================================
# 3.8 Integration with Braid Equations
# ========================================================================
class TestIntegration:
    def test_contracted_braid_unitary(self):
        braid = create_braid_equation([1, 2, 1], 3, sector="TSR")
        mat = contract_braid_tensor(braid)
        assert_allclose(mat.conj().T @ mat, np.eye(8, dtype=np.complex128), atol=1e-10)

    def test_jones_topological_invariant(self):
        b1 = create_braid_equation([1, 2, 1], 3, sector="TSR")
        b2 = create_braid_equation([2, 1, 2], 3, sector="TSR")
        V1 = jones_polynomial(b1)
        V2 = jones_polynomial(b2)
        assert pytest.approx(V2, abs=1e-10) == V1

    def test_validation_implies_unitarity(self):
        for i in range(1, 6):
            braid = create_braid_equation([i], 6, sector="TSR")
            if validate_braid_category(braid):
                mat = contract_braid_tensor(braid)
                assert_allclose(
                    mat.conj().T @ mat,
                    np.eye(64, dtype=np.complex128),
                    atol=1e-10,
                )

    def test_simplified_braids_preserve_equivalence(self):
        b_complex = create_braid_equation([1, -1, 2, -2, 3], 4, sector="TSR")
        b_simple = simplify_braid(b_complex)
        assert check_yb_equivalence_catlab(b_complex, b_simple)

    def test_registry_integration(self):
        braid = create_braid_equation([1, 2, 1], 3, sector="TSR")
        assert validate_braid_category(braid)

    @pytest.mark.parametrize("sector", ["TSR", "Ising", "Fibonacci", "SU2k2"])
    def test_cross_sector_validation_and_unitarity(self, sector):
        braid = create_braid_equation([1, 2, 1], 3, sector=sector)
        assert validate_braid_category(braid)
        mat = contract_braid_tensor(braid)
        assert_allclose(mat.conj().T @ mat, np.eye(8, dtype=np.complex128), atol=1e-10)

    def test_complexity_correlates_with_length(self):
        b_simple = create_braid_equation([1], 3, sector="TSR")
        b_complex = create_braid_equation([1, 2, 1, 2, 1, 2], 3, sector="TSR")
        c_s = compute_braid_complexity(b_simple)
        c_c = compute_braid_complexity(b_complex)
        assert c_c.length > c_s.length
        # Just confirm contraction completes
        contract_braid_tensor(b_simple)
        contract_braid_tensor(b_complex)

    def test_validation_does_not_mutate_braid(self):
        braid = create_braid_equation([1, 2, 3], 4, sector="TSR")
        gens_before = list(braid.generators)
        validate_braid_category(braid)
        assert braid.generators == gens_before


# ========================================================================
# 3.9 Performance
# ========================================================================
class TestPerformance:
    def test_validation_fast(self):
        braid = create_braid_equation(list(range(1, 10)), 10, sector="TSR")
        t0 = time.perf_counter()
        validate_braid_category(braid)
        assert time.perf_counter() - t0 < 0.1

    def test_diagram_creation_fast(self):
        t0 = time.perf_counter()
        for _ in range(100):
            create_braid_diagram([1, 2, 3], 5)
        assert time.perf_counter() - t0 < 0.5

    def test_yb_verification_fast(self):
        b1 = create_braid_equation([1, 2, 1, 2, 1], 3, sector="TSR")
        b2 = create_braid_equation([2, 1, 2, 1, 2], 3, sector="TSR")
        t0 = time.perf_counter()
        verify_yang_baxter_morphism(b1, b2)
        assert time.perf_counter() - t0 < 1.0

    def test_complexity_fast(self):
        braid = create_braid_equation(list(range(1, 51)), 51, sector="TSR")
        t0 = time.perf_counter()
        compute_braid_complexity(braid)
        assert time.perf_counter() - t0 < 0.01

    def test_equivalence_fast(self):
        b1 = create_braid_equation([1, 2, 3, 4, 5], 6, sector="TSR")
        b2 = create_braid_equation([1, 2, 3, 4, 5], 6, sector="TSR")
        t0 = time.perf_counter()
        check_yb_equivalence_catlab(b1, b2)
        assert time.perf_counter() - t0 < 1.0

    def test_batch_validation(self):
        braids = [create_braid_equation([i, i + 1], 5, sector="TSR") for i in range(1, 4)]
        t0 = time.perf_counter()
        for b in braids:
            validate_braid_category(b)
        assert time.perf_counter() - t0 < 0.5

    def test_validate_braid_category_bad_generator(self):
        """Cover the return False branch in validate_braid_category."""
        b = create_braid_equation([1, 2], 4, sector="TSR")
        # Manually inject an out-of-range generator to hit the branch
        b.generators.append(4)  # 4 >= n_strands=4
        assert validate_braid_category(b) is False
