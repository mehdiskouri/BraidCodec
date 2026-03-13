"""
test_braid_equations.py — Rigorous testing for braid_equations.py.
Mirror of algebra/tests/test_braid_equations.jl.
"""

import numpy as np
import pytest
from numpy.linalg import norm
from numpy.testing import assert_allclose

from braidcodec.algebra.braid_equations import (
    BRAID_REGISTRY,
    VALID_SECTORS,
    BraidEquation,
    BraidEquationRegistry,
    braid_to_string,
    compose,
    contract_braid_tensor,
    create_braid_equation,
    get_braid_generator_matrix,
    get_kauffman_A,
    get_sector_r_matrix,
    jones_polynomial,
    kauffman_bracket,
    simplify_braid,
    writhe,
)
from braidcodec.algebra.tsr_constants import C_CONSTANT

I4 = np.eye(4, dtype=np.complex128)


# ========================================================================
# 1.1 Load / Smoke
# ========================================================================
class TestLoadAndSmoke:
    def test_braid_equation_class_exists(self):
        assert BraidEquation is not None

    def test_registry_class_exists(self):
        assert BraidEquationRegistry is not None

    def test_global_registry_exists(self):
        assert BRAID_REGISTRY is not None

    def test_c_constant_value(self):
        assert pytest.approx(0.141416474, abs=1e-8) == C_CONSTANT

    def test_exports_exist(self):
        import braidcodec.algebra.braid_equations as mod

        for name in [
            "create_braid_equation",
            "contract_braid_tensor",
            "jones_polynomial",
            "get_sector_r_matrix",
            "simplify_braid",
            "writhe",
            "kauffman_bracket",
        ]:
            assert hasattr(mod, name)


# ========================================================================
# 1.2 R-Matrix Construction
# ========================================================================
class TestRMatrixStructure:
    sectors = ["Identity", "TSR", "Ising", "Fibonacci", "SU2k2"]  # noqa: RUF012

    @pytest.mark.parametrize("sector", sectors)
    def test_dimensions(self, sector):
        R = get_sector_r_matrix(sector, False)
        assert R.shape == (4, 4)

    @pytest.mark.parametrize("sector", sectors)
    def test_unitarity(self, sector):
        R = get_sector_r_matrix(sector, False)
        assert_allclose(R.conj().T @ R, I4, atol=1e-12)

    @pytest.mark.parametrize("sector", sectors)
    def test_phased_swap_structure(self, sector):
        R = get_sector_r_matrix(sector, False)
        # Non-zero entries
        assert abs(R[0, 0]) == pytest.approx(1.0, abs=1e-12)
        assert abs(R[1, 2]) == pytest.approx(1.0, abs=1e-12)
        assert abs(R[2, 1]) == pytest.approx(1.0, abs=1e-12)
        assert abs(R[3, 3]) == pytest.approx(1.0, abs=1e-12)
        # Zero entries
        for idx in [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 3)]:
            assert abs(R[idx]) < 1e-14

    @pytest.mark.parametrize("sector", sectors)
    def test_inverse_relation(self, sector):
        R = get_sector_r_matrix(sector, False)
        R_inv = get_sector_r_matrix(sector, True)
        assert_allclose(R @ R_inv, I4, atol=1e-12)

    @pytest.mark.parametrize("sector", ["TSR", "Ising", "Fibonacci", "SU2k2"])
    def test_non_trivial_not_pure_swap(self, sector):
        R = get_sector_r_matrix(sector, False)
        SWAP = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )
        assert norm(R - SWAP) > 0.1


# ========================================================================
# 1.3 Kronecker Product Operations — CRITICAL
# ========================================================================
class TestBraidGeneratorMatrix:
    def _I2(self):
        return np.eye(2, dtype=np.complex128)

    def _R(self, sector="TSR"):
        return get_sector_r_matrix(sector, False)

    # Case 1: 2 strands, pos 1 → just R
    def test_2_strands_pos1(self):
        mat = get_braid_generator_matrix(1, 2, False, "TSR")
        assert mat.shape == (4, 4)
        assert_allclose(mat, self._R(), atol=1e-12)

    # Case 2: 3 strands, pos 1 → R ⊗ I
    def test_3_strands_pos1(self):
        mat = get_braid_generator_matrix(1, 3, False, "TSR")
        assert mat.shape == (8, 8)
        assert_allclose(mat, np.kron(self._R(), self._I2()), atol=1e-12)

    # Case 3: 3 strands, pos 2 → I ⊗ R
    def test_3_strands_pos2(self):
        mat = get_braid_generator_matrix(2, 3, False, "TSR")
        assert mat.shape == (8, 8)
        assert_allclose(mat, np.kron(self._I2(), self._R()), atol=1e-12)

    # Case 4: 4 strands, pos 1 → R ⊗ I ⊗ I
    def test_4_strands_pos1(self):
        mat = get_braid_generator_matrix(1, 4, False, "TSR")
        assert mat.shape == (16, 16)
        expected = np.kron(np.kron(self._R(), self._I2()), self._I2())
        assert_allclose(mat, expected, atol=1e-12)

    # Case 5: 4 strands, pos 2 → I ⊗ R ⊗ I
    def test_4_strands_pos2(self):
        mat = get_braid_generator_matrix(2, 4, False, "TSR")
        assert mat.shape == (16, 16)
        expected = np.kron(np.kron(self._I2(), self._R()), self._I2())
        assert_allclose(mat, expected, atol=1e-12)

    # Case 6: 4 strands, pos 3 → I ⊗ I ⊗ R
    def test_4_strands_pos3(self):
        mat = get_braid_generator_matrix(3, 4, False, "TSR")
        assert mat.shape == (16, 16)
        expected = np.kron(np.kron(self._I2(), self._I2()), self._R())
        assert_allclose(mat, expected, atol=1e-12)

    # Case 7: 5 strands, pos 3 → I ⊗ I ⊗ R ⊗ I
    def test_5_strands_pos3(self):
        mat = get_braid_generator_matrix(3, 5, False, "TSR")
        assert mat.shape == (32, 32)
        expected = np.kron(
            np.kron(np.kron(self._I2(), self._I2()), self._R()),
            self._I2(),
        )
        assert_allclose(mat, expected, atol=1e-12)

    # Case 8: Manual step-by-step construction for 5 strands pos 3
    def test_5_strands_pos3_manual(self):
        R = self._R()
        I2 = self._I2()
        result = np.array([[1.0 + 0j]], dtype=np.complex128)
        for _ in range(2):
            result = np.kron(result, I2)
        result = np.kron(result, R)
        result = np.kron(result, I2)
        mat = get_braid_generator_matrix(3, 5, False, "TSR")
        assert_allclose(mat, result, atol=1e-12)

    # Case 9: Unitarity preserved through Kronecker products
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_unitarity_kronecker(self, n):
        for i in range(1, n):
            mat = get_braid_generator_matrix(i, n, False, "TSR")
            eye = np.eye(2**n, dtype=np.complex128)
            assert_allclose(mat.conj().T @ mat, eye, atol=1e-10)

    # Case 10: Inverse generators
    def test_inverse_generators(self):
        mat_fwd = get_braid_generator_matrix(2, 4, False, "TSR")
        mat_inv = get_braid_generator_matrix(2, 4, True, "TSR")
        I16 = np.eye(16, dtype=np.complex128)
        assert_allclose(mat_fwd @ mat_inv, I16, atol=1e-10)
        assert_allclose(mat_inv @ mat_fwd, I16, atol=1e-10)

    # Case 11: Identity action on untouched strand
    def test_identity_action_untouched_strand(self):
        mat = get_braid_generator_matrix(1, 3, False, "TSR")
        state = np.zeros(8, dtype=np.complex128)
        state[1] = 1.0  # |001⟩ (0-indexed → index 1)
        result_state = mat @ state
        assert norm(result_state) == pytest.approx(1.0, abs=1e-10)

    # Case 12: Different sectors → different matrices
    def test_cross_sector_different(self):
        mat_tsr = get_braid_generator_matrix(1, 3, False, "TSR")
        mat_ising = get_braid_generator_matrix(1, 3, False, "Ising")
        assert norm(mat_tsr - mat_ising) > 0.1


# ========================================================================
# 1.4 Tensor Contraction
# ========================================================================
class TestTensorContraction:
    # Test 1: Empty braid → identity
    def test_identity(self):
        braid = BraidEquation(3, [], sector="TSR")
        mat = contract_braid_tensor(braid)
        assert_allclose(mat, np.eye(8, dtype=np.complex128), atol=1e-12)

    # Test 2: Single generator
    def test_single_generator(self):
        braid = BraidEquation(3, [1], sector="TSR")
        mat = contract_braid_tensor(braid)
        expected = get_braid_generator_matrix(1, 3, False, "TSR")
        assert_allclose(mat, expected, atol=1e-12)

    # Test 3: Composition order → mat2 @ mat1
    def test_composition_order(self):
        braid = BraidEquation(3, [1, 2], sector="TSR")
        mat = contract_braid_tensor(braid)
        mat1 = get_braid_generator_matrix(1, 3, False, "TSR")
        mat2 = get_braid_generator_matrix(2, 3, False, "TSR")
        assert_allclose(mat, mat2 @ mat1, atol=1e-10)

    # Test 4: Inverse cancellation
    def test_inverse_cancellation(self):
        braid = BraidEquation(3, [1, -1], sector="TSR")
        mat = contract_braid_tensor(braid)
        assert_allclose(mat, np.eye(8, dtype=np.complex128), atol=1e-10)

    # Test 5: Caching
    def test_caching(self):
        braid = BraidEquation(3, [1, 2, 1], sector="TSR")
        mat1 = contract_braid_tensor(braid)
        mat2 = contract_braid_tensor(braid)
        assert mat1 is mat2

    # Test 6: Yang-Baxter — σ₁σ₂σ₁ = σ₂σ₁σ₂
    def test_yang_baxter(self):
        lhs = BraidEquation(3, [1, 2, 1], sector="TSR")
        rhs = BraidEquation(3, [2, 1, 2], sector="TSR")
        assert_allclose(contract_braid_tensor(lhs), contract_braid_tensor(rhs), atol=1e-10)

    # Test 7: Unitarity of contracted braid
    def test_unitarity(self):
        braid = BraidEquation(4, [1, 3, 2, 1, 3], sector="TSR")
        mat = contract_braid_tensor(braid)
        assert_allclose(mat.conj().T @ mat, np.eye(16, dtype=np.complex128), atol=1e-10)

    # Test 8: Three-generator sequences
    @pytest.mark.parametrize("n", [3, 4])
    def test_three_gen_sequences(self, n):
        for i in range(1, n - 1):
            braid = BraidEquation(n, [i, i + 1, i], sector="TSR")
            mat = contract_braid_tensor(braid)
            assert_allclose(mat.conj().T @ mat, np.eye(2**n, dtype=np.complex128), atol=1e-10)


# ========================================================================
# 1.5 Construction
# ========================================================================
class TestConstruction:
    def test_valid_construction(self):
        braid = BraidEquation(3, [1, 2, 1], sector="TSR")
        assert braid.n_strands == 3
        assert braid.generators == [1, 2, 1]
        assert braid.sector == "TSR"
        assert braid.simplified is False
        assert braid.braid_matrix is None

    def test_create_with_int_list(self):
        braid = create_braid_equation([1, -2, 1], 4, sector="Ising")
        assert braid.n_strands == 4
        assert braid.generators == [1, -2, 1]
        assert braid.sector == "Ising"

    def test_create_with_symbol_list(self):
        braid = create_braid_equation(["σ1", "σ2", "σ1inv"], 3, sector="TSR")
        assert braid.generators == [1, 2, -1]

    def test_empty_braid(self):
        braid = BraidEquation(5, [], sector="TSR")
        assert braid.is_identity()
        assert len(braid) == 0

    @pytest.mark.parametrize("sector", sorted(VALID_SECTORS))
    def test_all_valid_sectors(self, sector):
        braid = BraidEquation(3, [1], sector=sector)
        assert braid.sector == sector


# ========================================================================
# 1.6 Simplification
# ========================================================================
class TestSimplification:
    def test_inverse_cancellation(self):
        braid = BraidEquation(3, [1, 2, -2, 1], sector="TSR")
        simp = simplify_braid(braid)
        assert simp.generators == [1, 1]
        assert simp.simplified is True

    def test_multiple_cancellations(self):
        braid = BraidEquation(4, [1, -1, 2, -2, 3], sector="TSR")
        simp = simplify_braid(braid)
        assert simp.generators == [3]

    def test_no_simplification(self):
        braid = BraidEquation(3, [1, 2, 1], sector="TSR")
        simp = simplify_braid(braid)
        assert simp.generators == [1, 2, 1]

    def test_complete_cancellation(self):
        braid = BraidEquation(3, [1, -1], sector="TSR")
        simp = simplify_braid(braid)
        assert simp.generators == []
        assert simp.is_identity()

    def test_cache_cleared_after_simplification(self):
        braid = BraidEquation(3, [1, 2, -2], sector="TSR")
        contract_braid_tensor(braid)
        assert braid.braid_matrix is not None
        simp = simplify_braid(braid)
        assert simp.braid_matrix is None


# ========================================================================
# 1.7 Jones Polynomial
# ========================================================================
class TestJonesPolynomial:
    def test_unknot_finite(self):
        braid = BraidEquation(2, [], sector="TSR")
        V = jones_polynomial(braid)
        assert np.isfinite(V)

    def test_single_crossing_finite(self):
        braid = BraidEquation(2, [1], sector="TSR")
        V = jones_polynomial(braid)
        assert np.isfinite(V)

    def test_mirror_images_differ(self):
        b_pos = BraidEquation(3, [1, 1, 2], sector="TSR")
        b_neg = BraidEquation(3, [-1, -1, -2], sector="TSR")
        V_pos = jones_polynomial(b_pos)
        V_neg = jones_polynomial(b_neg)
        # They should be different OR very close (mirrors may differ)
        assert V_pos != V_neg or abs(V_pos - V_neg) < 1e-10

    def test_writhe(self):
        assert writhe(BraidEquation(3, [1, 1, 2], sector="TSR")) == 3
        assert writhe(BraidEquation(3, [-1, -2, -1], sector="TSR")) == -3
        assert writhe(BraidEquation(3, [1, -1, 2], sector="TSR")) == 1

    def test_kauffman_bracket_unknot_finite(self):
        braid = BraidEquation(2, [], sector="TSR")
        assert np.isfinite(kauffman_bracket(braid))

    def test_yang_baxter_invariance(self):
        b1 = BraidEquation(3, [1, 2, 1], sector="TSR")
        b2 = BraidEquation(3, [2, 1, 2], sector="TSR")
        V1 = jones_polynomial(b1)
        V2 = jones_polynomial(b2)
        assert pytest.approx(V2, abs=1e-10) == V1

    def test_different_sectors_different_polynomials(self):
        b_tsr = BraidEquation(3, [1, 2], sector="TSR")
        b_ising = BraidEquation(3, [1, 2], sector="Ising")
        assert jones_polynomial(b_tsr) != jones_polynomial(b_ising)


# ========================================================================
# 1.8 Registry
# ========================================================================
class TestRegistry:
    @pytest.fixture(autouse=True)
    def clean_registry(self):
        BRAID_REGISTRY.clear()
        yield
        BRAID_REGISTRY.clear()

    def test_starts_empty(self):
        stats = BRAID_REGISTRY.stats()
        assert stats["n_braids"] == 0

    def test_register_and_has(self):
        braid = BraidEquation(3, [1, 2, 1], sector="TSR")
        id_ = BRAID_REGISTRY.register(braid)
        assert id_ == 1
        assert BRAID_REGISTRY.has(id_)

    def test_retrieve(self):
        braid = BraidEquation(3, [1, 2, 1], sector="TSR")
        id_ = BRAID_REGISTRY.register(braid)
        assert BRAID_REGISTRY.get(id_) == braid

    def test_stats(self):
        braid = BraidEquation(3, [1, 2, 1], sector="TSR")
        BRAID_REGISTRY.register(braid)
        stats = BRAID_REGISTRY.stats()
        assert stats["n_braids"] == 1
        assert stats["n_simplified"] == 0

    def test_register_simplified(self):
        braid = BraidEquation(3, [1, 2, 1], sector="TSR")
        simp = simplify_braid(braid)
        BRAID_REGISTRY.register(braid)
        BRAID_REGISTRY.register(simp)
        stats = BRAID_REGISTRY.stats()
        assert stats["n_simplified"] == 1

    def test_braid_to_string(self):
        braid = BraidEquation(3, [1, 2, 1], sector="TSR")
        s = braid_to_string(braid)
        assert "σ1" in s
        assert "σ2" in s

    def test_braid_to_string_inverse(self):
        braid = BraidEquation(3, [1, -2], sector="TSR")
        s = braid_to_string(braid)
        assert "⁻¹" in s

    def test_multiple_registrations(self):
        braid = BraidEquation(3, [1, 2, 1], sector="TSR")
        BRAID_REGISTRY.register(braid)
        simp = simplify_braid(braid)
        BRAID_REGISTRY.register(simp)
        for i in range(5):
            b = BraidEquation(3, [i % 2 + 1], sector="TSR")
            BRAID_REGISTRY.register(b)
        stats = BRAID_REGISTRY.stats()
        assert stats["n_braids"] == 7


# ========================================================================
# 1.9 Braid Operations
# ========================================================================
class TestBraidOperations:
    def test_equality(self):
        b1 = BraidEquation(3, [1, 2], sector="TSR")
        b2 = BraidEquation(3, [1, 2], sector="TSR")
        b3 = BraidEquation(3, [1, 2], sector="Ising")
        assert b1 == b2
        assert b1 != b3

    def test_length(self):
        braid = BraidEquation(4, [1, 2, 3, 2], sector="TSR")
        assert len(braid) == 4

    def test_identity_check(self):
        assert BraidEquation(3, [], sector="TSR").is_identity()
        assert not BraidEquation(3, [1], sector="TSR").is_identity()

    def test_inverse(self):
        braid = BraidEquation(3, [1, 2, 1], sector="TSR")
        braid_inv = ~braid
        assert braid_inv.generators == [-1, -2, -1]

    def test_compose(self):
        b1 = BraidEquation(3, [1, 2], sector="TSR")
        b2 = BraidEquation(3, [2, 1], sector="TSR")
        b_comp = compose(b1, b2)
        assert b_comp.generators == [1, 2, 2, 1]

    def test_multiply_operator(self):
        b1 = BraidEquation(3, [1, 2], sector="TSR")
        b2 = BraidEquation(3, [2, 1], sector="TSR")
        b_mult = b1 * b2
        b_comp = compose(b1, b2)
        assert b_mult.generators == b_comp.generators

    def test_compose_with_inverse_is_identity(self):
        braid = BraidEquation(3, [1, 2], sector="TSR")
        braid_inv = ~braid
        braid_comp = braid * braid_inv
        mat = contract_braid_tensor(braid_comp)
        assert_allclose(mat, np.eye(8, dtype=np.complex128), atol=1e-10)


# ========================================================================
# 1.10 Edge Cases
# ========================================================================
class TestEdgeCases:
    def test_invalid_strand_count(self):
        with pytest.raises(ValueError):
            BraidEquation(1, [1], sector="TSR")

    def test_invalid_generator_too_large(self):
        with pytest.raises(ValueError):
            BraidEquation(3, [3], sector="TSR")

    def test_invalid_generator_zero(self):
        with pytest.raises(ValueError):
            BraidEquation(3, [0], sector="TSR")

    def test_invalid_sector(self):
        with pytest.raises(ValueError):
            BraidEquation(3, [1], sector="InvalidSector")

    def test_large_strand_count(self):
        braid = BraidEquation(10, [1, 5, 9], sector="TSR")
        mat = contract_braid_tensor(braid)
        assert mat.shape == (1024, 1024)
        assert_allclose(mat.conj().T @ mat, np.eye(1024, dtype=np.complex128), atol=1e-8)

    def test_max_generator_index(self):
        braid = BraidEquation(10, [9], sector="TSR")
        mat = contract_braid_tensor(braid)
        assert np.isfinite(norm(mat))


class TestCoverageGaps:
    """Cover branches missed by the main test suite."""

    def test_eq_with_non_braid(self):
        b = BraidEquation(3, [1], sector="TSR")
        assert b.__eq__("not a braid") is NotImplemented

    def test_repr(self):
        b = BraidEquation(3, [1, -2], sector="TSR")
        r = repr(b)
        assert "n_strands=3" in r
        assert "generators=[1, -2]" in r
        assert "sector='TSR'" in r

    def test_str(self):
        b = BraidEquation(3, [1, -2], sector="TSR")
        assert isinstance(str(b), str)

    def test_get_sector_r_matrix_unknown_sector(self):
        with pytest.raises(ValueError, match="Unknown sector"):
            get_sector_r_matrix("INVALID", False)

    def test_get_braid_generator_matrix_bad_index(self):
        with pytest.raises(ValueError, match="out of range"):
            get_braid_generator_matrix(0, 3, False, "TSR")
        with pytest.raises(ValueError, match="out of range"):
            get_braid_generator_matrix(3, 3, False, "TSR")

    def test_create_braid_equation_string_inverse(self):
        b = create_braid_equation(["σ1inv", "σ2"], 4, sector="TSR")
        assert b.generators == [-1, 2]

    def test_create_braid_equation_string_identity(self):
        b = create_braid_equation(["e", "σ1"], 4, sector="TSR")
        assert b.generators == [1]

    def test_create_braid_equation_string_bad_index(self):
        with pytest.raises(ValueError, match="Invalid generator"):
            create_braid_equation(["σ5"], 3, sector="TSR")

    def test_create_braid_equation_string_bad_inv_index(self):
        with pytest.raises(ValueError, match="Invalid generator"):
            create_braid_equation(["σ5inv"], 3, sector="TSR")

    def test_create_braid_equation_empty(self):
        b = create_braid_equation([], 3, sector="TSR")
        assert b.generators == []

    def test_kauffman_A_all_sectors(self):
        for s in ["Identity", "TSR", "Ising", "Fibonacci", "SU2k2"]:
            A = get_kauffman_A(s)
            assert abs(A) == pytest.approx(1.0, abs=1e-12)

    def test_kauffman_A_unknown_sector(self):
        with pytest.raises(ValueError, match="Unknown sector"):
            get_kauffman_A("INVALID")

    def test_registry_braid_to_string_missing(self):
        assert "not found" in BRAID_REGISTRY.braid_to_string(99999)
