"""
Tests for tsr_constants.py — Python mirror of Julia TSRConstants.jl.

Covers:
  1. Constant value verification (exact match to Julia)
  2. Derived constant consistency
  3. Physics function correctness (damping, coupling, Boltzmann, K_M)
  4. Numpy array broadcasting
  5. validate_tsr_constants() self-check
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from braidcodec.algebra.tsr_constants import (
    ALPHA,
    C_CONSTANT,
    C_LIGHT,
    COGNITIVE_TURN,
    D_PRIME,
    DEFAULT_HISTORY_DEPTH,
    DT_PSL,
    ELECTRON_MASS,
    ETA,
    FIXEDPOINT_MAX_ITER,
    FIXEDPOINT_TOL,
    GATE_SIGMA,
    HBAR,
    K_BOLTZMANN,
    K_M,
    KAPPA,
    L_CONTRACTION,
    OMEGA_ANCHOR,
    OMEGA_ANCHOR_HZ,
    OMEGA_MAX,
    OMEGA_MAX_HZ,
    OMEGA_MIN,
    OMEGA_MIN_HZ,
    PHI,
    Q_BASE,
    R_FLOOR,
    RHO_INFO,
    SQRT2_C,
    SQRT_2,
    boltzmann_weight,
    compute_coupling,
    compute_damping,
    effective_temperature,
    impedance_matching_exponent,
    validate_tsr_constants,
)


# =========================================================================
# 1. Fundamental constant values (must match Julia exactly)
# =========================================================================
class TestFundamentalConstants:
    def test_c_constant(self):
        assert pytest.approx(0.141416474, abs=1e-10) == C_CONSTANT

    def test_d_prime(self):
        assert pytest.approx(2.67e-16, abs=1e-20) == D_PRIME

    def test_hbar(self):
        assert pytest.approx(1.054571817e-34, abs=1e-44) == HBAR

    def test_alpha(self):
        assert pytest.approx(1.0 / 137.035999084, rel=1e-12) == ALPHA

    def test_k_boltzmann(self):
        assert pytest.approx(1.380649e-23, abs=1e-33) == K_BOLTZMANN

    def test_c_light(self):
        assert C_LIGHT == 299792458.0

    def test_electron_mass(self):
        assert pytest.approx(9.10938371e-31, abs=1e-40) == ELECTRON_MASS


# =========================================================================
# 2. Derived constant consistency
# =========================================================================
class TestDerivedConstants:
    def test_sqrt_2(self):
        assert pytest.approx(math.sqrt(2.0), abs=1e-15) == SQRT_2

    def test_phi_golden_ratio_value(self):
        assert pytest.approx((1.0 + math.sqrt(5.0)) / 2.0, abs=1e-15) == PHI

    def test_phi_golden_ratio_property(self):
        # φ² = φ + 1
        assert pytest.approx(PHI + 1, abs=1e-14) == PHI**2

    def test_rho_info_equals_phi(self):
        assert RHO_INFO == PHI

    def test_sqrt2_c(self):
        assert pytest.approx(SQRT_2 * C_CONSTANT, abs=1e-15) == SQRT2_C

    def test_cognitive_turn(self):
        assert pytest.approx(math.pi, abs=1e-15) == COGNITIVE_TURN

    def test_r_floor(self):
        assert pytest.approx(math.sqrt(2.0) * C_CONSTANT, abs=1e-15) == R_FLOOR

    def test_gate_sigma(self):
        assert pytest.approx(C_CONSTANT**2, abs=1e-15) == GATE_SIGMA


# =========================================================================
# 3. Oscillator constraints
# =========================================================================
class TestOscillatorConstraints:
    def test_omega_min(self):
        assert pytest.approx(2.0 * math.pi * OMEGA_MIN_HZ, abs=1e-12) == OMEGA_MIN

    def test_omega_max(self):
        assert pytest.approx(2.0 * math.pi * OMEGA_MAX_HZ, abs=1e-12) == OMEGA_MAX

    def test_omega_anchor(self):
        assert pytest.approx(2.0 * math.pi * OMEGA_ANCHOR_HZ, abs=1e-12) == OMEGA_ANCHOR

    def test_dt_psl(self):
        assert DT_PSL == 1.0e-4

    def test_q_base(self):
        assert Q_BASE == 5.0

    def test_frequency_ordering(self):
        assert OMEGA_MIN_HZ < OMEGA_ANCHOR_HZ < OMEGA_MAX_HZ


# =========================================================================
# 4. Causal sphere constants
# =========================================================================
class TestCausalSphereConstants:
    def test_kappa(self):
        assert KAPPA == 0.3

    def test_eta_formula(self):
        expected = math.sqrt(1.0 - KAPPA**2) - KAPPA
        assert pytest.approx(expected, abs=1e-15) == ETA

    def test_l_contraction_sum(self):
        assert pytest.approx(KAPPA + ETA, abs=1e-15) == L_CONTRACTION

    def test_l_contraction_less_than_one(self):
        assert L_CONTRACTION < 1.0

    def test_default_history_depth(self):
        assert DEFAULT_HISTORY_DEPTH == 16

    def test_fixedpoint_tol(self):
        assert FIXEDPOINT_TOL == 1e-7

    def test_fixedpoint_max_iter(self):
        assert FIXEDPOINT_MAX_ITER == 50


# =========================================================================
# 5. compute_damping
# =========================================================================
class TestComputeDamping:
    def test_scalar(self):
        omega = 1e9
        zeta = compute_damping(omega)
        expected = math.sqrt(D_PRIME / (HBAR * omega))
        assert float(zeta) == pytest.approx(expected, rel=1e-12)

    def test_known_value(self):
        # At ω = D'/ℏ, ζ should be 1.0
        omega_critical = D_PRIME / HBAR
        assert float(compute_damping(omega_critical)) == pytest.approx(1.0, abs=1e-12)

    def test_array_broadcast(self):
        omegas = np.array([1e8, 1e9, 1e10])
        zetas = compute_damping(omegas)
        assert zetas.shape == (3,)
        for i, omega in enumerate(omegas):
            expected = math.sqrt(D_PRIME / (HBAR * omega))
            assert float(zetas[i]) == pytest.approx(expected, rel=1e-12)

    def test_monotonic_decrease(self):
        # Higher frequency → lower damping
        omegas = np.logspace(6, 12, 20)
        zetas = compute_damping(omegas)
        assert np.all(np.diff(zetas) < 0)


# =========================================================================
# 6. compute_coupling
# =========================================================================
class TestComputeCoupling:
    def test_unit_coupling(self):
        # n=0, d=0: K = K0 * √2 * C^0 * exp(0) = K0 * √2
        K = compute_coupling(0.0, 0.0, 1.0, K0=1.0)
        assert float(K) == pytest.approx(SQRT_2, rel=1e-12)

    def test_decay_with_distance(self):
        lam = 0.1
        K_near = compute_coupling(1.0, 0.01, lam)
        K_far = compute_coupling(1.0, 1.0, lam)
        assert float(K_near) > float(K_far)

    def test_coupling_order(self):
        # Higher n → smaller coupling (C < 1 so C^n decreases)
        K_n1 = compute_coupling(1.0, 0.0, 1.0)
        K_n5 = compute_coupling(5.0, 0.0, 1.0)
        assert float(K_n1) > float(K_n5)

    def test_formula_match(self):
        n, d, lam, K0 = 2.0, 0.05, 0.1, 1e-3
        K = compute_coupling(n, d, lam, K0=K0)
        expected = K0 * SQRT_2 * (C_CONSTANT**n) * math.exp(-d / lam)
        assert float(K) == pytest.approx(expected, rel=1e-12)

    def test_array_broadcast(self):
        ns = np.array([1.0, 2.0, 3.0])
        ds = np.array([0.01, 0.02, 0.03])
        Ks = compute_coupling(ns, ds, 0.1)
        assert Ks.shape == (3,)
        for i in range(3):
            expected = SQRT_2 * (C_CONSTANT ** ns[i]) * math.exp(-ds[i] / 0.1)
            assert float(Ks[i]) == pytest.approx(expected, rel=1e-12)


# =========================================================================
# 7. impedance_matching_exponent
# =========================================================================
class TestImpedanceMatchingExponent:
    def test_same_frequency(self):
        # Same frequency → n = 0
        n = impedance_matching_exponent(100.0, 100.0)
        assert n == pytest.approx(0.0, abs=1e-12)

    def test_symmetry(self):
        n1 = impedance_matching_exponent(100.0, 200.0)
        n2 = impedance_matching_exponent(200.0, 100.0)
        assert n1 == pytest.approx(n2, abs=1e-12)

    def test_known_ratio(self):
        # If ω_j = ω_i / C, then n = 1
        omega_i = 1000.0
        omega_j = omega_i / C_CONSTANT
        n = impedance_matching_exponent(omega_i, omega_j)
        assert n == pytest.approx(1.0, abs=1e-10)

    def test_positive(self):
        n = impedance_matching_exponent(50.0, 500.0)
        assert n > 0


# =========================================================================
# 8. boltzmann_weight
# =========================================================================
class TestBoltzmannWeight:
    def test_zero_energy(self):
        # w(0) = exp(0) = 1
        assert float(boltzmann_weight(0.0)) == pytest.approx(1.0, abs=1e-15)

    def test_positive_energy(self):
        # w(E) < 1 for E > 0
        w = boltzmann_weight(1e-15)
        assert 0 < float(w) < 1

    def test_formula_match(self):
        E = 5e-16
        w = boltzmann_weight(E)
        expected = math.exp(-E / D_PRIME)
        assert float(w) == pytest.approx(expected, rel=1e-12)

    def test_array_broadcast(self):
        Es = np.array([0.0, 1e-16, 1e-15, 1e-14])
        ws = boltzmann_weight(Es)
        assert ws.shape == (4,)
        assert float(ws[0]) == pytest.approx(1.0, abs=1e-15)
        assert np.all(np.diff(ws) < 0)  # decreasing with energy


# =========================================================================
# 9. effective_temperature
# =========================================================================
class TestEffectiveTemperature:
    def test_value(self):
        T_eff = effective_temperature()
        expected = D_PRIME / K_BOLTZMANN
        assert T_eff == pytest.approx(expected, rel=1e-12)

    def test_order_of_magnitude(self):
        T_eff = effective_temperature()
        assert 1e6 < T_eff < 1e8  # ~1.93e7 K


# =========================================================================
# 10. K_M fixed-point operator
# =========================================================================
class TestKM:
    def test_scalar(self):
        x = 1.0
        result = K_M(x)
        expected = KAPPA * x + ETA * math.sin(x)
        assert float(result) == pytest.approx(expected, abs=1e-15)

    def test_zero(self):
        # K_M(0) = κ·0 + η·sin(0) = 0
        assert float(K_M(0.0)) == pytest.approx(0.0, abs=1e-15)

    def test_fixed_point_convergence(self):
        """Iterate K_M and verify convergence toward fixed point x*=0."""
        x = 1.0
        for _ in range(500):
            x = float(K_M(x))
        # The unique fixed point of K_M is x* = 0
        # With L ≈ 0.954, convergence is slow but monotonic
        assert abs(x) < 1e-9

    def test_fixed_point_from_different_initial(self):
        """Different initial conditions converge to same fixed point (0)."""
        results = []
        for x0 in [0.1, 1.0, 3.0, -2.0]:
            x = x0
            for _ in range(500):
                x = float(K_M(x))
            results.append(x)
        # All should converge to same fixed point
        for r in results[1:]:
            assert r == pytest.approx(results[0], abs=1e-6)

    def test_array_broadcast(self):
        xs = np.array([0.0, 1.0, 2.0, 3.0])
        results = K_M(xs)
        assert results.shape == (4,)
        for i, x in enumerate(xs):
            expected = KAPPA * x + ETA * math.sin(x)
            assert float(results[i]) == pytest.approx(expected, abs=1e-14)

    def test_contraction_property(self):
        """Verify |K_M(x) - K_M(y)| <= L|x - y| for several pairs."""
        pairs = [(0.5, 1.5), (1.0, 3.0), (-1.0, 2.0), (0.1, 0.2)]
        for x, y in pairs:
            lhs = abs(float(K_M(x)) - float(K_M(y)))
            rhs = L_CONTRACTION * abs(x - y)
            assert lhs <= rhs + 1e-14


# =========================================================================
# 11. validate_tsr_constants
# =========================================================================
class TestValidation:
    def test_all_constants_valid(self):
        assert validate_tsr_constants() is True

    def test_returns_bool(self):
        result = validate_tsr_constants()
        assert isinstance(result, bool)


# =========================================================================
# 12. Cross-validation with Julia values
# =========================================================================
class TestCrossValidationWithJulia:
    """
    Verify computed values match the Julia module output to machine precision.
    These are regression anchors — if any fail, Python/Julia parity is broken.
    """

    def test_damping_at_1ghz(self):
        # Julia: compute_damping(1e9) → sqrt(2.67e-16 / (1.054571817e-34 * 1e9))
        omega = 1e9
        expected = math.sqrt(2.67e-16 / (1.054571817e-34 * 1e9))
        assert float(compute_damping(omega)) == pytest.approx(expected, rel=1e-14)

    def test_coupling_n2_d001_lam01(self):
        n, d, lam = 2.0, 0.01, 0.1
        expected = SQRT_2 * (C_CONSTANT**2) * math.exp(-0.01 / 0.1)
        assert float(compute_coupling(n, d, lam)) == pytest.approx(expected, rel=1e-14)

    def test_boltzmann_at_d_prime(self):
        # w(D') = exp(-1) ≈ 0.3679
        assert float(boltzmann_weight(D_PRIME)) == pytest.approx(math.exp(-1), rel=1e-14)

    def test_eta_value(self):
        expected_eta = math.sqrt(1.0 - 0.3**2) - 0.3
        assert pytest.approx(expected_eta, abs=1e-15) == ETA

    def test_l_contraction_value(self):
        expected_L = 0.3 + math.sqrt(1.0 - 0.3**2) - 0.3
        assert pytest.approx(expected_L, abs=1e-15) == L_CONTRACTION

    def test_effective_temp_value(self):
        expected = 2.67e-16 / 1.380649e-23
        assert effective_temperature() == pytest.approx(expected, rel=1e-14)


class TestValidateFailureBranches:
    """Cover the failure branches inside validate_tsr_constants."""

    def test_validate_passes(self):
        assert validate_tsr_constants() is True

    def test_validate_c_constant_out_of_range(self, monkeypatch):
        import braidcodec.algebra.tsr_constants as mod

        monkeypatch.setattr(mod, "C_CONSTANT", -1.0)
        with pytest.warns(UserWarning, match="C_CONSTANT out of range"):
            assert validate_tsr_constants() is False

    def test_validate_d_prime_invalid(self, monkeypatch):
        import braidcodec.algebra.tsr_constants as mod

        monkeypatch.setattr(mod, "D_PRIME", -1.0)
        with pytest.warns(UserWarning, match="D_PRIME invalid"):
            assert validate_tsr_constants() is False

    def test_validate_hbar_mismatch(self, monkeypatch):
        import braidcodec.algebra.tsr_constants as mod

        monkeypatch.setattr(mod, "HBAR", 999.0)
        with pytest.warns(UserWarning, match="HBAR deviates"):
            assert validate_tsr_constants() is False

    def test_validate_alpha_mismatch(self, monkeypatch):
        import braidcodec.algebra.tsr_constants as mod

        monkeypatch.setattr(mod, "ALPHA", 999.0)
        with pytest.warns(UserWarning, match="ALPHA deviates"):
            assert validate_tsr_constants() is False

    def test_validate_phi_mismatch(self, monkeypatch):
        import braidcodec.algebra.tsr_constants as mod

        monkeypatch.setattr(mod, "PHI", 999.0)
        with pytest.warns(UserWarning, match="golden ratio"):
            assert validate_tsr_constants() is False

    def test_validate_l_contraction_ge_1(self, monkeypatch):
        import braidcodec.algebra.tsr_constants as mod

        monkeypatch.setattr(mod, "L_CONTRACTION", 2.0)
        with pytest.warns(UserWarning, match="L_CONTRACTION"):
            assert validate_tsr_constants() is False

    def test_validate_sqrt2c_mismatch(self, monkeypatch):
        import braidcodec.algebra.tsr_constants as mod

        monkeypatch.setattr(mod, "SQRT2_C", 999.0)
        with pytest.warns(UserWarning, match="SQRT2_C derivation"):
            assert validate_tsr_constants() is False

    def test_validate_kappa_eta_mismatch(self, monkeypatch):
        import braidcodec.algebra.tsr_constants as mod

        monkeypatch.setattr(mod, "KAPPA", 999.0)
        with pytest.warns(UserWarning, match="KAPPA\\+ETA"):
            assert validate_tsr_constants() is False
