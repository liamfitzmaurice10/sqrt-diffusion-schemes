"""Tests for the CIR schemes, the exact sampler, the errors and Heston."""

import numpy as np
import pytest
from scipy.stats import norm

from sqrtdiff import errors, exact
from sqrtdiff.heston import (
    HestonParams,
    analytic_call,
    correlated_increments,
    european_call,
    simulate_from_increments,
)
from sqrtdiff.schemes import SCHEMES, CIRParams, lamperti_implicit

BASE = CIRParams(kappa=1.0, lam=0.2, sigma=0.1, S0=0.2)

# Schemes

@pytest.mark.parametrize(
    "scheme",
    [SCHEMES[k] for k in SCHEMES if k != "Lamperti implicit"],
    ids=[k for k in SCHEMES if k != "Lamperti implicit"],
)
def test_zero_noise_recovers_the_deterministic_solution(scheme):
    """With dB = 0 the explicit schemes track the ODE dS = kappa*(lam - S)dt."""
    n = 2000
    dB = np.zeros((1, n))
    S = scheme(BASE, dB)[0, -1]
    assert S == pytest.approx(BASE.exact_mean(BASE.T), abs=1e-3)


def test_lamperti_zero_noise_limit_is_not_the_drift_ode():
    """Setting dB = 0 in Y-space leaves the Ito correction in place.

    The Lamperti drift is alpha/Y + beta*Y with alpha = (4*kappa*lam -
    sigma**2)/8, so the noise-free recursion solves du/dt = 2*alpha + 2*beta*u
    for u = Y**2, giving

        u(t) = (S0 - 2*alpha/kappa)*exp(-kappa*t) + 2*alpha/kappa,

    whose steady state is lam - sigma**2/(4*kappa), NOT lam. This is a
    property of the transform, not a defect of the implementation, and it is
    worth pinning down because it looks like a bug the first time it is seen.
    """
    n = 2000
    S = lamperti_implicit(BASE, np.zeros((1, n)))[0, -1]
    steady = 2 * BASE.alpha / BASE.kappa
    expected = (BASE.S0 - steady) * np.exp(-BASE.kappa * BASE.T) + steady
    assert steady == pytest.approx(BASE.lam - BASE.sigma**2 / (4 * BASE.kappa))
    assert S == pytest.approx(expected, abs=1e-5)


def test_schemes_coincide_while_paths_stay_positive():
    """At sigma = 0.1 the process never approaches zero, so the three explicit
    schemes are the same recursion and must agree to machine precision."""
    rng = np.random.default_rng(11)
    dB = errors.fine_increments(BASE.T, 128, 500, rng)
    absolute = SCHEMES["Explicit EM (absolute)"](BASE, dB)
    truncated = SCHEMES["Explicit EM (truncated)"](BASE, dB)
    reflected = SCHEMES["Explicit EM (reflected)"](BASE, dB)
    assert np.min(absolute) > 0
    assert np.allclose(absolute, truncated, rtol=0, atol=1e-14)
    assert np.allclose(absolute, reflected, rtol=0, atol=1e-14)


def test_reflected_scheme_is_non_negative_under_violated_feller():
    p = CIRParams(kappa=1.0, lam=0.2, sigma=1.2, S0=0.2)
    rng = np.random.default_rng(12)
    dB = errors.fine_increments(p.T, 64, 2000, rng)
    S = SCHEMES["Explicit EM (reflected)"](p, dB)
    assert np.all(S >= 0.0)


def test_absolute_scheme_does_go_negative_under_violated_feller():
    """The point of the comparison: this scheme is not positivity-preserving."""
    p = CIRParams(kappa=1.0, lam=0.2, sigma=1.2, S0=0.2)
    rng = np.random.default_rng(13)
    dB = errors.fine_increments(p.T, 64, 2000, rng)
    S = SCHEMES["Explicit EM (absolute)"](p, dB)
    assert errors.negative_fraction(S[:, -1]) > 0.0


def test_lamperti_fails_only_when_alpha_is_negative():
    rng = np.random.default_rng(14)
    ok = CIRParams(kappa=1.0, lam=0.2, sigma=0.8, S0=0.2)  # alpha > 0
    bad = CIRParams(kappa=1.0, lam=0.2, sigma=1.2, S0=0.2)  # alpha < 0
    assert ok.alpha > 0 and bad.alpha < 0
    dB = errors.fine_increments(1.0, 32, 5000, rng)
    assert errors.nan_fraction(lamperti_implicit(ok, dB)[:, -1]) == 0.0
    assert errors.nan_fraction(lamperti_implicit(bad, dB)[:, -1]) > 0.0


# Exact sampling


def test_exact_sampler_matches_the_first_two_moments():
    rng = np.random.default_rng(21)
    x = exact.sample_terminal(BASE, 200_000, rng)
    err, se = errors.weak_error(x, BASE)
    assert err < 4 * se
    assert x.var() == pytest.approx(BASE.exact_var(BASE.T), rel=0.03)
    assert np.all(x >= 0.0)

@pytest.mark.parametrize("sigma", [0.1, 0.8])
def test_call_expectation_matches_monte_carlo(sigma):
    p = CIRParams(kappa=1.0, lam=0.2, sigma=sigma, S0=0.2)
    rng = np.random.default_rng(23)
    x = exact.sample_terminal(p, 400_000, rng)
    payoff = np.maximum(x - 0.2, 0.0)
    err, se = errors.weak_error_functional(payoff, exact.call_expectation(p, 0.2))
    assert err < 4 * se


# Errors

def test_coarsening_preserves_the_path():
    rng = np.random.default_rng(31)
    fine = errors.fine_increments(1.0, 64, 10, rng)
    coarse = errors.coarsen(fine, 8)
    assert coarse.shape == (10, 8)
    assert np.allclose(coarse.sum(axis=1), fine.sum(axis=1))
    assert np.allclose(coarse[:, 0], fine[:, :8].sum(axis=1))


def test_fit_order_recovers_a_known_slope():
    h = np.array([2.0**-k for k in range(3, 9)])
    slope, intercept = errors.fit_order(h, 0.7 * h**0.5)
    assert slope == pytest.approx(0.5, abs=1e-10)
    assert np.exp(intercept) == pytest.approx(0.7, rel=1e-10)




# Heston


def test_correlated_increments_have_the_requested_correlation():
    p = HestonParams(S0=100, V0=0.04, r=0.03, lam=1.5, mu=0.04, sigma=0.4, rho=-0.5)
    rng = np.random.default_rng(41)
    dB1, dB2 = correlated_increments(p, 50, 20_000, rng)
    assert np.corrcoef(dB1.ravel(), dB2.ravel())[0, 1] == pytest.approx(-0.5, abs=0.01)
    assert dB2.var() == pytest.approx(p.T / 50, rel=0.02)


def test_analytic_price_collapses_to_black_scholes_as_vol_of_vol_vanishes():
    p = HestonParams(S0=100, V0=0.04, r=0.03, lam=1.5, mu=0.04, sigma=1e-6, rho=0.0)
    vol = np.sqrt(p.V0)
    d1 = (np.log(p.S0 / 100) + (p.r + 0.5 * vol**2) * p.T) / (vol * np.sqrt(p.T))
    d2 = d1 - vol * np.sqrt(p.T)
    bs = p.S0 * norm.cdf(d1) - 100 * np.exp(-p.r * p.T) * norm.cdf(d2)
    assert analytic_call(p, 100.0) == pytest.approx(bs, abs=1e-4)


def test_monte_carlo_price_approaches_the_analytic_benchmark():
    p = HestonParams(S0=100, V0=0.04, r=0.03, lam=1.5, mu=0.04, sigma=0.4, rho=-0.5)
    rng = np.random.default_rng(42)
    dB1, dB2 = correlated_increments(p, 256, 40_000, rng)
    spot, _ = simulate_from_increments(
        p, SCHEMES["Explicit EM (truncated)"], dB1, dB2
    )
    price, se = european_call(spot, 100.0, p.r, p.T)
    benchmark = analytic_call(p, 100.0)
    # discretisation bias plus Monte Carlo noise; loose but meaningful
    assert abs(price - benchmark) < 4 * se + 0.05

