"""
Heston model. 
Will the choice of CIR scheme have a significant impact on an option price?

The Heston model is given by:
dS(t) = r*S(t)*dt + sqrt(V(t))*S(t)*dB1(t)
dV(t) = lam*(mu - V(t))*dt + sigma*sqrt(V(t))*dB2(t)

with Corr(dB1, dB2) = rho.

The variance process is a square-root diffusion, so every scheme in schemes.py applies to V directly.

The price of a European call is reported under each variance scheme, 
with Monte Carlo confidence intervals and against a semi-analytic benchmark. 

The goal is to investigate whether the differences across schemes exceed the Monte Carlo noise.

The spot process is simulated in log space,
log S_{n+1} = log S_n + (r - V_n/2)*h + sqrt(max(V_n, 0))*dB1,
which keeps S positive and isolates the discretisation of V as the only difference between runs. 
"""

from dataclasses import dataclass
import numpy as np
from .schemes import CIRParams


@dataclass(frozen=True)
class HestonParams:
    S0: float
    V0: float
    r: float
    lam: float
    mu: float
    sigma: float
    rho: float
    T: float = 1.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho be in [-1, 1].")
        if min(self.S0, self.V0, self.lam, self.mu, self.sigma, self.T) <= 0:
            raise ValueError("S0, V0, lam, mu, sigma and T must be strictly positive.")

    @property
    def variance_params(self) -> CIRParams:
        return CIRParams(
            kappa=self.lam, lam=self.mu, sigma=self.sigma, S0=self.V0, T=self.T
        )


def correlated_increments(
    params: HestonParams,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlated Brownian increments (dB1, dB2) via Cholesky factorisation."""
    h = params.T / n_steps
    z1 = rng.normal(0.0, np.sqrt(h), size=(n_paths, n_steps))
    z2 = rng.normal(0.0, np.sqrt(h), size=(n_paths, n_steps))
    dB1 = z1
    dB2 = params.rho * z1 + np.sqrt(1.0 - params.rho**2) * z2
    return dB1, dB2


def simulate_from_increments(
    params: HestonParams,
    variance_scheme,
    dB1: np.ndarray,
    dB2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate spot and variance from increments supplied by the caller."""
    if dB1.shape != dB2.shape:
        raise ValueError(f"increment shapes differ: {dB1.shape} vs {dB2.shape}.")
    n_paths, n_steps = dB1.shape
    V = variance_scheme(params.variance_params, dB2)

    h = params.T / n_steps
    v_pos = np.maximum(V[:, :-1], 0.0)
    log_increments = (params.r - 0.5 * v_pos) * h + np.sqrt(v_pos) * dB1
    log_S = np.log(params.S0) + np.cumsum(log_increments, axis=1)
    spot = np.empty((n_paths, n_steps + 1), dtype=float)
    spot[:, 0] = params.S0
    spot[:, 1:] = np.exp(log_S)
    return spot, V


def european_call(
    spot_paths: np.ndarray,
    strike: float,
    r: float,
    T: float,
) -> tuple[float, float]:
    """Discounted Monte Carlo price and its standard error.
    Failed paths, NaN, are dropped but report `errors.nan_fraction` alongside."""
    terminal = spot_paths[:, -1]
    finite = np.isfinite(terminal)
    if finite.sum() < 2:
        return float("nan"), float("nan")
    payoff = np.exp(-r * T) * np.maximum(terminal[finite] - strike, 0.0)
    n = payoff.size
    return float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n))



# Benchmark


def characteristic_function(
    u: complex | np.ndarray, params: HestonParams
) -> complex | np.ndarray:
    """Heston characteristic function of log S(T), Albrecher et al. form."""
    k, theta, sig, rho, T = params.lam, params.mu, params.sigma, params.rho, params.T
    iu = 1j * u
    m = k - rho * sig * iu
    d = np.sqrt(m**2 + sig**2 * (iu + u**2))
    g = (m - d) / (m + d)
    exp_dt = np.exp(-d * T)

    term1 = iu * (np.log(params.S0) + params.r * T)
    term2 = theta * k / sig**2 * ((m - d) * T - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g)))
    term3 = params.V0 / sig**2 * (m - d) * (1.0 - exp_dt) / (1.0 - g * exp_dt)
    return np.exp(term1 + term2 + term3)


def analytic_call(
    params: HestonParams,
    strike: float,
    upper: float = 250.0,
    n_nodes: int = 1024,
) -> float:
    """Semi-analytic European call price by Fourier inversion.
    C = S0*P1 - K*exp(-r*T)*P2, with P1 and P2 obtained from the characteristic function."""
    log_k = np.log(strike)
    phi_minus_i = params.S0 * np.exp(params.r * params.T)  # = phi(-i)

    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    lo = 1e-8
    u = 0.5 * (upper - lo) * nodes + 0.5 * (upper + lo)
    w = 0.5 * (upper - lo) * weights

    damped = np.exp(-1j * u * log_k)
    f1 = np.real(damped * characteristic_function(u - 1j, params) / (1j * u * phi_minus_i))
    f2 = np.real(damped * characteristic_function(u, params) / (1j * u))

    p1 = 0.5 + float(w @ f1) / np.pi
    p2 = 0.5 + float(w @ f2) / np.pi
    return float(params.S0 * p1 - strike * np.exp(-params.r * params.T) * p2)
