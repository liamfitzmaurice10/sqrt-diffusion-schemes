"""
Exact sampling of the CIR transition law, and closed forms used as references.

For t > s, the transition density of CIR is a scaled noncentral chi-squared:

S(t) = Y / (2c),
c = 2*kappa / (sigma**2 * (1 - exp(-kappa*(t-s)))),
Y ~ noncentral chi-squared with degrees of freedom,
d = 4*kappa*lam / sigma**2,
noncentrality parameter,
nc = 2*c*S(s)*exp(-kappa*(t-s))

This is the reference for the weak error.

Note: exact sampling gives the correct marginal law at time T but is not driven by the same Brownian path as the schemes.
So this is a valid reference for weak error only. 
Strong error still needs afine-grid reference.

Two closed forms are used so that the weak error can be measured without Monte Carlo noise on the reference side:

`CIRParams.exact_mean`, E[S(T)];
`call_expectation`, E[(S(T) - K)^+].

The second follows from the identity, for Y ~ ncx2(d, nc),

E[(Y - k)^+] = d*SF_{d+2,nc}(k) + nc*SF_{d+4,nc}(k) - k*SF_{d,nc}(k),

where SF is the survival function. 
This follows from writing the noncentral chi-squared as a Poisson mixture of central chi-squareds 
and using the identity y*f_m(y) = m*f_{m+2}(y).
"""

import numpy as np
from scipy.stats import ncx2

from .schemes import CIRParams


def _transition_constants(params: CIRParams, dt: float) -> tuple[float, float]:
    """Return (d, c): degrees of freedom and the scale constant."""
    if dt <= 0:
        raise ValueError("dt must be strictly positive.")
    d = 4.0 * params.kappa * params.lam / params.sigma**2
    c = 2.0 * params.kappa / (params.sigma**2 * (1.0 - np.exp(-params.kappa * dt)))
    return d, c


def sample_terminal(
    params: CIRParams,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_paths exact samples of S(T) in a single step from S(0).
    Returns an array of shape (n_paths,)."""
    d, c = _transition_constants(params, params.T)
    nc = 2.0 * c * params.S0 * np.exp(-params.kappa * params.T)
    return rng.noncentral_chisquare(d, nc, size=n_paths) / (2.0 * c)


def call_expectation(params: CIRParams, strike: float, t: float | None = None) -> float:
    """E[(S(t) - K)^+] in closed form under the exact transition density."""
    t = params.T if t is None else t
    d, c = _transition_constants(params, t)
    nc = 2.0 * c * params.S0 * np.exp(-params.kappa * t)
    k = 2.0 * c * strike  # threshold in Y-space
    truncated = (
        d * ncx2.sf(k, d + 2.0, nc)
        + nc * ncx2.sf(k, d + 4.0, nc)
        - k * ncx2.sf(k, d, nc)
    )
    return float(truncated / (2.0 * c))
