from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CIRParams:
    """Parameters for the CIR process."""

    kappa: float
    lam: float
    sigma: float
    S0: float
    T: float = 1.0

    def __post_init__(self) -> None:
        if self.kappa <= 0 or self.lam <= 0 or self.sigma <= 0:
            raise ValueError("kappa, lam and sigma must be strictly positive.")
        if self.S0 <= 0:
            raise ValueError("S0 must be strictly positive.")
        if self.T <= 0:
            raise ValueError("T must be strictly positive.")

    @property
    def feller_ratio(self) -> float:
        """2*kappa*lam / sigma**2. Feller condition holds iff this is >= 1.

        Values below 1 mean the process can reach zero.
        """
        return 2.0 * self.kappa * self.lam / self.sigma**2

    @property
    def alpha(self) -> float:
        """(4*kappa*lam - sigma**2)/8, from the Lamperti transform."""
        return (4.0 * self.kappa * self.lam - self.sigma**2) / 8.0

    @property
    def beta(self) -> float:
        """-kappa/2, from the Lamperti transform."""
        return -self.kappa / 2.0

    @property
    def gamma(self) -> float:
        """sigma/2, from the Lamperti transform."""
        return self.sigma / 2.0

    def exact_mean(self, t: float) -> float:
        """E[S(t)] = S0*exp(-kappa*t) + lam*(1 - exp(-kappa*t))."""
        decay = np.exp(-self.kappa * t)
        return self.S0 * decay + self.lam * (1.0 - decay)

    def exact_var(self, t: float) -> float:
        """Var[S(t)]"""
        k, lam, sig = self.kappa, self.lam, self.sigma
        decay = np.exp(-k * t)
        return (
            self.S0 * sig**2 / k * (decay - decay**2)
            + lam * sig**2 / (2 * k) * (1 - decay) ** 2
        )


def _allocate(params: CIRParams, dB: np.ndarray) -> tuple[np.ndarray, float]:
    """Allocate the path array and return it with the stepsize h."""
    if dB.ndim != 2:
        raise ValueError(f"dB must be 2-dimensional (M, N), got shape {dB.shape}.")
    n_paths, n_steps = dB.shape
    h = params.T / n_steps
    path = np.empty((n_paths, n_steps + 1), dtype=float)
    path[:, 0] = params.S0
    return path, h


def euler_absolute(params: CIRParams, dB: np.ndarray) -> np.ndarray:
    """Explicit Euler-Maruyama using sqrt(|S_n|) m."""

    S, h = _allocate(params, dB)
    k, lam, sig = params.kappa, params.lam, params.sigma
    for n in range(dB.shape[1]):
        s = S[:, n]
        S[:, n + 1] = s + h * k * (lam - s) + sig * np.sqrt(np.abs(s)) * dB[:, n]
    return S


def euler_truncated(params: CIRParams, dB: np.ndarray) -> np.ndarray:
    """Truncated explicit Euler-Maruyama: sqrt(max(S_n, 0))."""

    S, h = _allocate(params, dB)
    k, lam, sig = params.kappa, params.lam, params.sigma
    for n in range(dB.shape[1]):
        s = S[:, n]
        S[:, n + 1] = s + h * k * (lam - s) + sig * np.sqrt(np.maximum(s, 0.0)) * dB[:, n]
    return S


def euler_reflected(params: CIRParams, dB: np.ndarray) -> np.ndarray:
    """Reflected explicit Euler-Maruyama: |S_n| carried forward."""

    S, h = _allocate(params, dB)
    k, lam, sig = params.kappa, params.lam, params.sigma
    for n in range(dB.shape[1]):
        s = S[:, n]
        S[:, n + 1] = np.abs(s + h * k * (lam - s) + sig * np.sqrt(s) * dB[:, n])
    return S


def lamperti_implicit(params: CIRParams, dB: np.ndarray) -> np.ndarray:
    """Drift-implicit Euler-Maruyama on the Lamperti transform Y = sqrt(S)."""

    n_paths, n_steps = dB.shape
    h = params.T / n_steps
    a, b, g = params.alpha, params.beta, params.gamma
    denom = 1.0 - b * h  # = 1 + kappa*h/2 > 0

    S = np.empty((n_paths, n_steps + 1), dtype=float)
    S[:, 0] = params.S0
    Y = np.full(n_paths, np.sqrt(params.S0))
    for n in range(n_steps):
        u = Y + g * dB[:, n]
        radicand = u**2 / (4.0 * denom**2) + a * h / denom
        # sqrt(nan) is nan and raises no warning; sqrt(negative) warns.
        Y = u / (2.0 * denom) + np.sqrt(np.where(radicand >= 0.0, radicand, np.nan))
        S[:, n + 1] = Y**2
    return S


SCHEMES = {
    "Explicit EM (absolute)": euler_absolute,
    "Explicit EM (truncated)": euler_truncated,
    "Explicit EM (reflected)": euler_reflected,
    "Lamperti implicit": lamperti_implicit,
}
