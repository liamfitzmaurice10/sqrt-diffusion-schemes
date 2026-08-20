"""Brownian path construction and weak/strong error estimation."""

import numpy as np
from .schemes import CIRParams

# Brownian increments

def fine_increments(
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Brownian increments on the finest mesh. Shape (n_paths, n_steps)."""
    h = T / n_steps
    return rng.normal(0.0, np.sqrt(h), size=(n_paths, n_steps))


def coarsen(dB_fine: np.ndarray, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError("factor must be a positive integer.")
    n_paths, n_steps = dB_fine.shape
    if n_steps % factor != 0:
        raise ValueError(
            f"cannot coarsen {n_steps} increments by a factor of {factor}: "
            "the finest mesh must be an exact multiple of the coarse mesh."
        )
    if factor == 1:
        return dB_fine
    return dB_fine.reshape(n_paths, n_steps // factor, factor).sum(axis=2)

# Errors

def _mean_and_stderr(x: np.ndarray) -> tuple[float, float]:
    """Sample mean and its standard error.
    Note: failed paths are ignored."""
    finite = np.isfinite(x)
    n = int(finite.sum())
    if n == 0:
        return float("nan"), float("nan")
    vals = x[finite]
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def weak_error(terminal_values: np.ndarray, params: CIRParams) -> tuple[float, float]:
    """|E[S(T)] - mean(S_N)| and the Monte Carlo standard error of the estimate.
    Returns (error, standard_error)."""
    reference = params.exact_mean(params.T)
    mean, stderr = _mean_and_stderr(terminal_values)
    return abs(mean - reference), stderr


def weak_error_functional(
    values: np.ndarray,
    reference: float,
) -> tuple[float, float]:
    """|reference - mean(values)| and its standard error, for any test function."""
    mean, stderr = _mean_and_stderr(values)
    return abs(mean - reference), stderr


def strong_error(
    terminal_values: np.ndarray,
    reference_values: np.ndarray,
) -> tuple[float, float]:
    """mean(|S_ref - S_N|) and its standard error.

    Both arrays must come from the same Brownian paths in the same order.
    Note: paths that failed in either array are dropped."""
    if terminal_values.shape != reference_values.shape:
        raise ValueError(
            "terminal and reference arrays must be aligned pathwise, got "
            f"{terminal_values.shape} and {reference_values.shape}."
        )
    return _mean_and_stderr(np.abs(reference_values - terminal_values))


def fit_order(h_values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept of log(error) against log(h)."""
    h_values = np.asarray(h_values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    usable = np.isfinite(errors) & (errors > 0) & np.isfinite(h_values) & (h_values > 0)
    if usable.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.log(h_values[usable]), np.log(errors[usable]), 1)
    return float(slope), float(intercept)


def nan_fraction(terminal_values: np.ndarray) -> float:
    """Fraction of paths that failed (produced NaN or infinity)."""
    terminal_values = np.asarray(terminal_values, dtype=float)
    if terminal_values.size == 0:
        return float("nan")
    return float(1.0 - np.isfinite(terminal_values).mean())


def negative_fraction(terminal_values: np.ndarray) -> float:
    """Fraction of surviving paths with a strictly negative value."""
    terminal_values = np.asarray(terminal_values, dtype=float)
    finite = np.isfinite(terminal_values)
    if finite.sum() == 0:
        return float("nan")
    return float((terminal_values[finite] < 0).mean())
