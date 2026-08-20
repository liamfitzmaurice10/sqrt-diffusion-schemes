"""
Reproduces every figure and table in the README.

Run from the repository root:

    python -m experiments.run_all # full run, a few minutes
    python -m experiments.run_all --quick # reduced sample sizes

Path counts are large enough that the arrays do not fit in memory at the finest mesh, 
so the Monte Carlo loops accumulate running sums over batches of paths rather than materialising all of them. 
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sqrtdiff import errors, exact
from sqrtdiff.heston import (
    HestonParams,
    analytic_call,
    correlated_increments,
    simulate_from_increments,
)
from sqrtdiff.schemes import SCHEMES, CIRParams


# Configuration
SEED = 20260804
ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
RESDIR = ROOT / "results"

BASE = CIRParams(kappa=1.0, lam=0.2, sigma=0.1, S0=0.2)
STRIKE = 0.2

LEVELS = [2**k for k in range(4, 10)]  # 2^4 ... 2^9
WEAK_SIGMAS = [0.1, 1.0]  # Feller comfortably satisfied, and badly violated
N_REF = 2**13  # 16x the finest tested level
FIT_DROP = 2  # finest two levels excluded from the strong-error fit

SWEEP_SIGMAS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2]
SWEEP_LEVELS = [2**k for k in range(4, 9)]
SWEEP_N_REF = 2**12

HESTON = HestonParams(S0=100.0, V0=0.04, r=0.03, lam=1.5, mu=0.04, sigma=0.4, rho=-0.5)
HESTON_SIGMAS = [0.4, 0.6]
HESTON_STRIKE = 100.0
HESTON_MESHES = [2**5, 2**8]
BASELINE_SCHEME = "Explicit EM (truncated)"

M_WEAK, B_WEAK = 200_000, 20_000
M_STRONG, B_STRONG = 20_000, 2_500
M_SWEEP, B_SWEEP = 10_000, 2_500
M_HESTON, B_HESTON = 100_000, 20_000

QUICK = {
    "M_WEAK": 20_000, "B_WEAK": 10_000,
    "M_STRONG": 2_000, "B_STRONG": 1_000,
    "M_SWEEP": 2_000, "B_SWEEP": 1_000,
    "M_HESTON": 10_000, "B_HESTON": 10_000,
    "N_REF": 2**11, "LEVELS": [2**k for k in range(4, 8)],
    "SWEEP_N_REF": 2**10, "SWEEP_LEVELS": [2**k for k in range(4, 7)],
}

COLOURS = dict(zip(SCHEMES, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]))
STYLES = dict(zip(SCHEMES, ["-", "--", "-.", ":"]))


class Accumulator:
    """Running mean, standard error and failure counts over batches of paths. Kept deliberately small."""

    def __init__(self) -> None:
        self.n_total = 0
        self.n_finite = 0
        self.n_negative = 0
        self._sum = 0.0
        self._sum_sq = 0.0

    def add(self, values: np.ndarray) -> "Accumulator":
        values = np.asarray(values, dtype=float).ravel()
        finite = np.isfinite(values)
        good = values[finite]
        self.n_total += values.size
        self.n_finite += int(finite.sum())
        self.n_negative += int((good < 0).sum())
        self._sum += float(good.sum())
        self._sum_sq += float((good**2).sum())
        return self

    @property
    def mean(self) -> float:
        return self._sum / self.n_finite if self.n_finite else float("nan")

    @property
    def stderr(self) -> float:
        if self.n_finite < 2:
            return float("nan")
        var = (self._sum_sq - self._sum**2 / self.n_finite) / (self.n_finite - 1)
        return float(np.sqrt(max(var, 0.0) / self.n_finite))

    @property
    def nan_fraction(self) -> float:
        return 1.0 - self.n_finite / self.n_total if self.n_total else float("nan")

    @property
    def negative_fraction(self) -> float:
        return self.n_negative / self.n_finite if self.n_finite else float("nan")


def batch_sizes(total: int, size: int) -> list[int]:
    """Split the total paths into batches of at most size."""
    full, remainder = divmod(total, size)
    return [size] * full + ([remainder] if remainder else [])


def write_table(name: str, header: list[str], rows: list[list[str]]) -> None:
    RESDIR.mkdir(exist_ok=True)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    (RESDIR / f"{name}.md").write_text("\n".join(lines) + "\n")
    print(f"\n{name}\n" + "\n".join(lines))


def save(fig: plt.Figure, name: str) -> None:
    FIGDIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGDIR / name, dpi=150)
    plt.close(fig)
    print(f"wrote figures/{name}")


# Sample paths


def experiment_1_sample_paths(rng: np.random.Generator) -> None:
    """One trajectory under all four schemes at N = 2**3, 2**7, 2**10. 
    Output: figures/01_sample_paths.png"""
    finest = 2**10
    resolutions = [2**3, 2**7, 2**10]
    sigmas = [BASE.sigma, 1.2]
    dB_fine = errors.fine_increments(BASE.T, finest, 1, rng)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for row, sigma in enumerate(sigmas):
        params = CIRParams(BASE.kappa, BASE.lam, sigma, BASE.S0, BASE.T)
        for col, n in enumerate(resolutions):
            ax = axes[row, col]
            dB = errors.coarsen(dB_fine, finest // n)
            t = np.linspace(0, params.T, n + 1)
            ax.plot(t, [params.exact_mean(s) for s in t], color="grey", lw=1,
                    ls=":", label="Exact mean")
            for name, scheme in SCHEMES.items():
                ax.plot(t, scheme(params, dB)[0], color=COLOURS[name],
                        ls=STYLES[name], lw=1.4, label=name)
            ax.axhline(0.0, color="black", lw=0.6)
            ax.set_title(rf"$\sigma={sigma}$, $N=2^{{{int(np.log2(n))}}}$", fontsize=10)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel("S(t)")
            if row == 1:
                ax.set_xlabel("t")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Single trajectory, same Brownian path at every resolution")
    save(fig, "01_sample_paths.png")



# Weak error


def experiment_2_weak_error(rng: np.random.Generator, m: int, batch: int,
                            levels: list[int]) -> None:
    """Weak error against h for each scheme, with MC standard-error bars.
    Output: figures/02_weak_error.png, results/02_weak_order.md"""
    finest = max(levels)
    h = np.array([BASE.T / n for n in levels])
    rows = []
    fig, axes = plt.subplots(len(WEAK_SIGMAS), 2, figsize=(12, 4.6 * len(WEAK_SIGMAS)))

    for row, sigma in enumerate(WEAK_SIGMAS):
        params = CIRParams(BASE.kappa, BASE.lam, sigma, BASE.S0, BASE.T)
        ref_mean = params.exact_mean(params.T)
        ref_call = exact.call_expectation(params, STRIKE)
        acc_id = {(s, n): Accumulator() for s in SCHEMES for n in levels}
        acc_call = {(s, n): Accumulator() for s in SCHEMES for n in levels}

        for size in batch_sizes(m, batch):
            dB_fine = errors.fine_increments(params.T, finest, size, rng)
            for n in levels:
                dB = errors.coarsen(dB_fine, finest // n)
                for name, scheme in SCHEMES.items():
                    terminal = scheme(params, dB)[:, -1]
                    acc_id[(name, n)].add(terminal)
                    acc_call[(name, n)].add(np.maximum(terminal - STRIKE, 0.0))

        for col, (acc, ref, label) in enumerate((
            (acc_id, ref_mean, "q(x)=x"),
            (acc_call, ref_call, f"q(x)=(x-{STRIKE})^+"),
        )):
            ax = axes[row, col]
            for name in SCHEMES:
                err = np.array([abs(acc[(name, n)].mean - ref) for n in levels])
                se = np.array([acc[(name, n)].stderr for n in levels])
                ax.errorbar(h, err, yerr=se, color=COLOURS[name], ls=STYLES[name],
                            marker="o", ms=4, lw=1.3, capsize=3, label=name)
                slope, _ = errors.fit_order(h, err)
                rows.append([f"{sigma:.1f}", label, name, f"{slope:.2f}",
                             f"{err[0]:.2e}", f"{se[0]:.2e}",
                             f"{err[0] / se[0]:.1f}",
                             f"{acc[(name, levels[0])].nan_fraction:.3f}"])
            # reference slope anchored at the finest level of the last scheme
            ax.plot(h, err[-1] * (h / h[-1]), color="black", lw=0.8, ls="--",
                    label="slope 1")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("h"); ax.set_ylabel("weak error")
            ax.set_title(rf"$\sigma={sigma}$,  ${label}$", fontsize=10)
            ax.grid(alpha=0.3, which="both")
            worst = max(acc[(n, levels[0])].nan_fraction for n in SCHEMES)
            if worst > 0:
                ax.text(0.03, 0.06,
                        f"up to {100 * worst:.0f}% of paths failed:\n"
                        "those curves are conditional on survival",
                        transform=ax.transAxes, fontsize=7, color="firebrick")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"Weak error, M = {m:,} paths, bars are one MC standard error")
    save(fig, "02_weak_error.png")
    write_table("02_weak_order",
                ["sigma", "Test function", "Scheme", "Fitted slope",
                 "Error at h=1/16", "MC std. err.", "Error / s.e.", "Failed paths"],
                rows)



# Strong error


def experiment_3_strong_error(rng: np.random.Generator, m: int, batch: int,
                              levels: list[int], n_ref: int) -> None:
    """Strong error against h, referenced to a mesh several levels finer.

    Each scheme is referenced to itself at N_ref on the same Brownian path.
    This measures self-convergence, not distance from the true solution. 
    
    As a partial check that the schemes share a limit, 
    the mean pairwise distance between schemes at N_ref is also reported.

    Output: figures/03_strong_error.png, results/03_strong_order.md, results/03_scheme_agreement.md """
    acc = {(s, n): Accumulator() for s in SCHEMES for n in levels}
    fails = {(s, n): Accumulator() for s in SCHEMES for n in levels}
    names = list(SCHEMES)
    agree = {(a, b): Accumulator() for i, a in enumerate(names) for b in names[i + 1:]}

    for size in batch_sizes(m, batch):
        dB_fine = errors.fine_increments(BASE.T, n_ref, size, rng)
        reference = {}
        for name, scheme in SCHEMES.items():
            reference[name] = scheme(BASE, dB_fine)[:, -1]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                agree[(a, b)].add(np.abs(reference[a] - reference[b]))
        for n in levels:
            dB = errors.coarsen(dB_fine, n_ref // n)
            for name, scheme in SCHEMES.items():
                terminal = scheme(BASE, dB)[:, -1]
                acc[(name, n)].add(np.abs(reference[name] - terminal))
                fails[(name, n)].add(terminal)

    h = np.array([BASE.T / n for n in levels])
    keep = slice(0, len(levels) - FIT_DROP)
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name in SCHEMES:
        err = np.array([acc[(name, n)].mean for n in levels])
        se = np.array([acc[(name, n)].stderr for n in levels])
        ax.errorbar(h, err, yerr=se, color=COLOURS[name], ls=STYLES[name],
                    marker="o", ms=4, lw=1.3, capsize=3, label=name)
        slope, _ = errors.fit_order(h[keep], err[keep])
        rows.append([name, f"{slope:.3f}", f"{err[0]:.2e}", f"{err[-1]:.2e}",
                     f"{fails[(name, levels[0])].negative_fraction:.3f}"])
    # reference slopes, anchored at the coarsest level of the last scheme drawn
    ax.plot(h, err[0] * (h / h[0]) ** 0.5, color="black", lw=0.8, ls="--",
            label="slope 1/2")
    ax.plot(h, err[0] * (h / h[0]) ** 1.0, color="black", lw=0.8, ls=":",
            label="slope 1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("h"); ax.set_ylabel(r"$E|S_{ref} - S_N|$")
    ax.set_title(f"Strong error vs self-reference at $N={n_ref}$, M = {m:,}")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, frameon=False)
    save(fig, "03_strong_error.png")

    write_table("03_strong_order",
                ["Scheme", f"Fitted order (coarsest {len(levels) - FIT_DROP} levels)",
                 "Error at h=1/16", f"Error at h=1/{levels[-1]}",
                 "P(S_N < 0) at h=1/16"], rows)
    write_table("03_scheme_agreement",
                ["Pair", f"Mean absolute difference at N={n_ref}",
                 "As % of E[S(T)]"],
                [[f"{a} vs {b}", f"{agree[(a, b)].mean:.2e}",
                  f"{100 * agree[(a, b)].mean / BASE.exact_mean(BASE.T):.3f}%"]
                 for (a, b) in agree])



#Feller sweep


def experiment_4_feller_sweep(rng: np.random.Generator, m: int, batch: int,
                              levels: list[int], n_ref: int) -> None:
    """Sweep sigma across the Feller boundary and the alpha < 0 boundary.

    With kappa = 1 and lam = 0.2 the two boundaries sit at sigma = sqrt(0.4),
    = 0.632 (Feller) and = sqrt(0.8) = 0.894 (alpha = 0), so the grid spans all three regimes.

    Output: figures/04_feller_sweep.png, figures/05_failure_rate.png, 
    results/04_feller_orders.md, results/05_failure_rate.md"""
    err = {}
    nanf = {}
    negf = {}
    for sigma in SWEEP_SIGMAS:
        params = CIRParams(BASE.kappa, BASE.lam, sigma, BASE.S0, BASE.T)
        acc = {(s, n): Accumulator() for s in SCHEMES for n in levels}
        term = {(s, n): Accumulator() for s in SCHEMES for n in levels}
        for size in batch_sizes(m, batch):
            dB_fine = errors.fine_increments(params.T, n_ref, size, rng)
            for name, scheme in SCHEMES.items():
                reference = scheme(params, dB_fine)[:, -1]
                for n in levels:
                    dB = errors.coarsen(dB_fine, n_ref // n)
                    terminal = scheme(params, dB)[:, -1]
                    acc[(name, n)].add(np.abs(reference - terminal))
                    term[(name, n)].add(terminal)
        for name in SCHEMES:
            err[(sigma, name)] = np.array([acc[(name, n)].mean for n in levels])
            nanf[(sigma, name)] = np.array([term[(name, n)].nan_fraction for n in levels])
            negf[(sigma, name)] = np.array([term[(name, n)].negative_fraction
                                            for n in levels])
        print(f"  sigma={sigma}: Feller ratio {params.feller_ratio:.2f}, "
              f"alpha {params.alpha:+.4f}")

    h = np.array([BASE.T / n for n in levels])
    keep = slice(0, max(2, len(levels) - 1))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True)
    rows = []
    for ax, sigma in zip(axes.ravel(), SWEEP_SIGMAS):
        params = CIRParams(BASE.kappa, BASE.lam, sigma, BASE.S0, BASE.T)
        for name in SCHEMES:
            e = err[(sigma, name)]
            ax.plot(h, e, color=COLOURS[name], ls=STYLES[name], marker="o",
                    ms=3.5, lw=1.2, label=name)
            slope, _ = errors.fit_order(h[keep], e[keep])
            rows.append([f"{sigma:.1f}", f"{params.feller_ratio:.2f}",
                         f"{params.alpha:+.4f}", name, f"{slope:.3f}",
                         f"{nanf[(sigma, name)][0]:.3f}"])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        regime = ("Feller holds" if params.feller_ratio >= 1
                  else ("Feller fails" if params.alpha > 0 else r"$\alpha<0$"))
        ax.set_title(rf"$\sigma={sigma}$  ({regime})", fontsize=10)
        failed = nanf[(sigma, "Lamperti implicit")].max()
        if failed > 0:
            ax.text(0.03, 0.9,
                    f"Lamperti: {100 * failed:.0f}% of paths failed;\n"
                    "its curve is conditional on survival",
                    transform=ax.transAxes, fontsize=7, color="firebrick")
    for ax in axes[1]:
        ax.set_xlabel("h")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$E|S_{ref}-S_N|$")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Strong error across the Feller and alpha boundaries, M = {m:,}")
    save(fig, "04_feller_sweep.png")
    write_table("04_feller_orders",
                ["sigma", "Feller ratio", "alpha", "Scheme", "Fitted strong order",
                 "Failure rate at h=1/16"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for sigma in SWEEP_SIGMAS:
        axes[0].plot(h, nanf[(sigma, "Lamperti implicit")], marker="o", ms=4,
                     lw=1.3, label=rf"$\sigma={sigma}$")
        axes[1].plot(h, negf[(sigma, "Explicit EM (absolute)")], marker="o", ms=4,
                     lw=1.3, label=rf"$\sigma={sigma}$")
    axes[0].set_title("Lamperti implicit: fraction of failed paths")
    axes[1].set_title("Explicit EM (absolute): fraction of negative $S_N$")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("h"); ax.set_ylabel("fraction of paths")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
    save(fig, "05_failure_rate.png")
    write_table("05_failure_rate",
                ["sigma"] + [f"h=1/{n}" for n in levels],
                [[f"{s:.1f}"] + [f"{v:.4f}" for v in nanf[(s, 'Lamperti implicit')]]
                 for s in SWEEP_SIGMAS])
    write_table("05_negative_fraction",
                ["sigma"] + [f"h=1/{n}" for n in levels],
                [[f"{s:.1f}"]
                 + [f"{v:.4f}" for v in negf[(s, 'Explicit EM (absolute)')]]
                 for s in SWEEP_SIGMAS])



# Heston pricing


def experiment_5_heston_pricing(rng: np.random.Generator, m: int, batch: int) -> None:
    """European call under Heston, one price per variance scheme.
    Output: figures/06_heston_prices.png, results/06_heston_prices.md"""
    fine_mesh = max(HESTON_MESHES)
    records = []
    for sigma in HESTON_SIGMAS:
        params = HestonParams(HESTON.S0, HESTON.V0, HESTON.r, HESTON.lam,
                              HESTON.mu, sigma, HESTON.rho, HESTON.T)
        benchmark = analytic_call(params, HESTON_STRIKE)
        v = params.variance_params

        price = {(s, n): Accumulator() for s in SCHEMES for n in HESTON_MESHES}
        vs_baseline = {(s, n): Accumulator() for s in SCHEMES for n in HESTON_MESHES}
        vs_fine = {s: Accumulator() for s in SCHEMES}

        for size in batch_sizes(m, batch):
            dB1_fine, dB2_fine = correlated_increments(params, fine_mesh, size, rng)
            payoff = {}
            for n_steps in HESTON_MESHES:
                factor = fine_mesh // n_steps
                dB1 = errors.coarsen(dB1_fine, factor)
                dB2 = errors.coarsen(dB2_fine, factor)
                for name, scheme in SCHEMES.items():
                    spot, _ = simulate_from_increments(params, scheme, dB1, dB2)
                    payoff[(name, n_steps)] = np.exp(-params.r * params.T) * np.maximum(
                        spot[:, -1] - HESTON_STRIKE, 0.0)
                    price[(name, n_steps)].add(payoff[(name, n_steps)])
            for name in SCHEMES:
                for n_steps in HESTON_MESHES:
                    vs_baseline[(name, n_steps)].add(
                        payoff[(name, n_steps)] - payoff[(BASELINE_SCHEME, n_steps)])
                vs_fine[name].add(
                    payoff[(name, min(HESTON_MESHES))] - payoff[(name, fine_mesh)])

        for n_steps in HESTON_MESHES:
            for name in SCHEMES:
                records.append(dict(
                    sigma=sigma, feller=v.feller_ratio, alpha=v.alpha,
                    n_steps=n_steps, scheme=name, benchmark=benchmark,
                    price=price[(name, n_steps)].mean,
                    se=price[(name, n_steps)].stderr,
                    diff=vs_baseline[(name, n_steps)].mean,
                    diff_se=vs_baseline[(name, n_steps)].stderr,
                    mesh_diff=vs_fine[name].mean if n_steps == min(HESTON_MESHES)
                    else float("nan"),
                    mesh_diff_se=vs_fine[name].stderr if n_steps == min(HESTON_MESHES)
                    else float("nan"),
                    failed=price[(name, n_steps)].nan_fraction))
        print(f"  sigma={sigma}: benchmark {benchmark:.4f}, "
              f"variance Feller ratio {v.feller_ratio:.2f}, alpha {v.alpha:+.4f}")

    RESDIR.mkdir(exist_ok=True)
    (RESDIR / "06_heston_prices.json").write_text(json.dumps(records, indent=2))
    write_table("06_heston_prices",
                ["sigma", "N", "Scheme", "MC price", "95% CI half-width",
                 "Bias vs benchmark", "Paired diff. vs truncated",
                 "Paired coarse - fine", "Failed paths"],
                [[f"{r['sigma']:.1f}", str(r["n_steps"]), r["scheme"],
                  f"{r['price']:.4f}", f"{1.96 * r['se']:.4f}",
                  f"{r['price'] - r['benchmark']:+.4f}",
                  f"{r['diff']:+.4f} +/- {1.96 * r['diff_se']:.4f}",
                  "-" if not np.isfinite(r["mesh_diff"]) else
                  f"{r['mesh_diff']:+.4f} +/- {1.96 * r['mesh_diff_se']:.4f}",
                  f"{r['failed']:.4f}"] for r in records])

    fig, axes = plt.subplots(1, len(HESTON_SIGMAS), figsize=(12, 4.8))
    for ax, sigma in zip(np.atleast_1d(axes), HESTON_SIGMAS):
        subset = [r for r in records if r["sigma"] == sigma]
        benchmark = subset[0]["benchmark"]
        width = 0.35
        for j, n_steps in enumerate(HESTON_MESHES):
            rows = [r for r in subset if r["n_steps"] == n_steps]
            x = np.arange(len(rows)) + (j - 0.5) * width
            ax.errorbar(x, [r["price"] for r in rows],
                        yerr=[1.96 * r["se"] for r in rows], fmt="o", ms=5,
                        capsize=4, label=f"N = {n_steps}")
        ax.axhline(benchmark, color="black", lw=1, ls="--", label="semi-analytic")
        ax.set_xticks(np.arange(len(SCHEMES)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in SCHEMES], fontsize=7)
        ax.set_ylabel("call price")
        ax.set_title(rf"$\sigma_V={sigma}$ (variance Feller ratio "
                     rf"{subset[0]['feller']:.2f}, $\alpha={subset[0]['alpha']:+.3f}$)",
                     fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
    fig.suptitle("European call under Heston, bars are 95% Monte Carlo intervals")
    save(fig, "06_heston_prices.png")




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="reduced sample sizes for a fast smoke run")
    args = parser.parse_args()

    cfg = dict(M_WEAK=M_WEAK, B_WEAK=B_WEAK, M_STRONG=M_STRONG, B_STRONG=B_STRONG,
               M_SWEEP=M_SWEEP, B_SWEEP=B_SWEEP, M_HESTON=M_HESTON,
               B_HESTON=B_HESTON, N_REF=N_REF, LEVELS=LEVELS,
               SWEEP_N_REF=SWEEP_N_REF, SWEEP_LEVELS=SWEEP_LEVELS)
    if args.quick:
        cfg.update(QUICK)

    FIGDIR.mkdir(exist_ok=True)
    RESDIR.mkdir(exist_ok=True)
    streams = np.random.default_rng(SEED).spawn(5)

    print("Experiment 1: sample paths")
    experiment_1_sample_paths(streams[0])
    print("Experiment 2: weak error")
    experiment_2_weak_error(streams[1], cfg["M_WEAK"], cfg["B_WEAK"], cfg["LEVELS"])
    print("Experiment 3: strong error")
    experiment_3_strong_error(streams[2], cfg["M_STRONG"], cfg["B_STRONG"],
                              cfg["LEVELS"], cfg["N_REF"])
    print("Experiment 4: Feller sweep")
    experiment_4_feller_sweep(streams[3], cfg["M_SWEEP"], cfg["B_SWEEP"],
                              cfg["SWEEP_LEVELS"], cfg["SWEEP_N_REF"])
    print("Experiment 5: Heston pricing")
    experiment_5_heston_pricing(streams[4], cfg["M_HESTON"], cfg["B_HESTON"])
    print("\nDone. Figures in figures/, tables in results/.")


if __name__ == "__main__":
    main()
