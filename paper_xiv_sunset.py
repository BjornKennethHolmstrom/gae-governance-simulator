"""
paper_xiv_sunset.py — Simulation for Paper XIV, Section 6.9 / Appendix A.4
"The Sunset Decision as Dual Control"

Model
-----
A regulator suppresses a disturbance class with latent rate lambda(t)
(incidents per year absent regulation). The regulator's true effectiveness
e is HIGH BUT NOT PRECISELY KNOWN to the observer: the operational record
y_reg ~ Poisson((1-p) * lambda * (1-e)) therefore cannot separate a low
threat from an effective regulator — the joint (lambda, e)
unidentifiability that makes a regulator's own record incapable of
certifying its necessity. A probe channel — a protected experimental space
covering fraction p of the exposure base, run at a *designed and therefore
known* reduced effectiveness e_probe — is the only source of direct
information about lambda:

    y_probe ~ Poisson( p * lambda * (1 - e_probe) ).

The latent rate is mean-reverting (OU in log space) around a level that
depends on scenario, with innovation std sigma_d per year: the environment
drifts, so information decays.

The observer runs a Bayesian grid filter over (log lambda, e), with a
random-walk diffusion kernel on log lambda (it does not know the
attractor). The prior on lambda is centred at lambda_high — the regulator
was created because the threat was real — so evidence must actively move
mass down. At an annual sunset review the regulator is removed iff

    P( lambda <= lambda_safe | data ) >= 1 - alpha,

i.e. removal must be risk-bounded, not necessity merely undemonstrated.

Scenario A (decidability): the problem is genuinely solved — lambda
reverts around lambda_low << lambda_safe from t=0 (the pre-history in
which the observer learned lambda ~ lambda_high is encoded in the prior).
Measured: years until justified removal, vs p. Prediction: divergence
below a critical probe rate p*; for p < p*, the sunset clause is
structurally dead letter.

Scenario B (probe cost / error): lambda reverts around 2*lambda_safe —
the regulator is still necessary. Measured: false-removal rate and
cumulative probe-channel incidents (the probe is paid for in real harm).

Analytic reference (Appendix A.4)
---------------------------------
In log space x = log lambda with per-year diffusion q = sigma_d^2, the
probe channel's yearly Fisher information about x is I = c * lambda with
c = p * (1 - e_probe). The steady-state posterior variance of the scalar
drift-plus-measurement filter is sigma_x^2 ~ sqrt(q / I). Removal is
decidable at true rate lambda_low iff z_alpha * sigma_x <= Delta with
Delta = log(lambda_safe / lambda_low), giving

    p* = q * z_alpha^4 / ( lambda_low * (1 - e_probe) * Delta^4 ).

Note the fourth-power dependence on the confidence requirement z_alpha and
(inversely) on the log safety margin Delta.

Outputs: outputs/paper_xiv_sunset_time_to_removal.png, outputs/paper_xiv_sunset_tradeoff.png
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20260702)

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
E_PROBE = 0.50          # designed (known) effectiveness inside the probe
LAMBDA_SAFE = 1.0       # acceptable incident rate without the regulator
ALPHA = 0.05
Z = norm.ppf(1 - ALPHA)

LAMBDA_HIGH = 10.0      # historical threat level (encoded in the prior)
LAMBDA_LOW = 0.2        # true level after the problem is solved (Scenario A)
LAMBDA_B = 2.0          # true level when still necessary (Scenario B)
E_TRUE = 0.97           # true regulator effectiveness (unknown to observer)
OU_PHI = 0.95           # mean reversion of log lambda
HORIZON = 300           # years (censoring point)

SIGMA_DRIFTS = [0.03, 0.06, 0.12]     # innovation std of log lambda per year
P_GRID = np.array([0.0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04,
                   0.06, 0.08, 0.15, 0.25, 0.40])
N_RUNS = 150

# Observer grid
X_GRID = np.linspace(np.log(1e-2), np.log(1e2), 200)   # log lambda
DX = X_GRID[1] - X_GRID[0]
LAM_GRID = np.exp(X_GRID)
E_GRID = np.linspace(0.90, 0.9995, 20)                  # regulator effectiveness
SAFE = LAM_GRID <= LAMBDA_SAFE


def analytic_pstar(sigma_d, lam_true=LAMBDA_LOW):
    q = sigma_d ** 2
    delta = np.log(LAMBDA_SAFE / lam_true)
    return q * Z ** 4 / (lam_true * (1 - E_PROBE) * delta ** 4)


def latent_paths(sigma_d, n_runs, lam_star):
    """OU in log space around log(lam_star), started at the attractor."""
    x_star = np.log(lam_star)
    x = np.full(n_runs, x_star)
    out = np.empty((n_runs, HORIZON + 1))
    out[:, 0] = x
    for t in range(1, HORIZON + 1):
        x = x_star + OU_PHI * (x - x_star) + rng.normal(0, sigma_d, n_runs)
        out[:, t] = x
    return np.exp(out)


def run_cell(p, sigma_d, lam_star):
    """N_RUNS trajectories at probe rate p. Observer: grid filter over
    (log lambda, e), RW diffusion on log lambda, annual sunset review.
    Returns (removal year or -1, cumulative probe incidents)."""
    lam = latent_paths(sigma_d, N_RUNS, lam_star)
    c_probe = p * (1 - E_PROBE)

    # Prior: lognormal on lambda centred at the historical threat level,
    # uniform on e. Shape: (runs, e, x) in log domain.
    lp_x = -0.5 * ((X_GRID - np.log(LAMBDA_HIGH)) / 1.0) ** 2
    log_post = np.broadcast_to(
        lp_x[None, None, :], (N_RUNS, E_GRID.size, X_GRID.size)
    ).copy()

    sig_units = sigma_d / DX
    removed_at = np.full(N_RUNS, -1, dtype=int)
    probe_inc = np.zeros(N_RUNS)
    active = np.ones(N_RUNS, dtype=bool)

    # Precompute channel coefficients on the grid
    c_reg_grid = (1 - p) * (1 - E_GRID)[:, None] * LAM_GRID[None, :]  # (e, x)
    mu_probe = c_probe * LAM_GRID                                      # (x,)
    log_lam_reg = np.log(np.maximum(c_reg_grid, 1e-300))
    log_lam_probe = np.log(np.maximum(mu_probe, 1e-300))

    for t in range(1, HORIZON + 1):
        if not active.any():
            break
        lt = lam[active, t]
        y_reg = rng.poisson((1 - p) * (1 - E_TRUE) * lt)
        y_probe = rng.poisson(c_probe * lt)
        probe_inc[active] += y_probe

        # predict: diffuse along log lambda
        post = np.exp(log_post[active])
        post = gaussian_filter1d(post, sig_units, axis=2, mode="nearest")

        # update
        lp = np.log(np.maximum(post, 1e-300))
        lp += (y_reg[:, None, None] * log_lam_reg[None, :, :]
               - c_reg_grid[None, :, :])
        if c_probe > 0:
            lp += (y_probe[:, None, None] * log_lam_probe[None, None, :]
                   - mu_probe[None, None, :])
        lp -= lp.max(axis=(1, 2), keepdims=True)
        log_post[active] = lp

        # sunset review on the lambda-marginal
        post = np.exp(lp)
        post /= post.sum(axis=(1, 2), keepdims=True)
        p_safe = post[:, :, SAFE].sum(axis=(1, 2))
        decide = p_safe >= 1 - ALPHA
        if decide.any():
            idx = np.flatnonzero(active)[decide]
            removed_at[idx] = t
            active[idx] = False

    return removed_at, probe_inc


def main():
    os.makedirs("outputs", exist_ok=True)
    px = np.maximum(P_GRID, 8e-4)  # for log axis

    # ---------------- Scenario A ----------------
    print("=" * 72)
    print(f"Scenario A: problem genuinely solved "
          f"(lambda_low={LAMBDA_LOW}, lambda_safe={LAMBDA_SAFE}, "
          f"e_true={E_TRUE} unknown to observer)")
    print("=" * 72)
    fig1, ax1 = plt.subplots(figsize=(7.5, 5.0))
    for sigma_d in SIGMA_DRIFTS:
        med, frac = [], []
        for p in P_GRID:
            removed_at, _ = run_cell(p, sigma_d, LAMBDA_LOW)
            d = np.where(removed_at >= 0, removed_at, np.inf)
            med.append(np.median(d))
            frac.append(np.mean(np.isfinite(d)))
        p_star = analytic_pstar(sigma_d)
        print(f"\nsigma_d={sigma_d:.2f}: analytic p* = {p_star:.4f}")
        for p, m, f in zip(P_GRID, med, frac):
            print(f"   p={p:5.3f}  median years to removal="
                  f"{m if np.isfinite(m) else float('inf'):8.1f}   "
                  f"removed within {HORIZON}y: {f:5.2f}")
        plot = np.where(np.isfinite(med), med, HORIZON * 1.6)
        line, = ax1.plot(px, plot, "o-", label=f"σ_d = {sigma_d}")
        if 0 < p_star < 0.6:
            ax1.axvline(p_star, ls=":", color=line.get_color(), alpha=0.8)

    ax1.axhline(HORIZON, color="grey", ls="--", lw=1)
    ax1.text(9e-4, HORIZON * 1.06, "censored (never removed within horizon)",
             fontsize=8, color="grey")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("probe rate p (fraction of exposure in protected space)")
    ax1.set_ylabel("median years from solved to justified removal")
    ax1.set_title("Decidability of the sunset decision vs probe rate\n"
                  "(dotted: analytic p* per drift level)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig("outputs/paper_xiv_sunset_time_to_removal.png", dpi=160)

    # ---------------- Scenario B ----------------
    print()
    print("=" * 72)
    print(f"Scenario B: regulator still necessary (lambda ~ {LAMBDA_B})")
    print("=" * 72)
    sigma_d = 0.06
    fr, cost = [], []
    for p in P_GRID:
        removed_at, probe_inc = run_cell(p, sigma_d, LAMBDA_B)
        fr.append(np.mean(removed_at >= 0))
        cost.append(np.mean(probe_inc))
        print(f"   p={p:5.3f}  false-removal rate={fr[-1]:5.2f}  "
              f"mean cumulative probe incidents={cost[-1]:8.1f}")

    fig2, ax2 = plt.subplots(figsize=(7.5, 5.0))
    ax2.plot(px, fr, "s-", color="tab:red")
    ax2.set_xscale("log"); ax2.set_ylim(-0.02, 1)
    ax2.set_xlabel("probe rate p")
    ax2.set_ylabel("false-removal rate (30y horizon equiv.)", color="tab:red")
    ax2b = ax2.twinx()
    ax2b.plot(px, cost, "^-", color="tab:blue")
    ax2b.set_ylabel("mean cumulative probe incidents", color="tab:blue")
    ax2.set_title("The probe is not free: error and cost vs probe rate\n"
                  f"(σ_d = {sigma_d}, regulator genuinely necessary)")
    fig2.tight_layout()
    fig2.savefig("outputs/paper_xiv_sunset_tradeoff.png", dpi=160)
    print("\nFigures written to outputs/.")


if __name__ == "__main__":
    main()
