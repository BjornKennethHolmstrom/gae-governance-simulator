"""
self_iii_formation.py
=====================

Simulation for the "Formation of the Observer" section of Self III.

It tests, rather than assumes, the central conjecture of that section: that the
conditions of early formation reproduce a formative source's BLINDNESS with
higher fidelity than its SIGHT, and that this asymmetry is governed by the
number of decorrelated alternative observers a child can reach.

Model
-----
State space of D dimensions. A formative SOURCE perceives a subset of them
(its observable set); the rest are its unobservable subspace. A child's own
observation matrix is written during a formative window by acquiring dimensions
from whatever sources it can reach. The source is always reachable. There are N
potential ALTERNATIVE sources, each with an independently drawn observable set;
their independence from the source is what makes them decorrelated and therefore
able, in principle, to supply what the source cannot see.

A dimension is ACQUIRED by the child if at least one reachable source that
observes that dimension successfully transmits it (per-source transmission
probability t). "Lock strength" L in [0,1] removes alternative sources from the
child's reach: the child reaches k = round((1-L) * N) alternatives. L=0 is the
open regime (all alternatives reachable); L=1 is the sole-source regime.

Two knobs of interest:
  * lock strength L      -- how many decorrelated alternatives retain standing
  * correlation rho      -- how much each alternative shares the SOURCE's
                            observable set (rho=0 fully decorrelated;
                            rho=1 alternatives copy the source, sharing its
                            blind spots and adding nothing)

Measured per condition:
  * inside_acq   -- mean acquisition rate over dims the source CAN see
  * outside_acq  -- mean acquisition rate over dims the source CANNOT see
  * overlap      -- fraction of the source's blind set the child is ALSO blind to
                    (the operational "inheritance of unobservability")
  * complete_out -- fraction of children who acquire the FULL outside set
                    (a stricter, completeness metric)

Nothing about a threshold is assumed. The shapes that come out of the run are
whatever the mechanism produces.

Run:  python3 self_iii_formation.py
Output: prints a numeric summary; writes outputs/self_iii_formation.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
SEED          = 7
D             = 120     # dimensionality of the state space
F_SOURCE      = 0.50    # fraction of dimensions the formative source can perceive
N_ALT         = 30      # number of potential alternative sources
Q_ALT         = 0.15    # per-source coverage of an alternative (decorrelated part)
T_TRANSMIT    = 0.60    # per-(source, dimension) transmission probability
TRIALS        = 600     # Monte Carlo trials per condition

LOCK_GRID     = np.linspace(0.0, 1.0, 21)   # Experiment 1: sweep lock strength
RHO_GRID      = np.linspace(0.0, 1.0, 21)   # Experiment 2: sweep alt/source correlation

OUTDIR        = "outputs"


# ----------------------------------------------------------------------
# Core mechanism
# ----------------------------------------------------------------------
def draw_source(rng):
    """Boolean observable set of the formative source over D dimensions."""
    return rng.random(D) < F_SOURCE


def draw_alternatives(rng, source, rho):
    """
    (N_ALT, D) boolean array of alternative sources' observable sets.

    With probability rho an alternative copies the source's status for a
    dimension (correlated, shares the source's blind spots); otherwise the
    dimension is covered independently with probability Q_ALT (decorrelated).
    """
    copy_mask = rng.random((N_ALT, D)) < rho
    independent = rng.random((N_ALT, D)) < Q_ALT
    return np.where(copy_mask, source[None, :], independent)


def acquire(rng, source, alternatives, k):
    """
    Realised acquisition (boolean over D) for one child reaching the source and
    the first k alternatives. A dimension is acquired iff at least one reachable
    observing source transmits it; OR of n independent Bernoulli(t) successes is
    Bernoulli(1 - (1-t)^n).
    """
    n_obs = source.astype(np.int64).copy()
    if k > 0:
        n_obs = n_obs + alternatives[:k].sum(axis=0)
    p_acq = 1.0 - (1.0 - T_TRANSMIT) ** n_obs
    return rng.random(D) < p_acq


def run_condition(rng, lock, rho):
    """Average metrics over TRIALS for a given lock strength and correlation."""
    k = int(round((1.0 - lock) * N_ALT))
    inside_acc = outside_acc = overlap_acc = complete_acc = 0.0

    for _ in range(TRIALS):
        source = draw_source(rng)
        alts = draw_alternatives(rng, source, rho)
        acquired = acquire(rng, source, alts, k)

        inside = source            # dims the source CAN see
        outside = ~source          # dims the source CANNOT see
        n_out = outside.sum()

        inside_acc  += acquired[inside].mean() if inside.any() else 0.0
        out_rate     = acquired[outside].mean() if n_out else 0.0
        outside_acc += out_rate
        # overlap: fraction of the source's blind set the child is also blind to
        overlap_acc += (~acquired[outside]).mean() if n_out else 0.0
        # completeness: did the child acquire EVERY outside dimension?
        complete_acc += 1.0 if (n_out and acquired[outside].all()) else 0.0

    inv = 1.0 / TRIALS
    return (inside_acc * inv, outside_acc * inv,
            overlap_acc * inv, complete_acc * inv)


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
def experiment_lock(rng, rho=0.0):
    """Sweep lock strength with decorrelated alternatives (rho=0)."""
    rows = []
    for L in LOCK_GRID:
        rows.append(run_condition(rng, lock=L, rho=rho))
    return np.array(rows)  # columns: inside, outside, overlap, complete


def experiment_corr(rng, lock=0.0):
    """Sweep alternative/source correlation with all alternatives reachable."""
    rows = []
    for r in RHO_GRID:
        rows.append(run_condition(rng, lock=lock, rho=r))
    return np.array(rows)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def print_summary(lock_res, corr_res):
    print("=" * 70)
    print("EXPERIMENT 1  -- lock strength sweep (decorrelated alternatives)")
    print("=" * 70)
    print(f"{'lock':>6} {'inside':>9} {'outside':>9} {'overlap':>9} {'complete':>9}")
    for L, row in zip(LOCK_GRID, lock_res):
        print(f"{L:6.2f} {row[0]:9.3f} {row[1]:9.3f} {row[2]:9.3f} {row[3]:9.3f}")

    inside = lock_res[:, 0]
    outside = lock_res[:, 1]
    complete = lock_res[:, 3]
    print("\nAsymmetry readout:")
    print(f"  inside  range over lock: {inside.max():.3f} -> {inside.min():.3f} "
          f"(span {inside.max()-inside.min():.3f}; floor near t={T_TRANSMIT})")
    print(f"  outside range over lock: {outside.max():.3f} -> {outside.min():.3f} "
          f"(span {outside.max()-outside.min():.3f})")
    # steepest segment of each curve (where the biggest single-step drop occurs)
    d_out = np.diff(outside)
    d_comp = np.diff(complete)
    i_out = int(np.argmin(d_out))
    i_comp = int(np.argmin(d_comp))
    print(f"  steepest drop in mean outside-acq between lock="
          f"{LOCK_GRID[i_out]:.2f} and {LOCK_GRID[i_out+1]:.2f} "
          f"(delta {d_out[i_out]:.3f})")
    print(f"  steepest drop in completeness   between lock="
          f"{LOCK_GRID[i_comp]:.2f} and {LOCK_GRID[i_comp+1]:.2f} "
          f"(delta {d_comp[i_comp]:.3f})")
    # half-fall point of completeness
    half = _crossing(LOCK_GRID, complete, 0.5)
    if half is not None:
        print(f"  completeness crosses 0.5 at lock ~ {half:.3f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2  -- correlation sweep (all alternatives reachable, lock=0)")
    print("=" * 70)
    print(f"{'rho':>6} {'inside':>9} {'outside':>9} {'overlap':>9} {'complete':>9}")
    for r, row in zip(RHO_GRID, corr_res):
        print(f"{r:6.2f} {row[0]:9.3f} {row[1]:9.3f} {row[2]:9.3f} {row[3]:9.3f}")
    out0, out1 = corr_res[0, 1], corr_res[-1, 1]
    print(f"\n  outside-acq with decorrelated alternatives (rho=0): {out0:.3f}")
    print(f"  outside-acq with correlated   alternatives (rho=1): {out1:.3f}")
    print(f"  decorrelation buys {out0-out1:.3f} of outside acquisition\n")


def _crossing(x, y, level):
    """First x at which y crosses `level` (linear interp); None if no crossing."""
    for i in range(len(y) - 1):
        a, b = y[i], y[i + 1]
        if (a - level) * (b - level) <= 0 and a != b:
            f = (level - a) / (b - a)
            return x[i] + f * (x[i + 1] - x[i])
    return None


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def make_figure(lock_res, corr_res):
    os.makedirs(OUTDIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.plot(LOCK_GRID, lock_res[:, 0], "o-", label="inside-acq (source can see)")
    ax1.plot(LOCK_GRID, lock_res[:, 1], "s-", label="outside-acq (source cannot see)")
    ax1.plot(LOCK_GRID, lock_res[:, 2], "^--", label="blind-set overlap (inherited)")
    ax1.plot(LOCK_GRID, lock_res[:, 3], "d:", label="full outside set acquired")
    ax1.axhline(T_TRANSMIT, color="grey", lw=0.8, ls=":", alpha=0.7)
    ax1.set_xlabel("lock strength  (alternative sources removed)")
    ax1.set_ylabel("rate")
    ax1.set_title("Experiment 1: inheritance asymmetry vs lock strength")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=8, loc="center left")
    ax1.grid(alpha=0.25)

    ax2.plot(RHO_GRID, corr_res[:, 1], "s-", label="outside-acq")
    ax2.plot(RHO_GRID, corr_res[:, 0], "o-", label="inside-acq")
    ax2.set_xlabel("alternative/source correlation  rho")
    ax2.set_ylabel("rate")
    ax2.set_title("Experiment 2: decorrelation is what fills the blind set")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(fontsize=8, loc="center left")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "self_iii_formation.png")
    fig.savefig(path, dpi=130)
    print(f"figure written to {path}")


# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    lock_res = experiment_lock(rng, rho=0.0)
    corr_res = experiment_corr(rng, lock=0.0)
    print_summary(lock_res, corr_res)
    make_figure(lock_res, corr_res)


if __name__ == "__main__":
    main()
