#!/usr/bin/env python3
"""
paper_xx-goodhart_demo.py
=====================
Registered Goodhart intervention-set demo per paper_xx-goodhart_demo_preregistration.md.

Tests the corollary of the intervention-set theorem: a lossy projection M = a^p
(which drops the target-relevant hidden dimension b) is UNSAFE only when b is
reachable by the optimizer. When b is frozen, the budget constraint forces the
proxy optimum onto the true optimum, and Goodhart vanishes.

Everything is closed-form: the optimum of M = a^p over each reachable set is
analytic, so there is no training and no stochastic optimization. The only
randomness is the per-seed world draw (C, p, alpha, beta). Runs in seconds.

Usage:
    python paper_xx-goodhart_demo.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def spearman(x, y):
    """Rank correlation, numpy-only (avg-rank ties)."""
    def rank(v):
        v = np.asarray(v, float)
        order = v.argsort()
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        # average ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts)); np.add.at(sums, inv, r)
        return (sums / counts)[inv]
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 30
R_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def world(seed):
    rng = np.random.default_rng(seed)
    C = rng.uniform(5, 20)
    p = rng.uniform(0.3, 0.7)
    alpha = rng.uniform(0.8, 1.2)
    beta = rng.uniform(0.8, 1.2)
    return C, p, alpha, beta


def target(a, b, p, alpha, beta):
    return alpha * a**p + beta * b**p


def true_optimum(C, p, alpha, beta):
    """Interior optimum of alpha a^p + beta b^p s.t. a + b = C.
    Marginal condition: alpha p a^(p-1) = beta p b^(p-1)
      => a/b = (alpha/beta)^(1/(1-p))."""
    ratio = (alpha / beta) ** (1.0 / (1.0 - p))   # a*/b*
    b_star = C / (1.0 + ratio)
    a_star = C - b_star
    return a_star, b_star


def proxy_optimum(C, p, alpha, beta, r):
    """Optimizer maximizes M = a^p. It may reduce b from b* down to (1-r) b*,
    reallocating the freed budget to a. Since M is increasing in a, it pushes b
    to its floor.  a = C - b_floor."""
    a_star, b_star = true_optimum(C, p, alpha, beta)
    b_floor = (1.0 - r) * b_star
    a = C - b_floor
    b = b_floor
    return a, b


def run():
    rows = []          # (seed, r, D, M, T)
    endpoint = {}      # seed -> {"D0":..., "D1":...}
    for seed in range(N_SEEDS):
        C, p, alpha, beta = world(seed)
        a_star, b_star = true_optimum(C, p, alpha, beta)
        T_star = target(a_star, b_star, p, alpha, beta)
        per_r = {}
        for r in R_GRID:
            a, b = proxy_optimum(C, p, alpha, beta, r)
            T = target(a, b, p, alpha, beta)
            M = a**p
            D = (T_star - T) / T_star
            rows.append((seed, r, D, M, T))
            per_r[r] = D
        endpoint[seed] = {"D0": per_r[0.0], "D1": per_r[1.0], "curve": per_r}

    # ---- registered predictions ----
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("=" * 60)
    out(f"GOODHART INTERVENTION-SET DEMO — {N_SEEDS} worlds")
    out("=" * 60)

    D1 = np.array([endpoint[s]["D1"] for s in range(N_SEEDS)])
    D0 = np.array([endpoint[s]["D0"] for s in range(N_SEEDS)])

    # P1
    p1_n = int(np.sum(D1 > 0.10))
    out(f"\nP1 (Goodhart at r=1, D>0.10): {p1_n}/{N_SEEDS} (pass >= 24)")
    out(f"  median D(1) = {np.median(D1):.3f} [{np.percentile(D1,25):.3f}, {np.percentile(D1,75):.3f}]")
    out(f"  -> {'PASS' if p1_n >= 24 else 'FAIL'}")

    # P2 (the real test)
    p2_n = int(np.sum(D0 < 0.01))
    out(f"\nP2 (safe at r=0, D<0.01 — lossy proxy, unreachable dim): "
        f"{p2_n}/{N_SEEDS} (pass >= 28)")
    out(f"  max D(0) across seeds = {np.max(D0):.2e}")
    out(f"  -> {'PASS' if p2_n >= 28 else 'FAIL'}")

    # P3
    all_r = [row[1] for row in rows]
    all_D = [row[2] for row in rows]
    rho = spearman(all_r, all_D)
    mono = 0
    for s in range(N_SEEDS):
        c = endpoint[s]["curve"]
        seq = [c[r] for r in R_GRID]
        if all(seq[i+1] >= seq[i] - 1e-12 for i in range(len(seq)-1)):
            mono += 1
    out(f"\nP3 (severity scales with reachability):")
    out(f"  pooled Spearman(r, D) = {rho:.3f} (pass > 0.9)")
    out(f"  per-seed monotone: {mono}/{N_SEEDS} (pass >= 27)")
    out(f"  -> {'PASS' if rho > 0.9 and mono >= 27 else 'FAIL'}")

    # degradation table
    out("\nDegradation D(r), median [IQR] across worlds:")
    for r in R_GRID:
        vals = np.array([row[2] for row in rows if row[1] == r])
        out(f"  r={r:.2f}: {np.median(vals):.3f} "
            f"[{np.percentile(vals,25):.3f}, {np.percentile(vals,75):.3f}]")

    with open(os.path.join(OUT, "goodhart_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- figure: the Goodhart scissors ----
    meanM = [np.mean([row[3] for row in rows if row[1] == r]) for r in R_GRID]
    meanT = [np.mean([row[4] for row in rows if row[1] == r]) for r in R_GRID]
    # normalize each to its r=0 value for a shared axis
    m0, t0 = meanM[0], meanT[0]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(R_GRID, [m/m0 for m in meanM], "o-", label="proxy M (normalized)", color="#4477aa")
    ax.plot(R_GRID, [t/t0 for t in meanT], "s-", label="target T (normalized)", color="#ee6677")
    ax.set_xlabel("reachability of the discarded dimension  (r)")
    ax.set_ylabel("mean value, normalized to r=0")
    ax.set_title("Goodhart scissors: proxy rises, target falls, as the\n"
                 "forgotten dimension becomes reachable")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "goodhart_scissors.png"), dpi=150)
    plt.close(fig)

    print(f"\nSummary -> {os.path.join(OUT, 'goodhart_summary.txt')}")
    print(f"Figure  -> {os.path.join(OUT, 'goodhart_scissors.png')}")


if __name__ == "__main__":
    run()
