#!/usr/bin/env python3
"""
paper_xxi-learning_adaptation_demo.py
=====================================
Registered demo for Paper XXI per paper_xxi-learning_adaptation_preregistration.md.

Shows that learning (model fidelity) and adaptation (coupling / viability) dissociate:
as the learning rate rises, the internal estimate tracks the drifting target better
(fidelity up, monotone), but the slew-limited actuator cannot absorb the faster,
noise-reactive revisions, so coupling peaks at an intermediate learning rate and
degrades at high rates. This is the absorptive-capacity inequality
    D_revealed + D_created <= C_absorb
made mechanical: the actuator slew s is C_absorb.

Pure numpy/matplotlib. Seconds on a CPU.

Usage:
    python paper_xxi-learning_adaptation_demo.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ---- fixed configuration (registered) ----
N_SEEDS = 30
STEPS = 4000
BURN_IN = 500
ETAS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

DRIFT = 0.02          # per-step drift of the latent target
NOISE = 0.12          # observation/process noise on the target
TAU = 0.25            # viability band half-width
STIFF = 0.08          # actuator spring stiffness toward the estimate
DAMP = 0.20           # actuator damping
COST = 1.5            # incorporation cost per unit estimate revision (D_created)
DECAY = 0.85          # per-step decay of accumulated disruption


def spearman(x, y):
    """numpy-only rank correlation (average-rank ties)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    def rank(v):
        order = v.argsort()
        r = np.empty(len(v)); r[order] = np.arange(len(v))
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts)); np.add.at(sums, inv, r)
        return (sums / counts)[inv]
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d else 0.0


def simulate(seed, eta):
    """Return (fidelity, coupling) for one run.

    The controller estimates a drifting target at learning rate eta (fidelity).
    A second-order actuator tracks the estimate, but every estimate REVISION
    injects disruption into the action loop proportional to the jump size --
    the incorporation cost D_created -- which decays over DECAY. Fast learning
    produces frequent large revisions, keeping the loop perpetually disrupted so
    the actuator spends more time outside the viability band. The loop's ability
    to absorb revisions is C_absorb; when revision demand exceeds it, coupling
    degrades though fidelity keeps improving.

    fidelity = mean(-|est - theta|) over post-burn-in steps.
    coupling = fraction of post-burn-in steps with |actuator - theta| <= TAU.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-1, 1)          # latent target
    est = theta + rng.normal(0, 0.2)    # controller estimate
    act = theta                          # actuator position
    vel = 0.0                            # actuator velocity (second-order)
    disrupt = 0.0                        # accumulated incorporation disruption
    drift_sign = rng.choice([-1, 1])

    fid_err = np.empty(STEPS)
    in_band = np.empty(STEPS, dtype=bool)
    for t in range(STEPS):
        if abs(theta) > 1.5:
            drift_sign = -np.sign(theta)
        theta = theta + drift_sign * DRIFT + rng.normal(0, NOISE)
        obs = theta + rng.normal(0, NOISE)
        # learning: estimate moves toward observation at rate eta
        prev = est
        est = est + eta * (obs - est)
        jump = abs(est - prev)                       # size of the model revision
        disrupt = DECAY * disrupt + COST * jump      # D_created, accumulated
        # action: second-order actuator chases the estimate
        vel += STIFF * (est - act) - DAMP * vel
        act += vel
        # the loop cannot settle while it is still absorbing revisions
        act_eff = act + rng.normal(0, disrupt)
        fid_err[t] = abs(est - theta)
        in_band[t] = abs(act_eff - theta) <= TAU

    fidelity = -fid_err[BURN_IN:].mean()
    coupling = in_band[BURN_IN:].mean()
    return fidelity, coupling


def run():
    # results[eta] = list over seeds of (fidelity, coupling)
    fid = {e: np.empty(N_SEEDS) for e in ETAS}
    cpl = {e: np.empty(N_SEEDS) for e in ETAS}
    for s in range(N_SEEDS):
        for e in ETAS:
            f, c = simulate(s, e)
            fid[e][s] = f; cpl[e][s] = c

    lines = []
    def out(msg=""):
        print(msg); lines.append(msg)

    out("=" * 60)
    out(f"LEARNING vs ADAPTATION DEMO — {N_SEEDS} seeds")
    out("=" * 60)
    out("\n eta    fidelity(med)   coupling(med [IQR])")
    for e in ETAS:
        fm = np.median(fid[e])
        cm = np.median(cpl[e])
        clo, chi = np.percentile(cpl[e], 25), np.percentile(cpl[e], 75)
        out(f"  {e:.2f}   {fm:7.3f}        {cm:.3f} [{clo:.3f}, {chi:.3f}]")

    # ---- P1: fidelity monotone in eta ----
    pooled_eta = np.concatenate([[e] * N_SEEDS for e in ETAS])
    pooled_fid = np.concatenate([fid[e] for e in ETAS])
    rho_fid = spearman(pooled_eta, pooled_fid)
    p1 = rho_fid > 0.9 and np.median(fid[ETAS[-1]]) > np.median(fid[ETAS[0]])
    out(f"\nP1 (fidelity monotone in eta): pooled Spearman = {rho_fid:.3f} (pass > 0.9)")
    out(f"  -> {'PASS' if p1 else 'FAIL'}")

    # ---- P2: coupling non-monotone, interior peak, high-eta drop >= 0.10 ----
    med_cpl = {e: np.median(cpl[e]) for e in ETAS}
    peak_eta = max(ETAS, key=lambda e: med_cpl[e])
    interior = peak_eta not in (ETAS[0], ETAS[-1])
    # per-seed: coupling at each seed's own peak eta minus coupling at highest eta
    drops = 0
    for s in range(N_SEEDS):
        seed_cpl = {e: cpl[e][s] for e in ETAS}
        seed_peak = max(ETAS, key=lambda e: seed_cpl[e])
        if seed_cpl[seed_peak] - seed_cpl[ETAS[-1]] >= 0.10:
            drops += 1
    p2 = interior and drops >= 24
    out(f"\nP2 (coupling non-monotone, interior peak):")
    out(f"  median-coupling-maximizing eta = {peak_eta} (interior: {interior})")
    out(f"  seeds with peak-to-highest drop >= 0.10: {drops}/{N_SEEDS} (pass >= 24)")
    out(f"  -> {'PASS' if p2 else 'FAIL'}")

    # ---- P3: dissociation regime (fidelity up while coupling down) ----
    diss = 0
    for s in range(N_SEEDS):
        seed_cpl = {e: cpl[e][s] for e in ETAS}
        seed_peak = max(ETAS, key=lambda e: seed_cpl[e])
        if seed_peak == ETAS[-1]:
            continue  # no interval above the peak
        fid_up = fid[ETAS[-1]][s] > fid[seed_peak][s]
        cpl_down = cpl[ETAS[-1]][s] < cpl[seed_peak][s]
        if fid_up and cpl_down:
            diss += 1
    p3 = diss >= 24
    out(f"\nP3 (fidelity up while coupling down, peak->highest eta):")
    out(f"  seeds showing the dissociation: {diss}/{N_SEEDS} (pass >= 24)")
    out(f"  -> {'PASS' if p3 else 'FAIL'}")

    with open(os.path.join(OUT, "learning_adaptation_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- figure: dissociation scissors ----
    fmed = [np.median(fid[e]) for e in ETAS]
    cmed = [np.median(cpl[e]) for e in ETAS]
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax2 = ax1.twinx()
    l1 = ax1.plot(ETAS, fmed, "o-", color="#4477aa", label="fidelity (−tracking error)")
    l2 = ax2.plot(ETAS, cmed, "s-", color="#ee6677", label="coupling (in-band fraction)")
    ax1.set_xscale("log")
    ax1.set_xlabel("learning rate  η  (log scale)")
    ax1.set_ylabel("model fidelity", color="#4477aa")
    ax2.set_ylabel("coupling / viability", color="#ee6677")
    ax1.set_title("Learning improves the map monotonically;\ncoupling peaks then degrades")
    lines_ = l1 + l2
    ax1.legend(lines_, [l.get_label() for l in lines_], loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning_adaptation_scissors.png"), dpi=150)
    plt.close(fig)

    print(f"\nSummary -> {os.path.join(OUT, 'learning_adaptation_summary.txt')}")
    print(f"Figure  -> {os.path.join(OUT, 'learning_adaptation_scissors.png')}")


if __name__ == "__main__":
    run()
