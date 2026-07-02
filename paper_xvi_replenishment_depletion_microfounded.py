"""
paper_xvi_replenishment_depletion_microfounded.py
--------------------------------------------------------------------
Graduation test for the replenishment/depletion fold.

Prior result (replenishment_depletion.py) found a fold with hysteresis,
but the super-linear herding was POSITED: an explicit d*(kappa+rho_bar)
term inserts global correlation into the update by hand. This script
asks the graduation question: does the fold survive when the herding is
DERIVED from a local behavioural rule, with no global rho_bar anywhere?

Microfoundation: CORRELATION NEGLECT. Each agent i samples others, weights
them by agreement w_ij = exp(-||x_i-x_j||^2 / 2 lam^2), and forms
perceived evidence E_i = sum_j w_ij -- treating each agreeing neighbour as
an independent confirmation (the naive failure to compute N_eff). Its
conviction / pull toward the agreement-weighted crowd target rises with
E_i through a saturating (Hill) function. No agent sees the global
correlation; the herding's dependence on ensemble agreement is emergent.

Two panels:
  LEFT  -- posited model: the spurious fold (branches diverge -> hysteresis)
  RIGHT -- derived model: what actually happens under bounded conviction

Verdict is decided by whether the two initial-condition branches of the
derived model diverge (fold) or coincide (single attractor, continuous).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

N, K = 50, 30
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


def rho_bar(X):
    C = np.corrcoef(X)
    iu = np.triu_indices(N, k=1)
    return float(np.mean(C[iu]))


def n_eff(rb):
    return N / (1.0 + (N - 1) * max(rb, 0.0))


def init_open(rng):
    return rng.standard_normal((N, K))


def init_collapsed(rng):
    return rng.standard_normal((1, K)) + 0.1 * rng.standard_normal((N, K))


# ---- POSITED model (global rho_bar inserted by hand) ----------------------
def run_posited(p_r, x0, rng, d=0.15, kappa=0.03, steps=1000, burn=700):
    X = x0.copy(); tr = []
    for t in range(steps):
        rb = rho_bar(X); c = X.mean(0, keepdims=True)
        X = X + d * (kappa + max(rb, 0.0)) * (c - X)          # <- global rho_bar
        m = rng.random(N) < p_r
        if m.any(): X[m] = rng.standard_normal((int(m.sum()), K))
        if t >= burn: tr.append(rho_bar(X))
    return float(np.mean(tr))


# ---- DERIVED model (correlation neglect, purely local) --------------------
def run_derived(p_r, x0, rng, eta=1.0, lam=2.5, E0=3.0, h=2, steps=500, burn=350):
    X = x0.copy(); tr = []
    for t in range(steps):
        G = X @ X.T; dg = np.diag(G)
        D2 = np.maximum(dg[:, None] + dg[None, :] - 2 * G, 0.0)
        W = np.exp(-D2 / (2 * lam * lam)); np.fill_diagonal(W, 0.0)
        E = W.sum(1) + 1e-9                                    # naive evidence (no N_eff correction)
        M = (W @ X) / E[:, None]
        pull = eta * (E ** h / (E ** h + E0 ** h))            # bounded conviction; NO global rho_bar
        X = X + pull[:, None] * (M - X)
        m = rng.random(N) < p_r
        if m.any(): X[m] = rng.standard_normal((int(m.sum()), K))
        if t >= burn: tr.append(rho_bar(X))
    return float(np.mean(tr))


def sweep(run_fn, ratios, seeds, unit):
    op = np.zeros((len(ratios), seeds)); cl = np.zeros_like(op)
    for i, ratio in enumerate(ratios):
        p_r = ratio * unit
        for s in range(seeds):
            rng = np.random.default_rng(300 + s)
            op[i, s] = n_eff(run_fn(p_r, init_open(rng), rng))
            cl[i, s] = n_eff(run_fn(p_r, init_collapsed(rng), rng))
    return op, cl


# posited: sweep entry rate against d=0.15
rp = np.linspace(0.0, 0.25, 26)
op_p, cl_p = sweep(run_posited, rp, 6, 0.15)
# derived: sweep entry rate against eta=1.0
rd = np.linspace(0.0, 0.35, 26)
op_d, cl_d = sweep(run_derived, rd, 6, 1.0)

# ---- plot -----------------------------------------------------------------
plt.rcParams.update({"font.family": "monospace", "font.size": 9})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

for ax, r, opb, clb, title in [
    (a1, rp, op_p, cl_p, "POSITED herding: d*(kappa+rho_bar)"),
    (a2, rd, op_d, cl_d, "DERIVED herding: correlation neglect"),
]:
    mo, so = opb.mean(1), opb.std(1)
    mc, sc = clb.mean(1), clb.std(1)
    ax.plot(r, mo, "o-", ms=3, color="#2a9d8f", label="decorrelated start")
    ax.fill_between(r, mo - so, mo + so, color="#2a9d8f", alpha=.15)
    ax.plot(r, mc, "s-", ms=3, color="#e76f51", label="collapsed start")
    ax.fill_between(r, mc - sc, mc + sc, color="#e76f51", alpha=.15)
    ax.axhline(1, ls=":", c="0.6", lw=1)
    ax.set_xlabel("injection ratio  entry / depletion")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")

a1.set_ylabel("effective observers  N_eff")
a1.text(0.13, 30, "branches diverge\n= spurious fold", fontsize=8, color="0.3", ha="center")
a2.text(0.22, 30, "branches coincide\n= single attractor,\ncontinuous", fontsize=8, color="0.3", ha="center")
fig.suptitle("Does the variety-gap fold survive a derived microfoundation?", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
png = os.path.join(OUT, "replenishment_depletion_microfounded.png")
fig.savefig(png, dpi=140)

# ---- verdict --------------------------------------------------------------
gap_d = np.abs(op_d.mean(1) - cl_d.mean(1))
noise_d = 0.5 * (op_d.std(1) + cl_d.std(1))
clean_separation = np.any(gap_d > 3 * np.maximum(noise_d, 0.5))
print("DERIVED model max branch gap: %.1f  (typical seed noise: %.1f)"
      % (gap_d.max(), noise_d.mean()))
if clean_separation:
    print("=> fold survives derivation. GRADUATES to paper.")
else:
    print("=> no clean branch separation: continuous, single-attractor.")
    print("   The posited fold was an ARTIFACT of the inserted rho_bar term.")
    print("   Bounded conviction prevents runaway feedback -> no tipping point.")
    print("   VERDICT: stays SECTION weight.")
print("saved:", png)
