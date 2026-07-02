"""
paper_xvi_replenishment_depletion.py
--------------------------------------------------------------------
Race between variety REPLENISHMENT (entry of independent observers) and
variety DEPLETION (herding / optimization toward consensus).

Question: is there a critical injection ratio below which the observer
ensemble collapses to N_eff -> 1, and is the collapse history-dependent?

Order parameter: rho_bar, mean off-diagonal correlation of the ensemble's
belief vectors. Effective observers:  N_eff = N / (1 + (N-1)*max(rho_bar,0)).

Per step, two competing operators:
  Depletion (herding, SUPER-LINEAR): each agent pulled toward consensus c
    with strength d*(kappa + rho_bar). The rho_bar factor makes herding
    self-reinforcing -- consensus begets consensus. kappa is a small
    baseline so rho_bar=0 is not neutrally stable.
  Replenishment (entry): each agent is, with probability p_r, replaced by
    a fresh independent draw. This is the "external source term" in its
    literal form -- immigration / mutation / a new cohort with independent
    priors -- not mere jitter around the existing consensus.

We sweep the ratio p_r/d from two initial conditions (decorrelated vs
already-collapsed). Where the two steady-state branches diverge over a
finite range of p_r/d, the system is BISTABLE: a fold (saddle-node), with
hysteresis. Where they coincide, the transition is continuous.

Tiering: a TOY ensemble, not a governance system. A threshold here is [IP]
structural evidence about the replenishment/depletion mechanism, not [R]
proof about institutions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

N, K   = 60, 30      # observers, belief dimensions
D      = 0.15        # depletion (herding) rate -- fixed; we sweep entry vs this
KAPPA  = 0.03        # baseline herding
STEPS  = 1200
BURN   = 800
SEEDS  = 6
OUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


def rho_bar(X):
    C = np.corrcoef(X)
    iu = np.triu_indices(N, k=1)
    return float(np.mean(C[iu]))


def n_eff(rb):
    return N / (1.0 + (N - 1) * max(rb, 0.0))


def run(p_r, x0, rng):
    X = x0.copy()
    tr = []
    for t in range(STEPS):
        rb = rho_bar(X)
        c = X.mean(axis=0, keepdims=True)
        X = X + D * (KAPPA + max(rb, 0.0)) * (c - X)      # depletion
        mask = rng.random(N) < p_r                        # replenishment (entry)
        if mask.any():
            X[mask] = rng.standard_normal((int(mask.sum()), K))
        if t >= BURN:
            tr.append(rho_bar(X))
    return float(np.mean(tr))


def init_open(rng):
    return rng.standard_normal((N, K))                    # decorrelated


def init_collapsed(rng):
    base = rng.standard_normal((1, K))
    return base + 0.1 * rng.standard_normal((N, K))       # rho_bar ~ 0.95


# ---- sweep ----------------------------------------------------------------
ratios = np.linspace(0.0, 0.25, 51)
open_neff = np.zeros_like(ratios)
coll_neff = np.zeros_like(ratios)

for i, ratio in enumerate(ratios):
    p_r = ratio * D
    o, c = [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(100 + s)
        o.append(run(p_r, init_open(rng), rng))
        c.append(run(p_r, init_collapsed(rng), rng))
    open_neff[i] = n_eff(np.mean(o))
    coll_neff[i] = n_eff(np.mean(c))

# ---- locate the fold edges ------------------------------------------------
level = N / 2.0
def first_above(x, y, lv):
    idx = np.where(y >= lv)[0]
    return float(x[idx[0]]) if len(idx) else np.nan

collapse_edge = first_above(ratios, open_neff, level)   # open state recovers above here
recovery_edge = first_above(ratios, coll_neff, level)   # collapsed state recovers above here
hyst = recovery_edge - collapse_edge

# ---- plot -----------------------------------------------------------------
plt.rcParams.update({"font.family": "monospace", "font.size": 10})
fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.plot(ratios, open_neff, "o-", ms=3, color="#2a9d8f", label="from decorrelated start")
ax.plot(ratios, coll_neff, "s-", ms=3, color="#e76f51", label="from collapsed start")
ax.axhline(1, ls=":", c="0.6", lw=1)
if not np.isnan(collapse_edge):
    ax.axvline(collapse_edge, ls="--", c="#2a9d8f", alpha=.5)
if not np.isnan(recovery_edge):
    ax.axvline(recovery_edge, ls="--", c="#e76f51", alpha=.5)
if not (np.isnan(collapse_edge) or np.isnan(recovery_edge)):
    ax.axvspan(collapse_edge, recovery_edge, color="0.85", alpha=.5, zorder=0)
    ax.text((collapse_edge + recovery_edge) / 2, N * 0.55, "bistable\n(hysteresis)",
            ha="center", va="center", fontsize=8, color="0.3")
ax.set_xlabel("injection ratio   entry rate / depletion rate")
ax.set_ylabel("effective observers   N_eff")
ax.set_title("Replenishment vs depletion: the variety-gap fold")
ax.legend(fontsize=8, loc="center right")
fig.tight_layout()
png = os.path.join(OUT, "replenishment_depletion.png")
fig.savefig(png, dpi=140)

# ---- report ---------------------------------------------------------------
print(f"collapse edge  (open branch collapses below):     r/d = {collapse_edge:.3f}")
print(f"recovery edge  (collapsed branch recovers above): r/d = {recovery_edge:.3f}")
print(f"hysteresis window width:                          {hyst:.3f}")
step = ratios[1] - ratios[0]
if not np.isnan(hyst) and hyst > step:
    print("=> FOLD: two stable states coexist across a finite window.")
    print("   Threshold found + history dependence. Graduation criterion met.")
else:
    print("=> continuous/soft transition. Stays section weight.")
print(f"saved: {png}")
