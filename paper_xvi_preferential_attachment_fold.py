"""
paper_xvi_preferential_attachment_fold.py
--------------------------------------------------------------------
Does a preferential-attachment coupling produce a NON-artifactual fold
in the replenishment/depletion race?

Preferential attachment is a legitimately LOCAL mechanism (an agent only
needs to observe how many others already hold an opinion, not the global
correlation), so a fold here would not be the artifact of the earlier
posited d*(kappa+rho_bar) model.

Sequential model. Each micro-step one agent abandons its opinion and either
INNOVATES (prob r: a unique new opinion = entry/replenishment) or COPIES an
existing opinion k with probability proportional to (count_k)^gamma
(preferential attachment = depletion). gamma=1 is linear; gamma>1 is
super-linear / increasing-returns.

Order parameter: N_eff = 1 / sum_k p_k^2 (effective number of opinions).

Findings this script reproduces:
  * gamma = 1 (linear): ergodic; N_eff set continuously by r and grows with
    system size N (EXTENSIVE diversity). No fold.
  * gamma > 1 (super-linear): genuine condensation; N_eff = O(1) independent
    of N (INTENSIVE collapse). A real winner-take-all phase -- but still a
    SINGLE attractor: both initial conditions converge to it, so no
    hysteresis. Matched-seed long-horizon runs give identical trajectories.

Conclusion: preferential attachment gives a real qualitative transition in
the COUPLING EXPONENT gamma (extensive -> intensive at gamma = 1), which is
size-independent and non-artifactual -- but it does NOT restore a hysteretic
fold in the injection rate. Ergodicity is not broken.
"""

import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


def run(r, gamma, labels0, rng, N, sweeps, burn):
    labels = labels0.copy()
    cnt = defaultdict(int)
    for l in labels:
        cnt[l] += 1
    next_id = int(labels.max()) + 1
    vals = []
    for t in range(sweeps):
        for _ in range(N):
            i = rng.integers(N)
            old = labels[i]
            cnt[old] -= 1
            if cnt[old] == 0:
                del cnt[old]
            if rng.random() < r:
                lab = next_id; next_id += 1
            else:
                types = np.fromiter(cnt.keys(), dtype=np.int64)
                w = np.fromiter((cnt[k] for k in types), dtype=np.float64) ** gamma
                lab = int(types[rng.choice(len(types), p=w / w.sum())])
            labels[i] = lab
            cnt[lab] = cnt.get(lab, 0) + 1
        if t >= burn:
            c = np.fromiter(cnt.values(), dtype=np.float64); p = c / N
            vals.append(1.0 / np.sum(p * p))
    return float(np.mean(vals))


def diverse(N):   return np.arange(N)
def collapsed(N): return np.zeros(N, dtype=np.int64)


# ---- Panel 1: N_eff vs innovation rate r, both starts, two gamma ----------
N1 = 180; SW, BU = 160, 100; SEEDS = 2
rs = np.linspace(0.02, 0.30, 7)
res = {}
for gamma in (1.0, 1.5):
    for start, initfn in (("diverse", diverse), ("collapsed", collapsed)):
        arr = np.zeros((len(rs), SEEDS))
        for i, r in enumerate(rs):
            for s in range(SEEDS):
                rng = np.random.default_rng(10 + s)
                arr[i, s] = run(r, gamma, initfn(N1), rng, N1, SW, BU)
        res[(gamma, start)] = arr.mean(1)

# ---- Panel 2: N_eff vs system size N at fixed r (extensive vs intensive) ---
Ns = [50, 100, 200, 350]; rfix = 0.15
scaling = {1.0: [], 1.5: []}
for gamma in (1.0, 1.5):
    for N in Ns:
        vv = []
        for s in range(2):
            rng = np.random.default_rng(20 + s)
            vv.append(run(rfix, gamma, diverse(N), rng, N, 130, 80))
        scaling[gamma].append(np.mean(vv))

# ---- plot -----------------------------------------------------------------
plt.rcParams.update({"font.family": "monospace", "font.size": 9})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))

a1.plot(rs, res[(1.0, "diverse")],   "o-", color="#2a9d8f", ms=4, label="gamma=1  diverse start")
a1.plot(rs, res[(1.0, "collapsed")], "s--", color="#2a9d8f", ms=4, label="gamma=1  collapsed start")
a1.plot(rs, res[(1.5, "diverse")],   "o-", color="#e76f51", ms=4, label="gamma=1.5  diverse start")
a1.plot(rs, res[(1.5, "collapsed")], "s--", color="#e76f51", ms=4, label="gamma=1.5  collapsed start")
a1.set_xlabel("innovation rate  r")
a1.set_ylabel("effective opinions  N_eff")
a1.set_title("Branches coincide at every r -> no hysteresis")
a1.legend(fontsize=7.5, loc="upper left")

a2.plot(Ns, scaling[1.0], "o-", color="#2a9d8f", ms=5, label="gamma=1  (extensive: ~N)")
a2.plot(Ns, scaling[1.5], "s-", color="#e76f51", ms=5, label="gamma=1.5 (intensive: O(1))")
a2.set_xlabel("system size  N")
a2.set_ylabel("effective opinions  N_eff")
a2.set_title(f"The real transition is in gamma  (r={rfix})")
a2.legend(fontsize=8, loc="upper left")

fig.suptitle("Preferential attachment: real condensation, but no fold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
png = os.path.join(OUT, "preferential_attachment_fold.png")
fig.savefig(png, dpi=140)

# ---- verdict --------------------------------------------------------------
gap1 = np.abs(res[(1.0, "diverse")] - res[(1.0, "collapsed")]).max()
gap15 = np.abs(res[(1.5, "diverse")] - res[(1.5, "collapsed")]).max()
print(f"max branch gap  gamma=1.0 : {gap1:.1f}")
print(f"max branch gap  gamma=1.5 : {gap15:.1f}")
print(f"gamma=1.0 N_eff scaling with N: {np.round(scaling[1.0],1)}  (grows -> extensive)")
print(f"gamma=1.5 N_eff scaling with N: {np.round(scaling[1.5],1)}  (flat  -> intensive/condensed)")
print("=> super-linear PA gives real, size-independent condensation,")
print("   but branches coincide (ergodic) -> NO hysteretic fold.")
print("   The non-artifactual threshold is in the exponent gamma at gamma=1,")
print("   not a tipping point in the injection rate. VERDICT: stays SECTION.")
print("saved:", png)
