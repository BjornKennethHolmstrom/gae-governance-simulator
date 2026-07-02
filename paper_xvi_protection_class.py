"""
paper_xvi_protection_class.py
--------------------------------------------------------------------
Backs Paper XVI §6 (the residual: the protection class between physically-
uneditable and politically-revocable). Three experiments on the same
replenishment/depletion race, sharing one order parameter:

    N_eff = 1 / sum_k p_k^2   (effective number of live opinions/observers)

The negative controls are load-bearing: the point is that a hysteretic trap
is NOT generic, so the two mechanisms that fail to produce one are kept
beside the one that succeeds.

  A) BOUNDED CONVICTION (correlation neglect).  Agents in R^K weight a crowd
     by apparent agreement, treat agreeing neighbours as independent evidence
     (the local N_eff illusion), and let conviction saturate. NULL: smooth,
     single attractor; both initial conditions coincide.

  B) PREFERENTIAL ATTACHMENT.  Sequential copy ~ count^gamma with innovation
     rate r. NULL for hysteresis: ergodic (forgets initial condition). But a
     REAL, size-independent condensation transition lives in the exponent:
     gamma=1 -> extensive diversity (N_eff ~ N); gamma>1 -> intensive collapse
     (N_eff = O(1)). Condensation, not a fold.

  C) SWITCHING BARRIER (sunk-cost lock-in).  Sequential copy plus a barrier:
     a holder of opinion k stays with prob B*(n_k/N). B parameterises exactly
     the in-between protection class. A genuine, horizon-robust FOLD appears,
     but only above a critical B* ~ 0.95: below it a diverse state is merely
     metastable (decays -> reversible); above it consensus is absorbing and a
     hysteresis window opens in r.

Tiering: [IP], shading to [H] for the specific numbers (B*, windows). The [R]
backbone would be a first-passage derivation of the mean escape time from the
consensus state diverging as B -> 1; simulated here, not derived.
"""

import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


def rho_bar(X, N):
    C = np.corrcoef(X); iu = np.triu_indices(N, 1)
    return float(np.mean(C[iu]))


def neff_rho(rb, N):
    return N / (1.0 + (N - 1) * max(rb, 0.0))


def neff_counts(cnt_vals, N):
    p = np.asarray(cnt_vals, float) / N
    return 1.0 / np.sum(p * p)


# =========================================================================
# A) BOUNDED CONVICTION (correlation neglect) -- vectorised belief vectors
# =========================================================================
def run_conviction(p_r, x0, rng, N, K, eta=1.0, lam=2.5, E0=3.0, h=2,
                   steps=500, burn=350):
    X = x0.copy(); tr = []
    for t in range(steps):
        G = X @ X.T; dg = np.diag(G)
        D2 = np.maximum(dg[:, None] + dg[None, :] - 2 * G, 0.0)
        W = np.exp(-D2 / (2 * lam * lam)); np.fill_diagonal(W, 0.0)
        E = W.sum(1) + 1e-9
        M = (W @ X) / E[:, None]
        pull = eta * (E ** h / (E ** h + E0 ** h))
        X = X + pull[:, None] * (M - X)
        m = rng.random(N) < p_r
        if m.any(): X[m] = rng.standard_normal((int(m.sum()), K))
        if t >= burn: tr.append(rho_bar(X, N))
    return neff_rho(float(np.mean(tr)), N)


def experiment_A():
    N, K = 50, 30
    ratios = np.linspace(0.0, 0.35, 15)
    dv = np.zeros_like(ratios); cl = np.zeros_like(ratios)
    for i, r in enumerate(ratios):
        d, c = [], []
        for s in range(3):
            rng = np.random.default_rng(10 + s)
            d.append(run_conviction(r, rng.standard_normal((N, K)), rng, N, K))
            rng = np.random.default_rng(10 + s)
            base = rng.standard_normal((1, K))
            c.append(run_conviction(r, base + 0.1 * rng.standard_normal((N, K)), rng, N, K))
        dv[i] = np.mean(d); cl[i] = np.mean(c)
    return ratios, dv, cl, N


# =========================================================================
# B) PREFERENTIAL ATTACHMENT (sequential) -- size scaling
# =========================================================================
def run_pa(r, gamma, labels0, rng, N, sweeps=130, burn=80):
    labels = labels0.copy(); cnt = defaultdict(int)
    for l in labels: cnt[l] += 1
    nid = int(labels.max()) + 1; vals = []
    for t in range(sweeps):
        for _ in range(N):
            i = rng.integers(N); k = labels[i]; cnt[k] -= 1
            if cnt[k] == 0: del cnt[k]
            if rng.random() < r:
                lab = nid; nid += 1
            else:
                ts = np.fromiter(cnt.keys(), dtype=np.int64)
                w = np.fromiter((cnt[x] for x in ts), float) ** gamma
                lab = int(ts[rng.choice(len(ts), p=w / w.sum())])
            labels[i] = lab; cnt[lab] = cnt.get(lab, 0) + 1
        if t >= burn: vals.append(neff_counts(list(cnt.values()), N))
    return float(np.mean(vals))


def experiment_B():
    Ns = [50, 100, 200, 350]; rfix = 0.15
    out = {1.0: [], 1.5: []}
    for g in (1.0, 1.5):
        for N in Ns:
            vv = [run_pa(rfix, g, np.arange(N), np.random.default_rng(20 + s), N)
                  for s in range(2)]
            out[g].append(np.mean(vv))
    return Ns, out, rfix


# =========================================================================
# C) SWITCHING BARRIER (sequential) -- fold + critical B*
# =========================================================================
def run_barrier(r, B, labels0, rng, N, sweeps=800, burn=500, gamma=1.0):
    labels = labels0.copy(); cnt = defaultdict(int)
    for l in labels: cnt[l] += 1
    nid = int(labels.max()) + 1; vals = []
    for t in range(sweeps):
        for _ in range(N):
            i = rng.integers(N); k = labels[i]; nk = cnt[k]
            if rng.random() < B * (nk / N):
                continue
            cnt[k] -= 1
            if cnt[k] == 0: del cnt[k]
            if rng.random() < r:
                lab = nid; nid += 1
            else:
                ts = np.fromiter(cnt.keys(), dtype=np.int64)
                w = np.fromiter((cnt[x] for x in ts), float) ** gamma
                lab = int(ts[rng.choice(len(ts), p=w / w.sum())])
            labels[i] = lab; cnt[lab] = cnt.get(lab, 0) + 1
        if t >= burn: vals.append(neff_counts(list(cnt.values()), N))
    return float(np.mean(vals))


def barrier_branch(r, B, initfn, N):
    return np.mean([run_barrier(r, B, initfn(N), np.random.default_rng(40 + s), N)
                    for s in range(2)])


def experiment_C():
    N = 100
    dv = lambda n: np.arange(n)
    cl = lambda n: np.zeros(n, dtype=np.int64)
    # loop in r at strong barrier
    Bfix = 0.99
    rs = np.linspace(0.05, 0.55, 9)
    loop_dv = np.array([barrier_branch(r, Bfix, dv, N) for r in rs])
    loop_cl = np.array([barrier_branch(r, Bfix, cl, N) for r in rs])
    # scan barrier at fixed r
    rfix = 0.35
    Bs = np.array([0.80, 0.90, 0.95, 0.97, 0.99])
    scan_dv = np.array([barrier_branch(rfix, B, dv, N) for B in Bs])
    scan_cl = np.array([barrier_branch(rfix, B, cl, N) for B in Bs])
    return N, rs, loop_dv, loop_cl, Bfix, Bs, scan_dv, scan_cl, rfix


# =========================================================================
# Run all, plot 2x2, print verdicts
# =========================================================================
print("A) bounded conviction ..."); rA, dvA, clA, NA = experiment_A()
print("B) preferential attachment ..."); NsB, outB, rB = experiment_B()
print("C) switching barrier ..."); NC, rsC, ldv, lcl, Bfix, Bs, sdv, scl, rC = experiment_C()

plt.rcParams.update({"font.family": "monospace", "font.size": 8.5})
fig, ax = plt.subplots(2, 2, figsize=(11, 8))

a = ax[0, 0]
a.plot(rA, dvA, "o-", color="#2a9d8f", ms=3, label="diverse start")
a.plot(rA, clA, "s--", color="#e76f51", ms=3, label="collapsed start")
a.set_title("A. Bounded conviction -> NULL (branches coincide)")
a.set_xlabel("injection ratio  entry/depletion"); a.set_ylabel("N_eff")
a.legend(fontsize=7, loc="upper left")

b = ax[0, 1]
b.plot(NsB, outB[1.0], "o-", color="#2a9d8f", ms=4, label="gamma=1  (extensive ~N)")
b.plot(NsB, outB[1.5], "s-", color="#e76f51", ms=4, label="gamma=1.5 (intensive O(1))")
b.set_title(f"B. Preferential attachment -> condensation, not fold (r={rB})")
b.set_xlabel("system size  N"); b.set_ylabel("N_eff")
b.legend(fontsize=7, loc="upper left")

c = ax[1, 0]
c.plot(rsC, ldv, "o-", color="#2a9d8f", ms=4, label="diverse start")
c.plot(rsC, lcl, "s-", color="#e76f51", ms=4, label="collapsed start")
bist = rsC[(ldv > NC / 4) & (lcl < NC / 4)]
if bist.size:
    c.axvspan(bist.min(), bist.max(), color="0.85", alpha=.6, zorder=0)
    c.text((bist.min() + bist.max()) / 2, NC * 0.5, "hysteresis",
           ha="center", fontsize=8, color="0.3")
c.set_title(f"C1. Switching barrier -> FOLD (B={Bfix})")
c.set_xlabel("innovation rate  r"); c.set_ylabel("N_eff")
c.legend(fontsize=7, loc="upper left")

d = ax[1, 1]
d.plot(Bs, sdv, "o-", color="#2a9d8f", ms=4, label="diverse start")
d.plot(Bs, scl, "s-", color="#e76f51", ms=4, label="collapsed start")
opened = Bs[(sdv > NC / 4) & (scl < NC / 4)]
if opened.size:
    d.axvline(opened.min(), ls="--", c="0.4")
    d.text(opened.min(), NC * 0.55, f" B* ~ {opened.min():.2f}", fontsize=8, color="0.3")
d.set_title(f"C2. Fold requires near-absorbing lock-in (r={rC})")
d.set_xlabel("barrier strength  B"); d.set_ylabel("N_eff")
d.legend(fontsize=7, loc="center left")

fig.suptitle("Paper XVI §6: the protection class is real but requires near-foreclosed exit",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
png = os.path.join(OUT, "paper_xvi_protection_class.png")
fig.savefig(png, dpi=140)

print("\n--- verdicts ---")
print(f"A bounded conviction: max branch gap = {np.abs(dvA - clA).max():.1f}  -> single attractor (null)")
print(f"B PA: N_eff(gamma=1) grows {np.round(outB[1.0],1)}; N_eff(gamma=1.5) flat {np.round(outB[1.5],1)}")
print(f"C fold hysteresis window in r: "
      f"[{bist.min():.2f},{bist.max():.2f}]" if bist.size else "C: no window")
print(f"C critical barrier B*: {opened.min():.2f}" if opened.size else "C: no B*")
print("saved:", png)
