"""
self_ii_appendix_d_two_sided_bound.py

Reproduces the numerical claims in Self II, Appendix D
(Adaptive Learning and the Two-Sided Bound): the lower bound (persistence
of excitation), exploration-starvation self-concealment, the coherence
coupling and two-sided bound, self-vs-institution, and protected spaces.

Flat repo: gae-governance-simulator/
Run: python self_ii_appendix_d_two_sided_bound.py
"""

import numpy as np


def run(r, kappa, f_sandbox=0.0, v=0.02, sigma=0.30, iota=0.10, c=8.0, T=4000, seed=0):
    """Self as a dual controller tracking a drifting target (D.1-D.4, D.7).

    r          revision (exploration) rate
    kappa      coherence cost per unit revision (>0 self; =0 institution) -- the [IP] premise
    f_sandbox  fraction of revision insulated from global coherence (a protected space, D.7)
    Returns    (mean true tracking error, mean coherence, mean perceived error)
    """
    rng = np.random.default_rng(seed)
    th_star = th_hat = 0.0
    C = 1.0
    Es, Cs, Ehat = [], [], []
    for t in range(T):
        th_star += v * rng.standard_normal()              # environment drifts
        obs = th_star + sigma * rng.standard_normal()      # noisy self-observation
        dth = r * C * (obs - th_hat)                       # revision gated by coherence
        th_hat += dth
        C = np.clip(C + iota * (1 - C) - kappa * abs(dth) * (1 - f_sandbox), 0.0, 1.0)
        if t > T // 5:                                     # discard burn-in
            e = abs(th_hat - th_star)                      # TRUE tracking error
            Es.append(e)
            Cs.append(C)
            Ehat.append(e * (1 - np.exp(-c * r)))          # PERCEIVED error (seen only via exploration)
    return np.mean(Es), np.mean(Cs), np.mean(Ehat)


def sweep(kappa, rs, seeds=8):
    return np.array([np.mean([run(r, kappa, seed=s) for s in range(seeds)], axis=0) for r in rs])


if __name__ == "__main__":
    rs = [0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30, 0.60]

    print("D.4/D.5  Two-sided bound: optimal revision rate, self vs institution")
    for kappa, lab in [(0.0, "institution"), (0.5, "self       ")]:
        R = sweep(kappa, rs)
        E, C = R[:, 0], R[:, 1]
        for lam in (0.5, 1.0, 2.0):
            J = E + lam * (1 - C)
            k = int(np.argmin(J))
            interior = 0 < k < len(rs) - 1
            print(f"   {lab} lam={lam}: r*={rs[k]:.3f}  interior={interior}  "
                  f"E@r*={E[k]:.4f}  best_E={E.min():.4f}  sacrifice={E[k]-E.min():.4f}")

    g = np.mean([run(0.02, 0.5, seed=s) for s in range(8)], axis=0)
    print(f"D.3  self-concealment gap at r=0.02: true E={g[0]:.4f}, perceived={g[2]:.4f}, gap={g[0]-g[2]:.4f}")

    print("D.7  protected space (r=0.15, vary sandbox fraction f): E ~ const, C recovers")
    for f in (0.0, 0.3, 0.6, 0.9):
        E, C, _ = np.mean([run(0.15, 0.5, f_sandbox=f, seed=s) for s in range(8)], axis=0)
        print(f"   f={f}: E={E:.4f}  C={C:.3f}  J={E + (1 - C):.4f}")
