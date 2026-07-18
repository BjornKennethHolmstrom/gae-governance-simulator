"""
Experiment 3 — does a task-trained representation land in the decoupling regime?

Experiments 1-2 used HAND-DRAWN tilings. The open question: when a representation is
LEARNED from data, does it spontaneously allocate fine resolution to the frequently
visited centre and coarse resolution to the costly periphery — i.e. land in Experiment
2's decoupling regime on its own?

Faithful pure-numpy stand-in for a task-trained representation: visitation-weighted
k-means over states. It places centroids where data is dense (spends categories there)
and lets low-frequency peripheral cells collapse into the nearest centroid. The learned
cluster labels ARE a tiling; we read its effective peripheral resolution and then USE it
as the proxy (arm B optimizes it) to measure exercisable reach.

Data regimes:
  task_only   task-directed rollouts to G1 (meadow-heavy)   -> predict coarse periphery
  passive     high-epsilon broad exploration
  structural  random-spawn exposure (arm-D style)           -> predict finer periphery
  mixed       equal blend

Prediction that can FAIL:
  task_only learns few peripheral categories -> optimizing it gives reach ~ 0 (decouple);
  structural learns many -> reach restored. If task_only ALSO restores, the resolution-
  bias story does not arise endogenously and the toy's relevance collapses.
"""

import numpy as np
from paper_xxiv_possibility_resolution_sweep_v2 import Grid, train_B, reach


def collect_visits(env, regime, episodes=2000, max_steps=120, seed=0):
    """Behaviour-policy visitation counts under a regime (task Q learned online)."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA)); visits = np.zeros(env.nS)
    goal = env.g1
    eps = {"task_only": 0.15, "passive": 0.9, "structural": 0.15, "mixed": 0.4}[regime]
    for ep in range(episodes):
        s = env.free[rng.integers(env.nS)] if regime == "structural" else env.start
        if regime == "mixed" and ep % 2 == 0:
            s = env.free[rng.integers(env.nS)]
        for _ in range(max_steps):
            si = env.free_idx[s]; visits[si] += 1
            a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[si]))
            ns = env.step(s, a); nsi = env.free_idx[ns]
            r = -0.01 + (1.0 if ns == goal else 0.0)
            done = ns == goal
            Q[si, a] += 0.5 * (r + (0.0 if done else 0.99 * Q[nsi].max()) - Q[si, a])
            s = ns
            if done: break
    return visits


def wkmeans(pts, w, K, seed, iters=60):
    rng = np.random.default_rng(seed)
    w = w + 1e-9
    idx = [rng.choice(len(pts), p=w / w.sum())]                 # weighted k-means++
    for _ in range(K - 1):
        d2 = np.min(((pts[:, None] - pts[idx][None]) ** 2).sum(-1), axis=1)
        pr = d2 * w; pr = pr / pr.sum() if pr.sum() > 0 else None
        idx.append(rng.choice(len(pts), p=pr))
    C = pts[idx].astype(float)
    for _ in range(iters):
        lab = np.argmin(((pts[:, None] - C[None]) ** 2).sum(-1), axis=1)
        newC = C.copy()
        for k in range(K):
            m = lab == k
            if m.any(): newC[k] = (pts[m] * w[m, None]).sum(0) / w[m].sum()
        if np.allclose(newC, C): break
        C = newC
    return np.argmin(((pts[:, None] - C[None]) ** 2).sum(-1), axis=1)


def run(K=78, kmeans_seed=0, reach_seeds=6, regimes=("task_only", "passive", "structural", "mixed")):
    env = Grid(k_far=1, m_meadow=13)                            # base env; tiling overwritten
    pts = np.array([[r, c] for (r, c) in env.free], float)
    peri = env.peripheral.astype(bool)
    print(f"K={K} total categories; {peri.sum()} peripheral / {(~peri).sum()} meadow cells")
    print(f"(Exp-2 ruler: at c_b=0, restoration needs peripheral resolution ~40+)\n")
    rows = []
    for reg in regimes:
        visits = collect_visits(env, reg, seed=kmeans_seed)
        lab = wkmeans(pts, visits, K, kmeans_seed)
        eff_far = len(np.unique(lab[peri]))
        eff_mead = len(np.unique(lab[~peri]))
        pshare = visits[peri].sum() / visits.sum()
        # use the learned tiling as the proxy; measure exercisable reach when optimized
        env.tile = lab.astype(int); env.nTiles = K
        rc = [reach(env, train_B(env, lam=0.7, c_b=0.0, seed=s)[2]) for s in range(reach_seeds)]
        rows.append((reg, eff_mead, eff_far, pshare, np.mean(rc), np.std(rc)))
        print(f"  {reg:<11} periph_res={eff_far:>2}  meadow_res={eff_mead:>2}  "
              f"periph_visit_share={pshare:.3f}  reach={np.mean(rc):.2f}±{np.std(rc):.2f}")
    return env, rows


def make_plot(rows, K, path="./outputs/paper_xxiv_experiment3.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    regs = [r[0] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    efff = [r[2] for r in rows]
    ax[0].bar(regs, efff, color=["#c0392b", "#e67e22", "#27ae60", "#8e44ad"])
    ax[0].axhline(40, color="gray", ls="--", label="Exp-2 restoration threshold (~40)")
    ax[0].set_ylabel("learned peripheral resolution (distinct categories)")
    ax[0].set_title(f"How each regime allocates its {K} categories to the periphery")
    ax[0].legend(fontsize=8); ax[0].tick_params(axis="x", rotation=15)
    rea = [r[4] for r in rows]; er = [r[5] for r in rows]
    ax[1].bar(regs, rea, yerr=er, color=["#c0392b", "#e67e22", "#27ae60", "#8e44ad"])
    ax[1].set_ylabel("exercisable reach (optimizing the learned proxy)")
    ax[1].set_ylim(0, 1.05); ax[1].set_title("Does optimizing the learned proxy decouple?")
    ax[1].tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    return path


if __name__ == "__main__":
    import sys
    K = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 78
    env, rows = run(K=K)
    print(f"\nsaved plot -> {make_plot(rows, K)}")
