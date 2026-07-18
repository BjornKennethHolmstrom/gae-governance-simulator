"""
Experiment 2 — resolution bias as the independent variable.

Experiment 1 showed a hand-drawn resolution-biased proxy decouples under optimization.
The obvious objection: you drew the tiling. This turns the tiling into the KNOB.

Peripheral resolution k_far = number of tiles the below-the-wall region is split into
(meadow stays fine, one tile per cell). rho = (peripheral cells / k_far) is the
resolution asymmetry. Traversal cost c_b = extra per-step penalty applied in the
peripheral region (a cost-to-reach-the-edge surrogate). Arm B optimizes the biased
occupancy proxy at fixed lambda; we measure zero-shot exercisable reach.

Central prediction (mine, sharper than a single-scalar phase transition):
The optimizer leaves the meadow only when discounted peripheral novelty beats meadow
residual novelty. Peripheral lure grows with k_far; the trek is discounted by c_b. So
decoupling is governed by k_far AND c_b jointly, and the falsifiable structural claim is:

    the peripheral resolution k_far* needed to restore exercisable reach rises
    monotonically with traversal cost c_b.

If instead reach were governed by rho alone, k_far* would not move with c_b.
"""

import numpy as np
from collections import defaultdict, deque

MAP = [
    "###############", "#.............#", "#.............#", "#......1......#",
    "#.....S.......#", "#.............#", "#.............#", "#.............#",
    "#######.#######", "#.............#", "#.###########.#", "#.#gg.....gg#.#",
    "#..gg..2..gg#.#", "#.#gg.....gg#.#", "#.###########.#", "#.............#",
    "###############",
]
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Grid:
    def __init__(self, k_far=1, m_meadow=None):
        self.grid = [list(r) for r in MAP]
        self.H, self.W = len(self.grid), len(self.grid[0])
        self.free, self.start, self.g1, self.g2, self.far_goals = [], None, None, None, []
        for r in range(self.H):
            for c in range(self.W):
                ch = self.grid[r][c]
                if ch != '#': self.free.append((r, c))
                if ch == 'S': self.start = (r, c)
                elif ch == '1': self.g1 = (r, c)
                elif ch == '2': self.g2 = (r, c)
                elif ch == 'g': self.far_goals.append((r, c))
        self.far_goals.append(self.g2)
        self.free_idx = {s: i for i, s in enumerate(self.free)}
        self.nS, self.nA = len(self.free), 4
        self.peripheral = np.array([1 if s[0] >= 8 else 0 for s in self.free])

        meadow = [s for s in self.free if s[0] <= 7]
        peri = [s for s in self.free if s[0] >= 8]
        m_meadow = len(meadow) if m_meadow is None else m_meadow   # default: one tile/cell
        tid = {}
        for grp, cells in enumerate(np.array_split(meadow, m_meadow)):
            for s in cells:
                tid[tuple(s)] = grp
        base = m_meadow
        for grp, cells in enumerate(np.array_split(peri, k_far)):
            for s in cells:
                tid[tuple(s)] = base + grp
        self.tile = np.array([tid[s] for s in self.free])
        self.nTiles = base + k_far
        self.k_far, self.n_peri, self.n_meadow = k_far, len(peri), len(meadow)
        assert self._connected()

    def _connected(self):
        seen, st = {self.start}, [self.start]
        while st:
            u = st.pop()
            for a in range(4):
                v = self.step(u, a)
                if v not in seen: seen.add(v); st.append(v)
        return all(g in seen for g in self.far_goals)

    def passable(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W and self.grid[r][c] != '#'

    def step(self, s, a):
        dr, dc = ACTIONS[a]
        nr, nc = s[0] + dr, s[1] + dc
        return (nr, nc) if self.passable(nr, nc) else s


def train_B(env, lam, c_b, eps=0.15, episodes=4000, max_steps=120,
            gamma=0.99, alpha=0.5, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))
    tile_visits = np.zeros(env.nTiles)
    model = {}
    goal = env.g1
    for ep in range(episodes):
        lam_t = lam * max(0.0, 1.0 - ep / (0.7 * episodes))
        s = env.start
        for _ in range(max_steps):
            si = env.free_idx[s]
            tile_visits[env.tile[si]] += 1
            a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[si]))
            ns = env.step(s, a); nsi = env.free_idx[ns]
            model[(si, a)] = nsi
            r = -0.01 - c_b * env.peripheral[nsi] + (1.0 if ns == goal else 0.0)
            r += lam_t / np.sqrt(tile_visits[env.tile[nsi]] + 1.0)
            done = ns == goal
            Q[si, a] += alpha * (r + (0.0 if done else gamma * np.max(Q[nsi])) - Q[si, a])
            s = ns
            if done: break
    return model


def exercisable_reach(env, model):
    adj = defaultdict(list)
    for (si, a), nsi in model.items():
        adj[si].append(nsi)
    start = env.free_idx[env.start]
    seen, q = {start}, deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen: seen.add(v); q.append(v)
    return float(np.mean([env.free_idx[g] in seen for g in env.far_goals]))


def cell(k_far, c_b, lam, seeds, episodes):
    env = Grid(k_far=k_far)
    return np.mean([exercisable_reach(env, train_B(env, lam, c_b, episodes=episodes, seed=s))
                    for s in range(seeds)])


def run(seeds=8, episodes=4000, smoke=False):
    if smoke:
        k_list, cb_list, lam_list = (1, 8, 65), (0.0, 0.05), (0.5,)
        seeds, episodes = 3, 800
    else:
        k_list = (1, 2, 4, 8, 16, 32, 65)
        cb_list = (0.0, 0.02, 0.05, 0.1)
        lam_list = (0.2, 0.5, 0.8)

    n_peri = Grid(1).n_peri
    print(f"peripheral cells = {n_peri}; k_far sweep = {k_list}\n")

    print("PANEL A — exercisable reach vs peripheral resolution k_far (c_b=0)")
    print("k_far:      " + "  ".join(f"{k:>4}" for k in k_list))
    A = {}
    for lam in lam_list:
        row = [cell(k, 0.0, lam, seeds, episodes) for k in k_list]
        A[lam] = row
        print(f"  lam={lam:<4} " + "  ".join(f"{v:4.2f}" for v in row))

    print("\nPANEL B — heatmap: exercisable reach over (k_far x c_b) at lam=0.5")
    print("            " + "  ".join(f"k={k:<4}" for k in k_list))
    H = {}
    for c_b in cb_list:
        row = [cell(k, c_b, 0.5, seeds, episodes) for k in k_list]
        H[c_b] = row
        print(f"  c_b={c_b:<5} " + "  ".join(f"{v:4.2f}" for v in row))

    # boundary: smallest k_far giving reach >= 0.5, per c_b
    print("\nBOUNDARY — smallest k_far with reach>=0.5, by traversal cost c_b:")
    for c_b in cb_list:
        row = H[c_b]
        kstar = next((k for k, v in zip(k_list, row) if v >= 0.5), None)
        print(f"  c_b={c_b:<5} k_far* = {kstar if kstar is not None else '>65 (no restore)'}")
    return k_list, cb_list, lam_list, A, H


def make_plot(k_list, cb_list, lam_list, A, H, path="./outputs/paper_xxiv_resolution_sweep.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    for lam in lam_list:
        ax[0].plot(k_list, A[lam], "o-", label=f"λ={lam}")
    ax[0].axhline(0.5, color="gray", ls=":", lw=1)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("peripheral resolution  k_far  (coarse → fine)")
    ax[0].set_ylabel("exercisable reach (arm B)")
    ax[0].set_title("Optimization decouples only under coarse periphery")
    ax[0].legend(); ax[0].set_ylim(-0.05, 1.05)

    M = np.array([H[c] for c in cb_list])
    im = ax[1].imshow(M, aspect="auto", origin="lower", cmap="viridis",
                      vmin=0, vmax=1)
    ax[1].set_xticks(range(len(k_list))); ax[1].set_xticklabels(k_list)
    ax[1].set_yticks(range(len(cb_list))); ax[1].set_yticklabels(cb_list)
    ax[1].set_xlabel("peripheral resolution  k_far")
    ax[1].set_ylabel("traversal cost  c_b")
    ax[1].set_title("Reach(k_far, c_b): boundary shifts right as cost rises")
    # overlay k_far* boundary (reach>=0.5)
    for i, c in enumerate(cb_list):
        row = H[c]
        js = [j for j, v in enumerate(row) if v >= 0.5]
        if js:
            ax[1].plot(min(js), i, "rx", markersize=11, markeredgewidth=2.5)
    fig.colorbar(im, ax=ax[1], label="exercisable reach")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    return path


if __name__ == "__main__":
    import sys
    smoke = "--smoke" in sys.argv
    k_list, cb_list, lam_list, A, H = run(smoke=smoke)
    if not smoke:
        print(f"\nsaved plot -> {make_plot(k_list, cb_list, lam_list, A, H)}")
