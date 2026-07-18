"""
Experiment 2 (corrected & self-contained) — resolution bias as the independent variable.

Fixes over the previous packaging:
  - m_meadow is an EXPLICIT parameter used everywhere (the interior-boundary regime is
    m_meadow=13, not the default one-tile-per-cell that produced the cramped result).
  - Every cell records the PROXY (tile entropy, what B optimizes), G1 success (validity
    gate), and exercisable reach — so the heatmap shows proxy->reach DECOUPLING, not
    reach alone.
  - Dense cost grid near zero and k_far near the transition (previous grid stepped over it).
  - Per-cell bootstrap CIs from per-seed values.
  - Partition-scheme check (row-major vs random) so k_far is a resolution knob, not a
    specific label assignment.

Run to fill the cache (resumable, time-boxed):   python3 thisfile.py <budget_seconds>
Then plot/tabulate:                              python3 thisfile.py --plot
"""

import json, os, sys, time
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
CACHE = "./outputs/paper_xxiv_sweep2_cache.json"


class Grid:
    def __init__(self, k_far, m_meadow, partition="rowmajor", pseed=0):
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
        if partition == "random":
            peri = list(np.random.default_rng(pseed).permutation(np.array(peri)))
            peri = [tuple(p) for p in peri]
        tid = {}
        for grp, cells in enumerate(np.array_split(meadow, m_meadow)):
            for s in cells: tid[tuple(s)] = grp
        for grp, cells in enumerate(np.array_split(peri, k_far)):
            for s in cells: tid[tuple(s)] = m_meadow + grp
        self.tile = np.array([tid[s] for s in self.free])
        self.nTiles = m_meadow + k_far
        self.n_peri, self.n_meadow = len(peri), len(meadow)
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
    return Q, tile_visits, model


def reach(env, model):
    adj = defaultdict(list)
    for (si, a), nsi in model.items(): adj[si].append(nsi)
    start = env.free_idx[env.start]
    seen, q = {start}, deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen: seen.add(v); q.append(v)
    return float(np.mean([env.free_idx[g] in seen for g in env.far_goals]))


def entropy(tv):
    p = tv / tv.sum(); p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def g1_ok(env, Q, max_steps=200):
    s = env.start
    for _ in range(max_steps):
        s = env.step(s, int(np.argmax(Q[env.free_idx[s]])))
        if s == env.g1: return 1.0
    return 0.0


def one(k_far, c_b, lam, m_meadow, partition, pseed, seed):
    env = Grid(k_far, m_meadow, partition, pseed)
    Q, tv, model = train_B(env, lam, c_b, seed=seed)
    return reach(env, model), entropy(tv), g1_ok(env, Q)


# --------------------------- sweep specification --------------------------- #
SEEDS = 6
MM = 13
CELLS = []
# main heatmap: lam=0.7, dense c_b, k_far near transition, row-major
for cb in (0.0, 0.005, 0.01, 0.015, 0.02):
    for k in (24, 32, 40, 48, 56, 65):
        CELLS.append((k, cb, 0.7, MM, "rowmajor", 0))
# lambda panel (with proxy+g1): explains why weak lambda never restores
for lam in (0.2, 0.5, 0.8):
    for k in (48, 65):
        CELLS.append((k, 0.0, lam, MM, "rowmajor", 0))
# partition robustness at c_b=0, lam=0.7
for part, ps in (("random", 1), ("random", 2)):
    for k in (48, 56, 65):
        CELLS.append((k, 0.0, 0.7, MM, part, ps))


def fill(budget):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    t0 = time.time()
    for (k, cb, lam, mm, part, ps) in CELLS:
        key = f"{k}|{cb}|{lam}|{mm}|{part}|{ps}"
        if key in cache and len(cache[key]["reach"]) >= SEEDS: continue
        if time.time() - t0 > budget: break
        R, P, G = [], [], []
        for s in range(SEEDS):
            r, p, g = one(k, cb, lam, mm, part, ps, s)
            R.append(r); P.append(p); G.append(g)
        cache[key] = dict(reach=R, proxy=P, g1=G)
        json.dump(cache, open(CACHE, "w"))
        print(f"{key}: reach={np.mean(R):.2f} proxy={np.mean(P):.2f} g1={np.mean(G):.2f} "
              f"({time.time()-t0:.0f}s)")
    print(f"cached {sum(1 for c in cache.values() if len(c['reach'])>=SEEDS)}/{len(CELLS)}")


def boot_ci(x, n=5000):
    rng = np.random.default_rng(0)
    m = [np.mean(rng.choice(x, len(x))) for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def report_and_plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cache = json.load(open(CACHE))
    g = lambda k, cb, lam, part="rowmajor", ps=0: cache.get(f"{k}|{cb}|{lam}|{MM}|{part}|{ps}")
    KM = (24, 32, 40, 48, 56, 65); CB = (0.0, 0.005, 0.01, 0.015, 0.02)

    print("MAIN GRID (lam=0.7, m_meadow=13) — reach [95% CI] | proxy | g1  (— = not cached)")
    M = np.full((len(CB), len(KM)), np.nan)
    for i, cb in enumerate(CB):
        cells = []
        for j, k in enumerate(KM):
            d = g(k, cb, 0.7)
            if d is None:
                cells.append(f"k{k}:—"); continue
            M[i, j] = np.mean(d["reach"])
            lo, hi = boot_ci(d["reach"])
            cells.append(f"k{k}:{M[i,j]:.2f}[{lo:.2f},{hi:.2f}]p{np.mean(d['proxy']):.1f}g{np.mean(d['g1']):.2f}")
        print(f" c_b={cb:<6}" + "  ".join(cells))
    for thr in (0.5, 0.9):
        print(f"\nBOUNDARY k_far* (reach>={thr}), by c_b:")
        for i, cb in enumerate(CB):
            ks = [KM[j] for j in range(len(KM)) if not np.isnan(M[i, j]) and M[i, j] >= thr]
            done = all(not np.isnan(M[i, j]) for j in range(len(KM)))
            print(f"  c_b={cb:<6} k_far* = {min(ks) if ks else ('>65' if done else 'incomplete')}")

    print("\nLAMBDA PANEL (c_b=0): proxy must inflate for reach to be possible")
    for lam in (0.2, 0.5, 0.8):
        for k in (48, 65):
            d = g(k, 0.0, lam)
            if d is None: print(f"  lam={lam} k={k}: — not cached"); continue
            print(f"  lam={lam} k={k}: reach={np.mean(d['reach']):.2f} "
                  f"proxy={np.mean(d['proxy']):.2f} g1={np.mean(d['g1']):.2f}")

    print("\nPARTITION ROBUSTNESS (c_b=0, lam=0.7): reach by tiling scheme")
    for k in (48, 56, 65):
        vals = []
        for part, ps in (("rowmajor", 0), ("random", 1), ("random", 2)):
            d = g(k, 0.0, 0.7, part, ps)
            vals.append(f"{part}{ps}:{np.mean(d['reach']):.2f}" if d else f"{part}{ps}:—")
        print(f"  k={k}: " + "  ".join(vals))

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for i, cb in enumerate(CB):
        ax[0].plot(KM, M[i], "o-", label=f"c_b={cb}")
    ax[0].axhline(0.5, color="gray", ls=":", lw=1)
    ax[0].set_xlabel("peripheral resolution k_far"); ax[0].set_ylabel("exercisable reach")
    ax[0].set_title("Reach vs resolution; curve shifts right as cost rises")
    ax[0].legend(fontsize=8); ax[0].set_ylim(-0.05, 1.05)
    im = ax[1].imshow(np.ma.masked_invalid(M), aspect="auto", origin="lower",
                      cmap="viridis", vmin=0, vmax=1)
    ax[1].set_xticks(range(len(KM))); ax[1].set_xticklabels(KM)
    ax[1].set_yticks(range(len(CB))); ax[1].set_yticklabels(CB)
    ax[1].set_xlabel("peripheral resolution k_far"); ax[1].set_ylabel("traversal cost c_b")
    ax[1].set_title("reach(k_far, c_b); ✕ = k_far* (reach≥0.5)")
    for i in range(len(CB)):
        js = [j for j in range(len(KM)) if M[i, j] >= 0.5]
        if js: ax[1].plot(min(js), i, "rx", ms=12, mew=3)
    fig.colorbar(im, ax=ax[1], label="exercisable reach")
    fig.tight_layout(); fig.savefig("./outputs/paper_xxiv_resolution_sweep_v2.png", dpi=130)
    print("\nsaved plot -> ./outputs/paper_xxiv_resolution_sweep_v2.png")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        report_and_plot()
    else:
        fill(float(sys.argv[1]) if len(sys.argv) > 1 else 180.0)
