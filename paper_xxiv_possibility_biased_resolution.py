"""
Possibility collapse — the real Goodhart test: passive coupling, optimized break.

The previous run left the interesting case untested. A lossy feature-proxy had
reach=0 even at lambda=0 (never coupled -> nothing to break); a lossless state-proxy
stayed coupled under optimization (no Goodhart). Neither shows the thing the theory
claims: a metric that WORKS AS A SENSOR but FAILS AS AN OBJECTIVE.

This version uses a BIASED-RESOLUTION proxy: visitation entropy over a tiling that is
fine-grained in the meadow (every meadow cell is its own tile) and coarse over the far
region (the entire below-the-wall region collapses to ONE tile). This mirrors real
legibility instruments: high resolution over the central/legible region, near-zero
resolution over the weird periphery where genuine alternatives live.

Mechanism:
  - PASSIVE regime (lambda=0, exploration varied via epsilon): exploration effort is a
    common cause -> agents that wander more raise the proxy AND occasionally reach far
    options -> proxy correlates POSITIVELY with option reach. The sensor works.
  - OPTIMIZATION regime (arm B optimizes the biased proxy, lambda swept): the cheapest
    way to raise tile-entropy is to spread across the MANY meadow tiles; the single far
    tile is negligible and expensive -> proxy rises, reach does not. The objective fails.

Preregistered predictions:
  P-couple : across passive (epsilon-swept) agents, corr(proxy, reach) > 0.
  P-break  : across optimizing (lambda-swept) agents, proxy rises while reach stays at
             the floor -> corr <= 0 or undefined (no reach variance). Same proxy,
             coupled when observed, decoupled when optimized.
  P-struct : arm D (structural random-spawn, no proxy) attains full option reach.

Validity gate: arm B must keep solving the task (g1 ~ 1). Do NOT tune geometry toward
a desired correlation; epsilon range for the passive regime is chosen only so that
reach has variance (a testability requirement decided on the passive regime alone).
"""

import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr

MAP = [
    "###############",
    "#.............#",
    "#.............#",
    "#......1......#",
    "#.....S.......#",
    "#.............#",
    "#.............#",
    "#.............#",
    "#######.#######",   # bottleneck out of the meadow (col 7)
    "#.............#",
    "#.###########.#",
    "#.#gg.....gg#.#",
    "#..gg..2..gg#.#",   # col-2 door: the only way into the far region
    "#.#gg.....gg#.#",
    "#.###########.#",
    "#.............#",
    "###############",
]
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Grid:
    def __init__(self, ascii_map=MAP):
        self.grid = [list(r) for r in ascii_map]
        self.H, self.W = len(self.grid), len(self.grid[0])
        self.free, self.start, self.g1, self.g2, self.far_goals = [], None, None, None, []
        for r in range(self.H):
            for c in range(self.W):
                ch = self.grid[r][c]
                if ch != '#':
                    self.free.append((r, c))
                if ch == 'S': self.start = (r, c)
                elif ch == '1': self.g1 = (r, c)
                elif ch == '2': self.g2 = (r, c)
                elif ch == 'g': self.far_goals.append((r, c))
        self.far_goals.append(self.g2)
        self.far_region = [s for s in self.free if 11 <= s[0] <= 13 and 3 <= s[1] <= 11]
        self.free_idx = {s: i for i, s in enumerate(self.free)}
        self.nS, self.nA = len(self.free), 4

        # BIASED-RESOLUTION TILING: meadow (rows 1..7) fine; everything below = 1 tile.
        meadow = [s for s in self.free if s[0] <= 7]
        self.tile = np.empty(self.nS, dtype=int)
        tid = {}
        for s in meadow:
            tid[s] = len(tid)
        coarse = len(tid)                                   # single coarse far tile
        for s in self.free:
            self.tile[self.free_idx[s]] = tid.get(s, coarse)
        self.nTiles = coarse + 1
        assert self._connected(), "far goals unreachable — bad map"

    def _connected(self):
        seen, stack = {self.start}, [self.start]
        while stack:
            u = stack.pop()
            for a in range(4):
                v = self.step(u, a)
                if v not in seen:
                    seen.add(v); stack.append(v)
        return all(g in seen for g in self.far_goals)

    def passable(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W and self.grid[r][c] != '#'

    def step(self, s, a):
        dr, dc = ACTIONS[a]
        nr, nc = s[0] + dr, s[1] + dc
        return (nr, nc) if self.passable(nr, nc) else s


def train(env, arm, lam=0.0, eps=0.15, episodes=4000, max_steps=120,
          gamma=0.99, alpha=0.5, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))
    visits = np.zeros(env.nS)
    tile_visits = np.zeros(env.nTiles)
    goal = env.g1
    step_cost = -0.01
    for ep in range(episodes):
        lam_t = lam * max(0.0, 1.0 - ep / (0.7 * episodes))   # decay -> keeps g1~1
        s = env.free[rng.integers(env.nS)] if arm == "D" else env.start
        for _ in range(max_steps):
            si = env.free_idx[s]
            visits[si] += 1
            tile_visits[env.tile[si]] += 1
            a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[si]))
            ns = env.step(s, a)
            nsi = env.free_idx[ns]
            r = step_cost + (1.0 if ns == goal else 0.0)
            if arm == "B":                                     # optimize the biased proxy
                r += lam_t / np.sqrt(tile_visits[env.tile[nsi]] + 1.0)
            done = ns == goal
            Q[si, a] += alpha * (r + (0.0 if done else gamma * np.max(Q[nsi])) - Q[si, a])
            s = ns
            if done:
                break
    return dict(Q=Q, visits=visits, tile_visits=tile_visits)


def tile_entropy(tv):
    p = tv / tv.sum(); p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def far_goal_reach(env, visits, k=1):
    idx = [env.free_idx[g] for g in env.far_goals]
    return float(np.mean([visits[i] >= k for i in idx]))


def g1_success(env, Q, max_steps=200):
    s = env.start
    for _ in range(max_steps):
        s = env.step(s, int(np.argmax(Q[env.free_idx[s]])))
        if s == env.g1:
            return 1.0
    return 0.0


def collect(env, arm, lam, eps, seeds, episodes):
    out = []
    for seed in range(seeds):
        r = train(env, arm, lam=lam, eps=eps, episodes=episodes, seed=seed)
        out.append(dict(arm=arm, lam=lam, eps=eps, seed=seed,
                        proxy=tile_entropy(r["tile_visits"]),
                        reach=far_goal_reach(env, r["visits"]),
                        g1=g1_success(env, r["Q"])))
    return out


def corr(rows):
    P = [r["proxy"] for r in rows]; R = [r["reach"] for r in rows]
    if np.std(R) < 1e-9 or np.std(P) < 1e-9:
        return None, None
    return pearsonr(P, R)


def run(seeds=12, episodes=4000, smoke=False):
    env = Grid()
    print(f"tiling: {env.nTiles} tiles ({env.nTiles-1} meadow + 1 coarse far tile), "
          f"{len(env.far_goals)} far goals\n")

    eps_list = (0.3, 0.9) if smoke else (0.1, 0.3, 0.5, 0.7, 0.9)
    lam_list = (0.2, 0.8) if smoke else (0.05, 0.1, 0.2, 0.4, 0.8)

    print("PASSIVE regime (lambda=0, epsilon swept) — is the proxy a working sensor?")
    passive = []
    for e in eps_list:
        rows = collect(env, "A", 0.0, e, seeds, episodes)
        passive += rows
        m = lambda k: np.mean([r[k] for r in rows])
        print(f"  eps={e}  proxy={m('proxy'):.3f}  reach={m('reach'):.2f}  g1={m('g1'):.2f}")
    c, p = corr(passive)
    print(f"  --> corr(proxy, reach) across passive agents = "
          f"{'n/a' if c is None else f'{c:+.2f} (p={p:.3f})'}\n")

    print("OPTIMIZATION regime (arm B optimizes the proxy, lambda swept) — does it break?")
    optim = []
    for l in lam_list:
        rows = collect(env, "B", l, 0.15, seeds, episodes)
        optim += rows
        m = lambda k: np.mean([r[k] for r in rows])
        print(f"  lam={l:<4} proxy={m('proxy'):.3f}  reach={m('reach'):.2f}  g1={m('g1'):.2f}")
    c2, p2 = corr(optim)
    print(f"  --> corr(proxy, reach) across optimizing agents = "
          f"{'n/a (no reach variance)' if c2 is None else f'{c2:+.2f} (p={p2:.3f})'}\n")

    print("STRUCTURAL reference (arm D, random spawn, no proxy):")
    d = collect(env, "D", 0.0, 0.15, seeds, episodes)
    m = lambda k: np.mean([r[k] for r in d])
    print(f"  proxy={m('proxy'):.3f}  reach={m('reach'):.2f}  g1={m('g1'):.2f}")

    return env, passive, optim, d


def make_plot(passive, optim, d, path="./outputs/paper_xxiv_biased_resolution_results.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # left: passive coupling (sensor works)
    xs = [r["proxy"] for r in passive]; ys = [r["reach"] for r in passive]
    cs = [r["eps"] for r in passive]
    sc = ax[0].scatter(xs, ys, c=cs, cmap="viridis", s=28)
    if np.std(ys) > 1e-9:
        b, a0 = np.polyfit(xs, ys, 1)
        xr = np.linspace(min(xs), max(xs), 50)
        ax[0].plot(xr, a0 + b * xr, "k--", lw=1)
        c, _ = pearsonr(xs, ys)
        ax[0].set_title(f"Passive: proxy tracks option value (r={c:+.2f})")
    ax[0].set_xlabel("biased-resolution proxy (nats)")
    ax[0].set_ylabel("option reach (far goals)")
    fig.colorbar(sc, ax=ax[0], label="epsilon (exploration)")

    # right: optimization break (objective fails)
    lams = sorted({r["lam"] for r in optim})
    def agg(rows, l, k):
        v = [r[k] for r in rows if r["lam"] == l]
        return np.mean(v), np.std(v) / np.sqrt(len(v))
    P = [agg(optim, l, "proxy")[0] for l in lams]
    R = [agg(optim, l, "reach")[0] for l in lams]
    Rse = [agg(optim, l, "reach")[1] for l in lams]
    axr = ax[1].twinx()
    ax[1].plot(lams, P, "o-", color="#c0392b", label="proxy")
    axr.errorbar(lams, R, yerr=Rse, fmt="s--", color="#2471a3", label="option reach")
    dmean = np.mean([r["reach"] for r in d])
    axr.axhline(dmean, color="#27ae60", ls=":", label="arm D (structural)")
    ax[1].set_xlabel("lambda (proxy optimized, arm B)")
    ax[1].set_ylabel("biased-resolution proxy (nats)", color="#c0392b")
    axr.set_ylabel("option reach (far goals)", color="#2471a3")
    axr.set_ylim(-0.05, 1.05)
    ax[1].set_title("Optimized: proxy rises, option value does not")
    axr.legend(loc="center right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    return path


if __name__ == "__main__":
    import sys
    smoke = "--smoke" in sys.argv
    seeds, eps = (4, 800) if smoke else (12, 4000)
    env, passive, optim, d = run(seeds=seeds, episodes=eps, smoke=smoke)
    if not smoke:
        print(f"\nsaved plot -> {make_plot(passive, optim, d)}")
