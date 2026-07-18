"""
Possibility collapse v3 — exercisable option value, not mere discovery.

Fixes three faults in the biased-resolution run:
  (1) DISCOVERY != OPTION. Previously "reach" = a far cell visited >=1 once. Here the
      latent property is EXERCISABLE: using only the transition model learned in
      training, can the agent PLAN AND EXECUTE a route to the far goal FROM THE COMMON
      START? (Deterministic env + exact model => directed-graph reachability from start
      is equivalent to an executable path; noted as a scoping condition.)
  (2) ARM D TELEPORT LEAKAGE. Random-spawn used to place D inside the far region, so its
      score was exogenous. Now random-spawn only enriches D's MODEL; every arm is tested
      by planning/adapting FROM THE SAME START, so D must still traverse the bottleneck.
  (3) NO SHIFT. Added an OOD phase: move the goal to the far region and measure adaptation
      cost (real environment steps to first success from the common start, Dyna-planning
      with the retained model). This is "how cheaply is retained experience converted into
      competence when a formerly irrelevant possibility becomes valuable."

Proxy under test is unchanged: biased-resolution occupancy entropy (meadow fine-grained,
far region = one coarse tile).

Preregistered predictions:
  P-couple : across passive (epsilon-swept) agents, proxy predicts exercisable reach > 0.
  P-break  : arm B drives proxy up while exercisable reach stays ~0 (proxy coupled when
             observed, decoupled when optimized).
  P-struct : arm D (random-spawn model enrichment, no proxy) has high exercisable reach
             AND low OOD adaptation cost, tested from the common start.
  P-adapt  : OOD adaptation cost  D < A ~ B, despite B's far-higher proxy.

Reports pooled, condition-mean, and within-condition correlations (clustering is real).
"""

import numpy as np
from collections import defaultdict, deque
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
    "#######.#######",
    "#.............#",
    "#.###########.#",
    "#.#gg.....gg#.#",
    "#..gg..2..gg#.#",
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
                if ch != '#': self.free.append((r, c))
                if ch == 'S': self.start = (r, c)
                elif ch == '1': self.g1 = (r, c)
                elif ch == '2': self.g2 = (r, c)
                elif ch == 'g': self.far_goals.append((r, c))
        self.far_goals.append(self.g2)
        self.free_idx = {s: i for i, s in enumerate(self.free)}
        self.nS, self.nA = len(self.free), 4
        meadow = [s for s in self.free if s[0] <= 7]
        self.tile = np.empty(self.nS, dtype=int)
        tid = {s: i for i, s in enumerate(meadow)}
        coarse = len(tid)
        for s in self.free:
            self.tile[self.free_idx[s]] = tid.get(s, coarse)
        self.nTiles = coarse + 1
        assert self._connected(), "far goals unreachable — bad map"

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


def train(env, arm, lam=0.0, eps=0.15, episodes=4000, max_steps=120,
          gamma=0.99, alpha=0.5, seed=0):
    """Phase 1: learn task Q for G1 AND a goal-agnostic transition model."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))
    tile_visits = np.zeros(env.nTiles)
    model = {}                                            # (si,a) -> nsi  (exact, det env)
    goal = env.g1
    for ep in range(episodes):
        lam_t = lam * max(0.0, 1.0 - ep / (0.7 * episodes))
        s = env.free[rng.integers(env.nS)] if arm == "D" else env.start
        for _ in range(max_steps):
            si = env.free_idx[s]
            tile_visits[env.tile[si]] += 1
            a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[si]))
            ns = env.step(s, a); nsi = env.free_idx[ns]
            model[(si, a)] = nsi
            r = -0.01 + (1.0 if ns == goal else 0.0)
            if arm == "B":
                r += lam_t / np.sqrt(tile_visits[env.tile[nsi]] + 1.0)
            done = ns == goal
            Q[si, a] += alpha * (r + (0.0 if done else gamma * np.max(Q[nsi])) - Q[si, a])
            s = ns
            if done: break
    return dict(Q=Q, tile_visits=tile_visits, model=model)


def tile_entropy(tv):
    p = tv / tv.sum(); p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def exercisable_reach(env, model):
    """0-shot: fraction of far goals for which the LEARNED model admits a directed path
    from the COMMON START (executable in the real deterministic env)."""
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


def g1_success(env, Q, max_steps=200):
    s = env.start
    for _ in range(max_steps):
        s = env.step(s, int(np.argmax(Q[env.free_idx[s]])))
        if s == env.g1: return 1.0
    return 0.0


def adaptation_cost(env, model0, seed=0, budget=20000, gamma=0.99, alpha=0.5,
                    eps=0.2, planning=10):
    """OOD phase: goal -> G2. From the COMMON START, Dyna-Q warm-started with the retained
    model. Return real-env steps to first success (capped at budget). Agents whose retained
    model already spans the corridor propagate goal value by planning and solve fast."""
    rng = np.random.default_rng(seed + 777)
    goal = env.free_idx[env.g2]
    Q = np.zeros((env.nS, env.nA))
    model = dict(model0)                                  # retained transitions
    seen_sa = list(model.keys())
    s = env.start; steps = 0
    while steps < budget:
        si = env.free_idx[s]
        a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[si]))
        ns = env.step(s, a); nsi = env.free_idx[ns]; steps += 1
        model[(si, a)] = nsi
        if (si, a) not in seen_sa: seen_sa.append((si, a))
        r = 1.0 if nsi == goal else -0.01
        Q[si, a] += alpha * (r + (0.0 if nsi == goal else gamma * np.max(Q[nsi])) - Q[si, a])
        for _ in range(planning):                         # planning backups on retained model
            psi, pa = seen_sa[rng.integers(len(seen_sa))]
            pnsi = model[(psi, pa)]
            pr = 1.0 if pnsi == goal else -0.01
            Q[psi, pa] += alpha * (pr + (0.0 if pnsi == goal else gamma * np.max(Q[pnsi]))
                                   - Q[psi, pa])
        if nsi == goal:
            return steps
        s = ns if nsi != goal else env.start
    return budget


def collect(env, arm, lam, eps, seeds, episodes, do_adapt=False):
    out = []
    for seed in range(seeds):
        r = train(env, arm, lam=lam, eps=eps, episodes=episodes, seed=seed)
        row = dict(arm=arm, lam=lam, eps=eps, seed=seed,
                   proxy=tile_entropy(r["tile_visits"]),
                   exreach=exercisable_reach(env, r["model"]),
                   g1=g1_success(env, r["Q"]))
        if do_adapt:
            row["adapt"] = adaptation_cost(env, r["model"], seed=seed)
        out.append(row)
    return out


def report_corr(rows, label):
    P = np.array([r["proxy"] for r in rows]); R = np.array([r["exreach"] for r in rows])
    # pooled
    pooled = pearsonr(P, R) if np.std(R) > 1e-9 else (None, None)
    # condition means (by eps)
    epss = sorted({r["eps"] for r in rows})
    mp = [np.mean([r["proxy"] for r in rows if r["eps"] == e]) for e in epss]
    mr = [np.mean([r["exreach"] for r in rows if r["eps"] == e]) for e in epss]
    cm = pearsonr(mp, mr) if np.std(mr) > 1e-9 else (None, None)
    # within-condition (pooled over eps where variance exists)
    within = []
    for e in epss:
        g = [r for r in rows if r["eps"] == e]
        pe = [r["proxy"] for r in g]; re = [r["exreach"] for r in g]
        if np.std(re) > 1e-9 and np.std(pe) > 1e-9:
            within.append(pearsonr(pe, re)[0])
    fmt = lambda t: "n/a" if t[0] is None else f"{t[0]:+.2f}"
    print(f"  {label}: pooled r={fmt(pooled)} (n={len(rows)}); "
          f"condition-mean r={fmt(cm)} (n={len(epss)}); "
          f"within-eps mean r={'n/a' if not within else f'{np.mean(within):+.2f}'}")


def run(seeds=12, episodes=4000, smoke=False):
    env = Grid()
    print(f"tiling {env.nTiles} tiles; {len(env.far_goals)} far goals; "
          f"latent property = EXERCISABLE reach from common start\n")
    eps_list = (0.3, 0.9) if smoke else (0.1, 0.3, 0.5, 0.7, 0.9)
    lam_list = (0.2, 0.8) if smoke else (0.05, 0.1, 0.2, 0.4, 0.8)

    print("PASSIVE (lambda=0, epsilon swept) — proxy vs EXERCISABLE reach")
    passive = []
    for e in eps_list:
        rows = collect(env, "A", 0.0, e, seeds, episodes)
        passive += rows
        m = lambda k: np.mean([r[k] for r in rows])
        print(f"  eps={e}  proxy={m('proxy'):.3f}  exreach={m('exreach'):.2f}  g1={m('g1'):.2f}")
    report_corr(passive, "coupling")

    print("\nOPTIMIZATION (arm B optimizes proxy) — does exercisable reach follow?")
    optim = []
    for l in lam_list:
        rows = collect(env, "B", l, 0.15, seeds, episodes, do_adapt=True)
        optim += rows
        m = lambda k: np.mean([r[k] for r in rows])
        print(f"  lam={l:<4} proxy={m('proxy'):.3f}  exreach={m('exreach'):.2f}  "
              f"adapt={m('adapt'):.0f}  g1={m('g1'):.2f}")

    print("\nBASELINE A (task only) and STRUCTURAL D (random-spawn model, common-start test)")
    A = collect(env, "A", 0.0, 0.15, seeds, episodes, do_adapt=True)
    D = collect(env, "D", 0.0, 0.15, seeds, episodes, do_adapt=True)
    for nm, g in (("A", A), ("D", D)):
        m = lambda k: np.mean([r[k] for r in g])
        print(f"  {nm}: proxy={m('proxy'):.3f}  exreach={m('exreach'):.2f}  "
              f"adapt={m('adapt'):.0f}  g1={m('g1'):.2f}")
    return env, passive, optim, A, D


def make_plot(passive, optim, A, D, path="./outputs/paper_xxiv_exercisable_results.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    xs = [r["proxy"] for r in passive]; ys = [r["exreach"] for r in passive]
    cs = [r["eps"] for r in passive]
    sc = ax[0].scatter(xs, ys, c=cs, cmap="viridis", s=26)
    if np.std(ys) > 1e-9:
        b, a0 = np.polyfit(xs, ys, 1); xr = np.linspace(min(xs), max(xs), 40)
        ax[0].plot(xr, a0 + b * xr, "k--", lw=1)
        ax[0].set_title(f"Passive: proxy vs exercisable reach (r={pearsonr(xs, ys)[0]:+.2f})")
    ax[0].set_xlabel("biased-resolution proxy (nats)"); ax[0].set_ylabel("exercisable reach")
    fig.colorbar(sc, ax=ax[0], label="epsilon")

    lams = sorted({r["lam"] for r in optim})
    ag = lambda rows, l, k: (np.mean([r[k] for r in rows if r["lam"] == l]),
                             np.std([r[k] for r in rows if r["lam"] == l]) /
                             np.sqrt(sum(r["lam"] == l for r in rows)))
    P = [ag(optim, l, "proxy")[0] for l in lams]
    R = [ag(optim, l, "exreach")[0] for l in lams]
    Rse = [ag(optim, l, "exreach")[1] for l in lams]
    axr = ax[1].twinx()
    ax[1].plot(lams, P, "o-", color="#c0392b")
    axr.errorbar(lams, R, yerr=Rse, fmt="s--", color="#2471a3")
    axr.axhline(np.mean([r["exreach"] for r in D]), color="#27ae60", ls=":", label="arm D")
    ax[1].set_xlabel("lambda (proxy optimized)")
    ax[1].set_ylabel("proxy (nats)", color="#c0392b")
    axr.set_ylabel("exercisable reach", color="#2471a3"); axr.set_ylim(-0.05, 1.05)
    ax[1].set_title("Optimized: proxy up, exercisable reach flat"); axr.legend(fontsize=8)

    names = ["A task", f"B λ={lams[-1]}", "D struct"]
    grps = [A, [r for r in optim if r["lam"] == lams[-1]], D]
    means = [np.mean([r["adapt"] for r in g]) for g in grps]
    errs = [np.std([r["adapt"] for r in g]) / np.sqrt(len(g)) for g in grps]
    ax[2].bar(names, means, yerr=errs, color=["#7f8c8d", "#c0392b", "#27ae60"])
    ax[2].set_ylabel("OOD adaptation cost (env steps)")
    ax[2].set_title("Adaptation to far goal from common start")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    return path


if __name__ == "__main__":
    import sys
    smoke = "--smoke" in sys.argv
    seeds, eps = (4, 800) if smoke else (12, 4000)
    env, passive, optim, A, D = run(seeds=seeds, episodes=eps, smoke=smoke)
    if not smoke:
        print(f"\nsaved plot -> {make_plot(passive, optim, A, D)}")
