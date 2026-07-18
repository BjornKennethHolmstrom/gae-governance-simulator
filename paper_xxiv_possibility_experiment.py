"""
Possibility collapse under optimization pressure — a discriminating probe.

Central claim under test (the sharpened "possibility theorem"):
    A metric of possibility, once made ACTION-SELECTIVE (used as an objective),
    preferentially preserves the possibilities that are legible to it, and its
    correlation with the latent property it was meant to proxy BREAKS under
    optimization pressure.

This is the Goodhart mechanism, NOT the Heisenberg/observer mechanism. A logging
observer in a simulation changes nothing, so the observation-collapse claim from
the theory is not testable here; only the optimization claim is. (See notes.)

Design:
  - Gridworld with a large "meadow" (cheap-to-visit diversity garden) holding the
    TRAINING goal G1, a costly single-cell bottleneck, and a small far region
    holding the OOD goal G2. The training task NEVER requires the bottleneck.
  - Agent: tabular Q-learning for the task + count-based transition model learned
    from whatever the behaviour policy experiences.
  - Legible proxy (what arm B optimizes): visitation entropy.
  - Latent property (the "truth"): OPTION REACH — fraction of far/held-out goals
    reachable from start using only transitions the agent has actually experienced.
    A live option = an experienced path exists. A token option = does not.

Arms:
  A  task reward only                         (baseline)
  B  task reward + lambda * count-based novelty (metric-coupled; lambda swept)
  D  task reward only, random spawn each episode (STRUCTURAL preservation, no proxy)

Preregistered predictions:
  P1 proxy inflation:      entropy_B(large lambda) > entropy_A, entropy_D
  P2 no free lunch:        option_reach_B does NOT rise with entropy_B
  P4 structural advantage: option_reach_D > option_reach_A >= option_reach_B(large lambda)
  P3 correlation break:    corr(entropy, option_reach) across seeds is ~>=0 at
                           lambda=0 but goes <=0 as lambda grows.

NOTE ON HONESTY: the smoke test at the bottom is only to confirm the code runs and
produces sane numbers. Do not tune geometry until the hypothesis "works" — that is
p-hacking the probe. Run the full config once, report whatever comes out.
"""

import numpy as np
from collections import defaultdict

# ----------------------------------------------------------------------------- #
#  Environment
# ----------------------------------------------------------------------------- #
# Legend: '#' wall, '.' free, 'S' start, '1' G1 (train goal), '2' G2 (OOD goal),
#         'g' extra held-out far goals (part of the latent option set)
MAP = [
    "###############",
    "#.............#",
    "#.............#",
    "#......1......#",   # G1 sits IN the meadow -> task never needs the bottleneck
    "#.....S.......#",
    "#.............#",
    "#.............#",
    "#.............#",
    "#######.#######",   # solid wall, single gap col7 = bottleneck out of the meadow
    "#.............#",   # open corridor row (reaches the col-1 side corridor)
    "#.###########.#",
    "#.#gg.....gg#.#",   # far region interior (cols 3..11)
    "#..gg..2..gg#.#",   # col2 door -> the ONLY way in, via the long col-1 detour
    "#.#gg.....gg#.#",
    "#.###########.#",
    "#.............#",
    "###############",
]

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U D L R


class Grid:
    def __init__(self, ascii_map=MAP):
        self.grid = [list(r) for r in ascii_map]
        self.H, self.W = len(self.grid), len(self.grid[0])
        self.free, self.start, self.g1, self.g2, self.far_goals = [], None, None, None, []
        self.meadow = set()  # cells above the dividing wall
        for r in range(self.H):
            for c in range(self.W):
                ch = self.grid[r][c]
                if ch != '#':
                    self.free.append((r, c))
                    if r < 8:
                        self.meadow.add((r, c))
                if ch == 'S':
                    self.start = (r, c)
                elif ch == '1':
                    self.g1 = (r, c)
                elif ch == '2':
                    self.g2 = (r, c)
                elif ch == 'g':
                    self.far_goals.append((r, c))
        self.far_goals.append(self.g2)                  # held-out latent option set
        self.far_region = [s for s in self.free                # interior only
                           if 11 <= s[0] <= 13 and 3 <= s[1] <= 11]
        self.free_idx = {s: i for i, s in enumerate(self.free)}
        self.nS, self.nA = len(self.free), 4
        # LOSSY FEATURE LABELS: F feature values, all of which already occur in the
        # meadow. The far region carries the SAME features (adds no new feature).
        # So an agent maximizing FEATURE novelty can saturate the proxy in the meadow
        # and has no proxy incentive to pay the corridor cost. Feature-entropy is thus
        # a legible projection of possibility that discards the spatial reach dimension.
        self.F = 6
        self.feat = np.array([i % self.F for i in range(self.nS)])
        assert self._connected(), "far goals not reachable in the true env — bad map"

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
        return (nr, nc) if self.passable(nr, nc) else s   # walls block, stay put


# ----------------------------------------------------------------------------- #
#  Agent / training
# ----------------------------------------------------------------------------- #
def train(env, arm, lam=0.0, episodes=4000, max_steps=120,
          gamma=0.99, alpha=0.5, eps=0.15, seed=0, proxy_mode="feature"):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))
    total_visits = np.zeros(env.nS)                     # per-state visits
    feat_visits = np.zeros(env.F)                       # per-feature visits (lossy)
    exp_edges = {}                                      # (s_idx, a) -> next_s_idx
    goal = env.g1
    step_cost = -0.01

    for ep in range(episodes):
        # lambda decays to 0: strong intrinsic drive early, task exploitation late,
        # so arm B stays a competent task-solver (validity gate) yet its CUMULATIVE
        # visitation still reflects where the bonus sent it.
        lam_t = lam * max(0.0, 1.0 - ep / (0.7 * episodes))
        s = env.free[rng.integers(env.nS)] if arm == "D" else env.start
        for _ in range(max_steps):
            si = env.free_idx[s]
            total_visits[si] += 1
            feat_visits[env.feat[si]] += 1
            a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[si]))
            ns = env.step(s, a)
            nsi = env.free_idx[ns]
            exp_edges[(si, a)] = nsi

            r = step_cost + (1.0 if ns == goal else 0.0)
            if arm == "B":                              # count-based novelty bonus
                if proxy_mode == "feature":             # LOSSY proxy (gameable)
                    r += lam_t / np.sqrt(feat_visits[env.feat[nsi]] + 1.0)
                else:                                   # STATE proxy (~reachability)
                    r += lam_t / np.sqrt(total_visits[nsi] + 1.0)
            done = ns == goal
            target = r + (0.0 if done else gamma * np.max(Q[nsi]))
            Q[si, a] += alpha * (target - Q[si, a])
            s = ns
            if done:
                break

    return dict(Q=Q, visits=total_visits, feat_visits=feat_visits, edges=exp_edges)


# ----------------------------------------------------------------------------- #
#  Metrics
# ----------------------------------------------------------------------------- #
def visitation_entropy(visits):
    p = visits / visits.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())               # nats


def feature_entropy(feat_visits):
    p = feat_visits / feat_visits.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())               # the legible, lossy proxy


def coverage(visits, k=1):
    return float((visits >= k).mean())


def far_goal_reach(env, visits, k=1):
    """PRIMARY latent property: fraction of held-out far goals the agent actually
    reached (visited >= k) during training. A goal you never reached cannot be a
    live option. This is exactly what meadow-novelty gaming should starve."""
    idx = [env.free_idx[g] for g in env.far_goals]
    return float(np.mean([visits[i] >= k for i in idx]))


def far_region_coverage(env, visits, k=1):
    """Secondary latent property: coverage of the whole enclosed far region."""
    idx = [env.free_idx[s] for s in env.far_region]
    return float(np.mean([visits[i] >= k for i in idx]))


def plan_reach(env, edges):
    """Tertiary: fraction of far goals for which a path from start exists using the
    LEARNED model (any experienced (s,a)->s' is a usable transition)."""
    adj = defaultdict(set)
    for (si, a), nsi in edges.items():
        adj[si].add(nsi)
    start = env.free_idx[env.start]
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); stack.append(v)
    return float(np.mean([env.free_idx[g] in seen for g in env.far_goals]))


def g1_success(env, Q, max_steps=120):
    s, si = env.start, env.free_idx[env.start]
    for _ in range(max_steps):
        a = int(np.argmax(Q[env.free_idx[s]]))
        s = env.step(s, a)
        if s == env.g1:
            return 1.0
    return 0.0


# ----------------------------------------------------------------------------- #
#  Experiment
# ----------------------------------------------------------------------------- #
def run(seeds=12, lambdas=(0.0, 0.05, 0.1, 0.2, 0.4, 0.8), episodes=4000,
        proxy_mode="feature", verbose=True):
    env = Grid()
    rows = []
    configs = [("A", 0.0)] + [("B", l) for l in lambdas if l > 0] + [("D", 0.0)]
    for arm, lam in configs:
        for seed in range(seeds):
            out = train(env, arm, lam=lam, episodes=episodes, seed=seed,
                        proxy_mode=proxy_mode)
            proxy = (feature_entropy(out["feat_visits"]) if proxy_mode == "feature"
                     else visitation_entropy(out["visits"]))
            rows.append(dict(
                arm=arm, lam=lam, seed=seed,
                entropy=proxy,                                     # optimized proxy
                state_H=visitation_entropy(out["visits"]),
                option_reach=far_goal_reach(env, out["visits"]),   # latent truth
                far_cov=far_region_coverage(env, out["visits"]),
                plan_reach=plan_reach(env, out["edges"]),
                g1=g1_success(env, out["Q"]),                      # validity gate
            ))
        if verbose:
            sub = [r for r in rows if r["arm"] == arm and r["lam"] == lam]
            m = lambda k: np.mean([r[k] for r in sub])
            print(f"arm {arm} lam={lam:<4}  H={m('entropy'):.3f}  "
                  f"reach={m('option_reach'):.2f}  far_cov={m('far_cov'):.2f}  "
                  f"plan={m('plan_reach'):.2f}  g1={m('g1'):.2f}")
    return env, rows


def correlation_break(rows):
    """corr(entropy, option_reach) across seeds, within each lambda of arm B (+A at 0)."""
    from scipy.stats import pearsonr
    print("\ncorr(entropy, option_reach) across seeds, by lambda:")
    pts = [r for r in rows if r["arm"] in ("A", "B")]
    for lam in sorted({r["lam"] for r in pts}):
        g = [r for r in pts if r["lam"] == lam]
        H = [r["entropy"] for r in g]
        R = [r["option_reach"] for r in g]
        if np.std(R) < 1e-9 or np.std(H) < 1e-9:
            print(f"  lambda={lam:<4}  corr=n/a (no variance)  meanR={np.mean(R):.2f}")
            continue
        c, p = pearsonr(H, R)
        print(f"  lambda={lam:<4}  corr={c:+.2f}  (p={p:.2f})  "
              f"meanH={np.mean(H):.2f} meanR={np.mean(R):.2f}")


def make_plots(rows, path="./outputs/paper_xxiv_possibility_results.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lams = sorted({r["lam"] for r in rows if r["arm"] == "B"})
    def agg(arm, lam, key):
        v = [r[key] for r in rows if r["arm"] == arm and r["lam"] == lam]
        return np.mean(v), np.std(v) / np.sqrt(len(v))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    # left: proxy vs truth along the B lambda-sweep
    xs = [0.0] + lams
    Hs = [agg("A", 0.0, "entropy")[0]] + [agg("B", l, "entropy")[0] for l in lams]
    Rs = [agg("A", 0.0, "option_reach")[0]] + [agg("B", l, "option_reach")[0] for l in lams]
    Rse = [agg("A", 0.0, "option_reach")[1]] + [agg("B", l, "option_reach")[1] for l in lams]
    axr = ax[0].twinx()
    ax[0].plot(xs, Hs, "o-", color="#c0392b", label="diversity proxy")
    axr.errorbar(xs, Rs, yerr=Rse, fmt="s--", color="#2471a3", label="option reach (truth)")
    ax[0].set_xlabel("lambda (novelty reward weight, arm B)")
    ax[0].set_ylabel("diversity proxy (nats)", color="#c0392b")
    axr.set_ylabel("option reach (far goals)", color="#2471a3")
    axr.set_ylim(-0.05, 1.05)
    ax[0].set_title("Proxy rises, truth does not (Goodhart)")

    # right: arm comparison on the latent property
    arms = [("A", 0.0, "A task-only"),
            ("B", lams[-1], f"B lambda={lams[-1]}"),
            ("D", 0.0, "D structural")]
    names = [a[2] for a in arms]
    means = [agg(a[0], a[1], "option_reach")[0] for a in arms]
    errs = [agg(a[0], a[1], "option_reach")[1] for a in arms]
    ax[1].bar(names, means, yerr=errs, color=["#7f8c8d", "#c0392b", "#27ae60"])
    ax[1].set_ylabel("option reach (far goals)")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Latent option value by arm")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path


if __name__ == "__main__":
    import sys
    smoke = "--smoke" in sys.argv
    seeds, eps = (4, 800) if smoke else (12, 4000)
    lams = (0.0, 0.2, 0.8) if smoke else (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)

    print("=" * 70)
    print("MODE 1: LOSSY proxy (feature-novelty) — the metric-capture test")
    print("=" * 70)
    env, rows_f = run(seeds=seeds, lambdas=lams, episodes=eps, proxy_mode="feature")
    correlation_break(rows_f)

    print("\n" + "=" * 70)
    print("MODE 2: LOSSLESS proxy (state-novelty) — control; proxy ~ reachability")
    print("=" * 70)
    env, rows_s = run(seeds=seeds, lambdas=lams, episodes=eps, proxy_mode="state")
    correlation_break(rows_s)

    if not smoke:
        p = make_plots(rows_f, "./outputs/paper_xxiv_possibility_results.png")
        print(f"\nsaved plot -> {p}")
