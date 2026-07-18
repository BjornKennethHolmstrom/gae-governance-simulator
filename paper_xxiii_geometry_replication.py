#!/usr/bin/env python3
"""
paper_xxiii_geometry_replication.py
===================================
Registered replication of the geometry/topology findings inherited from Paper XIX,
which held them as illustrative.  Trains N_ZOOS independent zoos and runs three
preregistered tests, each with a stated null.

  TEST 1  Scale vs shape.
          Does stress *reshape* factorization space, or merely *rescale* it?
          Distance matrices are normalized by their mean off-diagonal entry, so
          a pure global rescaling becomes the identity.  A split-half noise
          ceiling is computed within each regime, giving the correlation two
          halves of the SAME regime achieve -- the highest any comparison could
          reach.  Between-regime correlation is then read against that ceiling.

          H_warp : median between-regime normalized correlation falls below the
                   within-regime ceiling by >= WARP_MARGIN, in >= PASS_BAR zoos.
          NULL   : it does not -- the environment sets the SCALE of factorization
                   space, not its SHAPE, and Paper XIX's headline motivation
                   does not survive replication.

  TEST 2  Bridge identity.
          Is the bridge a property of the model, or of the regime?
          NOTE: evaluated at SLACK thresholds (1.25, 1.5 x eps_c), not at eps_c.
          At eps_c the graph is barely connected by construction -- a near-tree,
          and near-trees are made of articulation points.  eps_c results are
          printed for comparison but are NOT the registered outcome.

          H_bridge : the top-betweenness model at 1.25*eps_c is not the same in
                     all regimes, in >= PASS_BAR zoos.
          NULL     : bridge identity is regime-invariant, and XIX's per-regime
                     bridge variation was an artifact of standing on eps_c.

  TEST 3  Topological transition.
          Regime-to-regime variation is not a transition.  A transition needs a
          swept parameter and a discontinuity.  Stress is swept continuously and
          graph invariants are read at a SCALE-INVARIANT threshold (a fixed
          quantile of the off-diagonal distances), so that a global rescaling
          cannot masquerade as a topological event.

          H_transition : n_components jumps by >= JUMP_MIN at some stress level,
                         at a consistent location (+/- 1 grid step) across
                         >= PASS_BAR zoos.
          NULL         : invariants vary smoothly, or jump locations are
                         inconsistent across zoos.

Outputs (in ./xxiii_out/):
    distances_zoo{z}.npz          per-regime distance matrices, per zoo  [PERSISTED --
                                  the original run saved only heatmap PNGs, which is
                                  why none of this could be re-tested from artifacts]
    test1_scale_vs_shape.csv
    test2_bridge_identity.csv
    test3_topological_sweep.csv
    xxiii_summary.txt             registered outcomes, pass/fail

Runtime: ~1.5-2 h for N_ZOOS=20 on 8 cores, CPU-only.  Set N_ZOOS=3 for a smoke test.
"""

import os, json, itertools
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import networkx as nx
from scipy.ndimage import convolve
from scipy.stats import pearsonr

# ======================================================================
# REGISTERED CONSTANTS -- fixed before any run
# ======================================================================
N_ZOOS        = 20
PASS_BAR      = 16      # of N_ZOOS

WARP_MARGIN   = 0.20    # Test 1: required shortfall below the split-half ceiling
SLACK_FACTORS = [1.0, 1.25, 1.5]   # 1.0 = eps_c, reported but NOT registered
BRIDGE_TAU    = 1.25    # Test 2: the registered threshold
SWEEP_QUANTILE = 0.40   # Test 3: scale-invariant graph threshold
JUMP_MIN      = 2       # Test 3: minimum discontinuity in n_components

torch.set_num_threads(os.cpu_count() or 8)
OUT = 'xxiii_out'; os.makedirs(OUT, exist_ok=True)

INPUT_LEN = 20
OFFSETS   = [5, 10, 20]
MAX_OFF   = max(OFFSETS)
RENDER    = 16

MODEL_SPECS = [
    ('normal_h8',       8, 'normal', 'base'),
    ('normal_h16',     16, 'normal', 'base'),
    ('compressed_h2',   2, 'normal', 'base'),
    ('wind_h8',         8, 'wind',   'base'),
    ('damped_h8',       8, 'damped', 'base'),
    ('blur_h8',         8, 'blur',   'base'),
    ('velocity_aux_h8', 8, 'normal', 'velocity'),
]
MODEL_NAMES = [s[0] for s in MODEL_SPECS]
N_MODELS = len(MODEL_NAMES)

REGIME_SCHEDULE = [('normal',500), ('wind',500), ('damped',500),
                   ('blur',500), ('normal',500), ('wind',500)]
REGIME_KEYS = [f"{i+1:02d}_{r}" for i,(r,_) in enumerate(REGIME_SCHEDULE)]

# ======================================================================
# Environment (identical dynamics to 04a; wind_accel exposed for the sweep)
# ======================================================================
class BouncingDotEnv:
    def __init__(self, regime='normal', box_size=1.0, dt=0.05, render_size=RENDER,
                 noise_std=0.05, wind_accel=0.3, damp_factor=0.6, rng=None):
        self.box_size, self.dt, self.render_size = box_size, dt, render_size
        self.noise_std, self.wind_accel, self.damp_factor = noise_std, wind_accel, damp_factor
        self.regime = regime
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        self.x = self.rng.random() * self.box_size
        self.y = self.rng.random() * self.box_size
        self.vx = (self.rng.random() - 0.5) * 2.0
        self.vy = (self.rng.random() - 0.5) * 2.0

    def set_regime(self, r): self.regime = r

    def step(self):
        if self.regime == 'wind':
            self.vx += self.wind_accel * self.dt
            self.vy += self.wind_accel * self.dt
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        for ax in ('x','y'):
            p, v = getattr(self, ax), getattr(self, 'v'+ax)
            if p <= 0 or p >= self.box_size:
                v = v * (-self.damp_factor if self.regime == 'damped' else -1.0)
                p = float(np.clip(p, 0, self.box_size))
            setattr(self, ax, p); setattr(self, 'v'+ax, v)
        img = np.zeros((self.render_size, self.render_size))
        px = int(np.clip(self.x/self.box_size*(self.render_size-1), 0, self.render_size-1))
        py = int(np.clip(self.y/self.box_size*(self.render_size-1), 0, self.render_size-1))
        img[max(0,px-1):min(self.render_size,px+2), max(0,py-1):min(self.render_size,py+2)] = 1.0
        img += self.rng.standard_normal((self.render_size, self.render_size)) * self.noise_std
        img = np.clip(img, 0, 1)
        if self.regime == 'blur':
            img = convolve(img, np.ones((3,3))/9.0, mode='constant', cval=0.0)
        return img.flatten(), np.array([self.x, self.y, self.vx, self.vy])


def rollout(regime, n_steps, rng, wind_accel=0.3):
    env = BouncingDotEnv(regime=regime, wind_accel=wind_accel, rng=rng)
    F, P = [], []
    for _ in range(n_steps):
        f, s = env.step(); F.append(f); P.append(s)
    return np.array(F, dtype=np.float32), np.array(P, dtype=np.float32)


def schedule_stream(rng, wind_accel=0.3):
    env = BouncingDotEnv(regime='normal', wind_accel=wind_accel, rng=rng)
    F, P = [], []
    for reg, dur in REGIME_SCHEDULE:
        env.set_regime(reg)
        for _ in range(dur):
            f, s = env.step(); F.append(f); P.append(s)
    return np.array(F, dtype=np.float32), np.array(P, dtype=np.float32)

# ======================================================================
# Models
# ======================================================================
class BasePredictor(nn.Module):
    def __init__(self, hidden_dim=8, input_dim=RENDER*RENDER, num_offsets=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, 2*num_offsets)
        self.num_offsets = num_offsets
    def forward(self, x):
        _, h = self.gru(x)
        return self.decoder(h.squeeze(0)).view(x.shape[0], self.num_offsets, 2)

class VelocityAuxPredictor(nn.Module):
    def __init__(self, hidden_dim=8, input_dim=RENDER*RENDER, num_offsets=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, 2*num_offsets)
        self.speed_head = nn.Linear(hidden_dim, 2)
        self.num_offsets = num_offsets
    def forward(self, x):
        _, h = self.gru(x); h = h.squeeze(0)
        return self.decoder(h).view(x.shape[0], self.num_offsets, 2), self.speed_head(h)


def make_model(hid, kind):
    return VelocityAuxPredictor(hidden_dim=hid) if kind == 'velocity' else BasePredictor(hidden_dim=hid)


def windows(F, P):
    """Vectorised (X, Ypos, Yvel) windows from a rollout."""
    T = len(F) - INPUT_LEN - MAX_OFF
    X   = np.stack([F[t:t+INPUT_LEN] for t in range(T)])
    Yp  = np.stack([[P[t+INPUT_LEN+o-1, :2] for o in OFFSETS] for t in range(T)])
    Yv  = np.stack([P[t+INPUT_LEN-1, 2:] for t in range(T)])
    return (torch.from_numpy(X), torch.from_numpy(Yp), torch.from_numpy(Yv))


def train_model(spec, rng, epochs=6, batch=128, lr=1e-3, n_episodes=24, ep_len=600):
    name, hid, train_regime, kind = spec
    Xs, Yps, Yvs = [], [], []
    for _ in range(n_episodes):
        F, P = rollout(train_regime, ep_len, rng)
        x, yp, yv = windows(F, P); Xs.append(x); Yps.append(yp); Yvs.append(yv)
    X = torch.cat(Xs); Yp = torch.cat(Yps); Yv = torch.cat(Yvs)
    model = make_model(hid, kind)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            out = model(X[idx])
            if kind == 'velocity':
                pos, spd = out
                loss = nn.functional.mse_loss(pos, Yp[idx]) + 0.1*nn.functional.mse_loss(spd, Yv[idx])
            else:
                loss = nn.functional.mse_loss(out, Yp[idx])
            loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_stream(model, F, batch=512):
    """Batched predictions over a stream -> (T, n_offsets, 2)."""
    T = len(F) - INPUT_LEN - MAX_OFF
    X = torch.from_numpy(np.stack([F[t:t+INPUT_LEN] for t in range(T)]))
    outs = []
    for i in range(0, T, batch):
        o = model(X[i:i+batch])
        outs.append((o[0] if isinstance(o, tuple) else o))
    return torch.cat(outs).numpy()

# ======================================================================
# Distances
# ======================================================================
def dmat_from_preds(preds):
    """preds: (n_models, T, off, 2) -> symmetric MSE distance matrix."""
    n = preds.shape[0]
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        d = float(np.mean((preds[i] - preds[j])**2))
        D[i, j] = D[j, i] = d
    return D

TRIU = np.triu_indices(N_MODELS, k=1)

def norm_triu(D):
    """Upper triangle normalized by its own mean -> pure global rescaling becomes identity."""
    v = D[TRIU]
    m = v.mean()
    return v / m if m > 0 else v

# ======================================================================
# Graph helpers
# ======================================================================
def eps_c(D):
    """Single-linkage connectivity threshold == MST bottleneck edge (exact, no grid)."""
    G = nx.Graph()
    for i, j in itertools.combinations(range(N_MODELS), 2):
        G.add_edge(i, j, weight=D[i, j])
    mst = nx.minimum_spanning_tree(G)
    return max(e['weight'] for _, _, e in mst.edges(data=True))

def graph_at(D, tau):
    G = nx.Graph(); G.add_nodes_from(range(N_MODELS))
    for i, j in itertools.combinations(range(N_MODELS), 2):
        if D[i, j] <= tau: G.add_edge(i, j)
    return G

def top_bridge(D, tau):
    G = graph_at(D, tau)
    if G.number_of_edges() == 0: return None, []
    bt = nx.betweenness_centrality(G, normalized=True)
    best = max(bt.values())
    if best <= 0: return None, list(nx.articulation_points(G))
    winners = [m for m, v in bt.items() if v == best]
    # ties are unresolvable -> report as a tie, never silently pick one
    return (MODEL_NAMES[winners[0]] if len(winners) == 1 else 'TIE'), \
           [MODEL_NAMES[a] for a in nx.articulation_points(G)]

def invariants(D, tau):
    G = graph_at(D, tau)
    comps = list(nx.connected_components(G))
    return dict(n_components=len(comps),
                largest=max((len(c) for c in comps), default=0),
                cycle_rank=G.number_of_edges() - N_MODELS + len(comps),
                edge_density=G.number_of_edges()/(N_MODELS*(N_MODELS-1)/2))

# ======================================================================
# Per-zoo pipeline
# ======================================================================
def run_zoo(z):
    rng = np.random.default_rng(1000 + z)
    torch.manual_seed(1000 + z)
    models = [train_model(s, rng) for s in MODEL_SPECS]

    # ---- shared regime-shift stream, per-regime distance matrices ----
    F, P = schedule_stream(np.random.default_rng(5000 + z))
    preds = np.stack([predict_stream(m, F) for m in models])   # (M, T, off, 2)

    cum = np.cumsum([0] + [d for _, d in REGIME_SCHEDULE])
    D_by_regime, D_split = {}, {}
    for r, (reg, _) in enumerate(REGIME_SCHEDULE):
        s = max(INPUT_LEN, cum[r]) - INPUT_LEN
        e = min(cum[r+1] - MAX_OFF, len(F) - MAX_OFF) - INPUT_LEN
        if e - s < 60: continue
        seg = preds[:, s:e]
        key = REGIME_KEYS[r]
        D_by_regime[key] = dmat_from_preds(seg)
        h = (e - s)//2                                  # split-half noise ceiling
        D_split[key] = (dmat_from_preds(seg[:, :h]), dmat_from_preds(seg[:, h:]))

    np.savez(f'{OUT}/distances_zoo{z}.npz',
             **{k: v for k, v in D_by_regime.items()},
             **{f'{k}__A': v[0] for k, v in D_split.items()},
             **{f'{k}__B': v[1] for k, v in D_split.items()})

    # ---- TEST 1: scale vs shape ----
    keys = list(D_by_regime)
    ceiling = np.median([pearsonr(norm_triu(D_split[k][0]), norm_triu(D_split[k][1]))[0]
                         for k in keys])
    between_raw  = [pearsonr(D_by_regime[a][TRIU], D_by_regime[b][TRIU])[0]
                    for a, b in itertools.combinations(keys, 2)]
    between_norm = [pearsonr(norm_triu(D_by_regime[a]), norm_triu(D_by_regime[b]))[0]
                    for a, b in itertools.combinations(keys, 2)]
    scale_spread = max(D_by_regime[k][TRIU].mean() for k in keys) / \
                   min(D_by_regime[k][TRIU].mean() for k in keys)
    t1 = dict(zoo=z, ceiling=ceiling,
              between_norm_median=float(np.median(between_norm)),
              between_norm_min=float(np.min(between_norm)),
              between_raw_median=float(np.median(between_raw)),
              shortfall=float(ceiling - np.median(between_norm)),
              scale_spread=float(scale_spread),
              warp=bool(ceiling - np.median(between_norm) >= WARP_MARGIN))

    # ---- TEST 2: bridge identity ----
    t2 = []
    for f in SLACK_FACTORS:
        bridges = {}
        for k in keys:
            tau = f * eps_c(D_by_regime[k])
            b, arts = top_bridge(D_by_regime[k], tau)
            bridges[k] = (b, ';'.join(arts))
        ids = [b for b, _ in bridges.values()]
        t2.append(dict(zoo=z, slack=f,
                       n_distinct_bridges=len(set(ids)),
                       regime_dependent=bool(len(set(ids)) > 1),
                       any_articulation=any(a for _, a in bridges.values()),
                       **{f'bridge_{k}': bridges[k][0] for k in keys},
                       **{f'artic_{k}': bridges[k][1] for k in keys}))

    # ---- TEST 3: continuous stress sweep ----
    t3 = []
    for w in np.linspace(0.0, 0.6, 13):
        Fw, _ = rollout('wind', 1200, np.random.default_rng(9000 + z), wind_accel=float(w))
        pw = np.stack([predict_stream(m, Fw) for m in models])
        Dw = dmat_from_preds(pw)
        tau = np.quantile(Dw[TRIU], SWEEP_QUANTILE)     # scale-invariant
        inv = invariants(Dw, tau)
        t3.append(dict(zoo=z, wind_accel=float(w), mean_dist=float(Dw[TRIU].mean()), **inv))

    return t1, t2, t3

# ======================================================================
def main():
    T1, T2, T3 = [], [], []
    for z in range(N_ZOOS):
        print(f'--- zoo {z+1}/{N_ZOOS} ---', flush=True)
        a, b, c = run_zoo(z)
        T1.append(a); T2.extend(b); T3.extend(c)

    d1 = pd.DataFrame(T1); d2 = pd.DataFrame(T2); d3 = pd.DataFrame(T3)
    d1.to_csv(f'{OUT}/test1_scale_vs_shape.csv', index=False)
    d2.to_csv(f'{OUT}/test2_bridge_identity.csv', index=False)
    d3.to_csv(f'{OUT}/test3_topological_sweep.csv', index=False)

    L = []
    def say(s): print(s); L.append(s)

    def iqr(x): return f'{np.median(x):.3f} [{np.percentile(x,25):.3f}, {np.percentile(x,75):.3f}]'

    say('\n================ REGISTERED OUTCOMES ================\n')

    # TEST 1
    n_warp = int(d1.warp.sum())
    say('TEST 1 -- Scale vs shape')
    say(f'  split-half ceiling (within-regime) : {iqr(d1.ceiling)}')
    say(f'  between-regime corr, NORMALIZED    : {iqr(d1.between_norm_median)}')
    say(f'  between-regime corr, raw           : {iqr(d1.between_raw_median)}')
    say(f'  shortfall below ceiling            : {iqr(d1.shortfall)}   (margin = {WARP_MARGIN})')
    say(f'  scale spread (max/min mean dist)   : {iqr(d1.scale_spread)}')
    say(f'  H_warp: {n_warp}/{N_ZOOS} zoos (bar {PASS_BAR})  -> '
        f'{"PASS: stress reshapes" if n_warp >= PASS_BAR else "FAIL (null holds): stress RESCALES, it does not reshape"}')

    # TEST 2
    say('\nTEST 2 -- Bridge identity')
    for f in SLACK_FACTORS:
        sub = d2[d2.slack == f]
        n_rd = int(sub.regime_dependent.sum())
        tag = 'REGISTERED' if f == BRIDGE_TAU else ('disqualified -- knife-edge' if f == 1.0 else 'secondary')
        say(f'  tau = {f:.2f}*eps_c [{tag}]: regime-dependent in {n_rd}/{N_ZOOS}; '
            f'articulation present in {int(sub.any_articulation.sum())}/{N_ZOOS}')
    n_rd = int(d2[d2.slack == BRIDGE_TAU].regime_dependent.sum())
    say(f'  H_bridge (at {BRIDGE_TAU}*eps_c): {n_rd}/{N_ZOOS} (bar {PASS_BAR}) -> '
        f'{"PASS: the bridge is a property of the regime, not the model" if n_rd >= PASS_BAR else "FAIL (null holds): bridge identity is regime-invariant"}')

    # TEST 3
    say('\nTEST 3 -- Topological transition')
    jumps = []
    for z in range(N_ZOOS):
        s = d3[d3.zoo == z].sort_values('wind_accel')
        dj = np.abs(np.diff(s.n_components.values))
        jumps.append((int(dj.max()) if len(dj) else 0,
                      int(np.argmax(dj)) if len(dj) else -1))
    big = [(m, loc) for m, loc in jumps if m >= JUMP_MIN]
    locs = [loc for m, loc in big]
    consistent = 0
    if locs:
        mode = max(set(locs), key=locs.count)
        consistent = sum(1 for l in locs if abs(l - mode) <= 1)
    say(f'  zoos with a jump >= {JUMP_MIN} in n_components : {len(big)}/{N_ZOOS}')
    say(f'  of those, at a consistent location (+/-1)   : {consistent}')
    say(f'  mean |d n_components| per step              : '
        f'{np.mean([np.abs(np.diff(d3[d3.zoo==z].sort_values("wind_accel").n_components.values)).mean() for z in range(N_ZOOS)]):.3f}')
    say(f'  H_transition: {consistent}/{N_ZOOS} (bar {PASS_BAR}) -> '
        f'{"PASS: a topological transition exists" if consistent >= PASS_BAR else "FAIL (null holds): invariants vary smoothly / jumps are inconsistent"}')

    say('\nNOTE: eps_c is the MST bottleneck edge by construction (single linkage).')
    say('Any claim resting on eps_c varying across regimes is a claim about distance')
    say('magnitude, not about geometry.  Test 1 is what adjudicates that.\n')

    open(f'{OUT}/xxiii_summary.txt', 'w').write('\n'.join(L))
    print(f'Wrote {OUT}/xxiii_summary.txt')

if __name__ == '__main__':
    main()
