#!/usr/bin/env python3
"""
paper_xxiii_geometry_replication_v2.py
======================================
v2.  Two statistics rebuilt after the v1 smoke test exposed them.

WHAT CHANGED
------------
TEST 1.  v1 reported a "normalized" Pearson correlation alongside a raw one and
         they came out identical to three decimals -- because Pearson correlation
         is invariant under scaling, so dividing a distance matrix by its mean
         changes nothing.  The line was a no-op.
         The split-half ceiling comparison it sat next to was, and remains, valid:
         it asks whether two regimes' distance matrices agree as well as two halves
         of the SAME regime's stream do.  v2 keeps it and adds a second, independent
         shape statistic that is NOT automatically scale-invariant -- a Frobenius
         distance between mean-normalized matrices, read against its own split-half
         ceiling.  Two statistics, one question, and H_warp passes if EITHER detects
         a warp.  Being generous to the alternative makes a null result mean more.

TEST 2.  v1 counted 'TIE' and 'None' as bridge identities alongside real model
         names, so "3 distinct bridges" could mean {TIE, normal_h8, normal_h16}.
         The statistic was measuring whether a bridge EXISTS, not which model it IS,
         and its PASS was an artifact.
         v2 asks the question directly and robustly: does the full betweenness
         VECTOR rank models the same way across regimes?  Ties no longer need
         breaking.  An admission gate is added, because the smoke test suggests the
         common outcome at slack is no bridge structure at all -- and if that holds,
         it is the finding, not an obstacle to one.

TEST 3.  Unchanged; it was sound.

Runtime: ~1.5-2 h for N_ZOOS=20 on 8 cores, CPU-only.  N_ZOOS=3 to smoke-test.
"""

import os, itertools
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import networkx as nx
from scipy.ndimage import convolve
from scipy.stats import pearsonr, spearmanr

# ======================================================================
# REGISTERED CONSTANTS
# ======================================================================
N_ZOOS         = 20
PASS_BAR       = 16

# Test 1
WARP_MARGIN    = 0.20    # Pearson: required shortfall below the split-half ceiling
FROB_RATIO_BAR = 1.50    # Frobenius: between-regime shape distance / within-regime ceiling

# Test 2
BRIDGE_TAU     = 1.25    # registered threshold (NOT eps_c -- see the audit)
SLACK_FACTORS  = [1.0, 1.25, 1.5]
BRIDGE_RHO     = 0.50    # betweenness rankings this correlated or less => regime-dependent
DEGENERATE_TOL = 1e-9    # a betweenness vector of all-zeros carries no ranking
MIN_QUALIFYING = 2       # regimes with non-degenerate betweenness needed to judge a zoo

# Test 3
SWEEP_QUANTILE = 0.40
JUMP_MIN       = 2

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
# Environment
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
        self.x = self.rng.random()*self.box_size; self.y = self.rng.random()*self.box_size
        self.vx = (self.rng.random()-0.5)*2.0;    self.vy = (self.rng.random()-0.5)*2.0
    def set_regime(self, r): self.regime = r
    def step(self):
        if self.regime == 'wind':
            self.vx += self.wind_accel*self.dt; self.vy += self.wind_accel*self.dt
        self.x += self.vx*self.dt; self.y += self.vy*self.dt
        for ax in ('x','y'):
            p, v = getattr(self, ax), getattr(self, 'v'+ax)
            if p <= 0 or p >= self.box_size:
                v = v*(-self.damp_factor if self.regime=='damped' else -1.0)
                p = float(np.clip(p, 0, self.box_size))
            setattr(self, ax, p); setattr(self, 'v'+ax, v)
        img = np.zeros((self.render_size, self.render_size))
        px = int(np.clip(self.x/self.box_size*(self.render_size-1), 0, self.render_size-1))
        py = int(np.clip(self.y/self.box_size*(self.render_size-1), 0, self.render_size-1))
        img[max(0,px-1):min(self.render_size,px+2), max(0,py-1):min(self.render_size,py+2)] = 1.0
        img += self.rng.standard_normal((self.render_size, self.render_size))*self.noise_std
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
    return VelocityAuxPredictor(hidden_dim=hid) if kind=='velocity' else BasePredictor(hidden_dim=hid)

def windows(F, P):
    T = len(F) - INPUT_LEN - MAX_OFF
    X  = np.stack([F[t:t+INPUT_LEN] for t in range(T)])
    Yp = np.stack([[P[t+INPUT_LEN+o-1, :2] for o in OFFSETS] for t in range(T)])
    Yv = np.stack([P[t+INPUT_LEN-1, 2:] for t in range(T)])
    return torch.from_numpy(X), torch.from_numpy(Yp), torch.from_numpy(Yv)

def train_arch(hid, kind, regime, rng, epochs=6, batch=128, lr=1e-3, n_episodes=24, ep_len=600):
    """Train a fresh model of a given ARCHITECTURE on a given REGIME.  Used both for the
    zoo and (in the cost script) for the capacity-matched reference floors."""
    Xs, Yps, Yvs = [], [], []
    for _ in range(n_episodes):
        F, P = rollout(regime, ep_len, rng)
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
    model.eval(); return model

def train_model(spec, rng, **kw):
    _, hid, regime, kind = spec
    return train_arch(hid, kind, regime, rng, **kw)

@torch.no_grad()
def predict_stream(model, F, batch=512):
    T = len(F) - INPUT_LEN - MAX_OFF
    X = torch.from_numpy(np.stack([F[t:t+INPUT_LEN] for t in range(T)]))
    outs = []
    for i in range(0, T, batch):
        o = model(X[i:i+batch])
        outs.append(o[0] if isinstance(o, tuple) else o)
    return torch.cat(outs).numpy()

# ======================================================================
# Distances
# ======================================================================
TRIU = np.triu_indices(N_MODELS, k=1)

def dmat_from_preds(preds):
    n = preds.shape[0]; D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        D[i,j] = D[j,i] = float(np.mean((preds[i]-preds[j])**2))
    return D

def mean_normalized(D):
    """D / mean(off-diagonal).  A pure global rescaling maps to the identity."""
    v = D[TRIU]; m = v.mean()
    return D/m if m > 0 else D

def frob_shape_dist(D1, D2):
    """Relative Frobenius distance between mean-normalized matrices.
    Unlike Pearson, this is NOT automatically scale-invariant before normalization,
    so it is a genuinely independent read on shape once normalization is applied."""
    A, B = mean_normalized(D1), mean_normalized(D2)
    denom = np.linalg.norm(A[TRIU])
    return float(np.linalg.norm(A[TRIU]-B[TRIU]) / denom) if denom > 0 else np.nan

# ======================================================================
# Graph helpers
# ======================================================================
def eps_c(D):
    """Single-linkage connectivity threshold == MST bottleneck edge.  Exact, no grid."""
    G = nx.Graph()
    for i, j in itertools.combinations(range(N_MODELS), 2):
        G.add_edge(i, j, weight=D[i,j])
    mst = nx.minimum_spanning_tree(G)
    return max(e['weight'] for _,_,e in mst.edges(data=True))

def graph_at(D, tau):
    G = nx.Graph(); G.add_nodes_from(range(N_MODELS))
    for i, j in itertools.combinations(range(N_MODELS), 2):
        if D[i,j] <= tau: G.add_edge(i, j)
    return G

def betweenness_vector(D, tau):
    """Full betweenness vector -- no argmax, no tie-breaking.  Returns None if the
    vector is degenerate (all zeros), which carries no ranking information: that is
    the case of a complete or edgeless graph, where nothing bridges anything."""
    G = graph_at(D, tau)
    bt = nx.betweenness_centrality(G, normalized=True)
    v = np.array([bt[i] for i in range(N_MODELS)])
    return None if v.max() <= DEGENERATE_TOL else v

def invariants(D, tau):
    G = graph_at(D, tau)
    comps = list(nx.connected_components(G))
    return dict(n_components=len(comps),
                largest=max((len(c) for c in comps), default=0),
                cycle_rank=G.number_of_edges()-N_MODELS+len(comps),
                edge_density=G.number_of_edges()/(N_MODELS*(N_MODELS-1)/2))

# ======================================================================
def run_zoo(z):
    rng = np.random.default_rng(1000+z); torch.manual_seed(1000+z)
    models = [train_model(s, rng) for s in MODEL_SPECS]

    F, _ = schedule_stream(np.random.default_rng(5000+z))
    preds = np.stack([predict_stream(m, F) for m in models])

    cum = np.cumsum([0]+[d for _, d in REGIME_SCHEDULE])
    D_reg, D_split = {}, {}
    for r, (reg, _) in enumerate(REGIME_SCHEDULE):
        s = max(INPUT_LEN, cum[r]) - INPUT_LEN
        e = min(cum[r+1]-MAX_OFF, len(F)-MAX_OFF) - INPUT_LEN
        if e - s < 60: continue
        seg = preds[:, s:e]; key = REGIME_KEYS[r]
        D_reg[key] = dmat_from_preds(seg)
        h = (e-s)//2
        D_split[key] = (dmat_from_preds(seg[:, :h]), dmat_from_preds(seg[:, h:]))

    np.savez(f'{OUT}/distances_zoo{z}.npz',
             **D_reg,
             **{f'{k}__A': v[0] for k, v in D_split.items()},
             **{f'{k}__B': v[1] for k, v in D_split.items()})

    keys = list(D_reg)

    # ---------------- TEST 1: two independent shape statistics ----------------
    pear_ceiling = np.median([pearsonr(D_split[k][0][TRIU], D_split[k][1][TRIU])[0] for k in keys])
    pear_between = [pearsonr(D_reg[a][TRIU], D_reg[b][TRIU])[0]
                    for a, b in itertools.combinations(keys, 2)]
    frob_ceiling = np.median([frob_shape_dist(*D_split[k]) for k in keys])
    frob_between = [frob_shape_dist(D_reg[a], D_reg[b])
                    for a, b in itertools.combinations(keys, 2)]
    frob_ratio = float(np.median(frob_between)/frob_ceiling) if frob_ceiling > 0 else np.nan
    scale_spread = (max(D_reg[k][TRIU].mean() for k in keys) /
                    min(D_reg[k][TRIU].mean() for k in keys))

    warp_pear = bool(pear_ceiling - np.median(pear_between) >= WARP_MARGIN)
    warp_frob = bool(frob_ratio >= FROB_RATIO_BAR)
    t1 = dict(zoo=z,
              pear_ceiling=float(pear_ceiling),
              pear_between=float(np.median(pear_between)),
              pear_between_min=float(np.min(pear_between)),
              pear_shortfall=float(pear_ceiling - np.median(pear_between)),
              frob_ceiling=float(frob_ceiling),
              frob_between=float(np.median(frob_between)),
              frob_ratio=frob_ratio,
              scale_spread=float(scale_spread),
              warp_pear=warp_pear, warp_frob=warp_frob,
              warp=bool(warp_pear or warp_frob))       # generous to the alternative

    # ---------------- TEST 2: betweenness-vector agreement ----------------
    t2 = []
    for f in SLACK_FACTORS:
        vecs = {k: betweenness_vector(D_reg[k], f*eps_c(D_reg[k])) for k in keys}
        named = [k for k, v in vecs.items() if v is not None]
        qualifies = len(named) >= MIN_QUALIFYING
        if qualifies:
            rhos = [spearmanr(vecs[a], vecs[b])[0]
                    for a, b in itertools.combinations(named, 2)]
            rhos = [r for r in rhos if not np.isnan(r)]
            med = float(np.median(rhos)) if rhos else np.nan
        else:
            med = np.nan
        t2.append(dict(zoo=z, slack=f,
                       n_regimes_with_structure=len(named),
                       qualifies=bool(qualifies),
                       median_betweenness_rho=med,
                       regime_dependent=bool(qualifies and not np.isnan(med) and med <= BRIDGE_RHO),
                       **{f'has_structure_{k}': (vecs[k] is not None) for k in keys}))

    # ---------------- TEST 3: continuous stress sweep ----------------
    t3 = []
    for w in np.linspace(0.0, 0.6, 13):
        Fw, _ = rollout('wind', 1200, np.random.default_rng(9000+z), wind_accel=float(w))
        Dw = dmat_from_preds(np.stack([predict_stream(m, Fw) for m in models]))
        tau = np.quantile(Dw[TRIU], SWEEP_QUANTILE)
        t3.append(dict(zoo=z, wind_accel=float(w), mean_dist=float(Dw[TRIU].mean()),
                       **invariants(Dw, tau)))
    return t1, t2, t3

# ======================================================================
def main():
    T1, T2, T3 = [], [], []
    for z in range(N_ZOOS):
        print(f'--- zoo {z+1}/{N_ZOOS} ---', flush=True)
        a, b, c = run_zoo(z); T1.append(a); T2.extend(b); T3.extend(c)
    d1 = pd.DataFrame(T1); d2 = pd.DataFrame(T2); d3 = pd.DataFrame(T3)
    d1.to_csv(f'{OUT}/test1_scale_vs_shape.csv', index=False)
    d2.to_csv(f'{OUT}/test2_bridge_identity.csv', index=False)
    d3.to_csv(f'{OUT}/test3_topological_sweep.csv', index=False)

    L = []
    def say(s): print(s); L.append(s)
    def iqr(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        if not len(x): return 'n/a'
        return f'{np.median(x):.3f} [{np.percentile(x,25):.3f}, {np.percentile(x,75):.3f}]'

    say('\n================ REGISTERED OUTCOMES (v2) ================\n')

    # ---- TEST 1 ----
    say('TEST 1 -- Does stress RESHAPE factorization space, or only RESCALE it?')
    say(f'  [Pearson]   within-regime split-half ceiling : {iqr(d1.pear_ceiling)}')
    say(f'  [Pearson]   between-regime                   : {iqr(d1.pear_between)}')
    say(f'  [Pearson]   shortfall below ceiling          : {iqr(d1.pear_shortfall)}  (bar >= {WARP_MARGIN})')
    say(f'  [Frobenius] within-regime split-half ceiling : {iqr(d1.frob_ceiling)}')
    say(f'  [Frobenius] between-regime shape distance    : {iqr(d1.frob_between)}')
    say(f'  [Frobenius] ratio between/within             : {iqr(d1.frob_ratio)}  (bar >= {FROB_RATIO_BAR})')
    say(f'  scale spread (max/min mean distance)         : {iqr(d1.scale_spread)}')
    say(f'  warp detected by Pearson   : {int(d1.warp_pear.sum())}/{N_ZOOS}')
    say(f'  warp detected by Frobenius : {int(d1.warp_frob.sum())}/{N_ZOOS}')
    nw = int(d1.warp.sum())
    say(f'  H_warp (EITHER statistic): {nw}/{N_ZOOS} (bar {PASS_BAR}) -> '
        f'{"PASS: stress reshapes the space" if nw >= PASS_BAR else "FAIL (null holds): stress RESCALES, it does not reshape"}')
    if nw < PASS_BAR:
        say('  >> Two independent shape statistics, both generous to the alternative, fail to')
        say('  >> distinguish between-regime shape from within-regime sampling noise, while the')
        say('  >> scale of the map moves substantially.  The environment sets the SIZE of')
        say('  >> factorization space, not its SHAPE.')

    # ---- TEST 2 ----
    say('\nTEST 2 -- Is the bridge a property of the MODEL or of the REGIME?')
    for f in SLACK_FACTORS:
        sub = d2[d2.slack == f]
        tag = ('REGISTERED' if f == BRIDGE_TAU else
               'disqualified -- knife-edge' if f == 1.0 else 'secondary')
        say(f'  tau = {f:.2f}*eps_c [{tag}]')
        say(f'      regimes with any bridge structure : {iqr(sub.n_regimes_with_structure)} of 6')
        say(f'      zoos qualifying (>= {MIN_QUALIFYING} such regimes) : {int(sub.qualifies.sum())}/{N_ZOOS}')
        say(f'      median betweenness-rank rho        : {iqr(sub.median_betweenness_rho)}')
    reg = d2[d2.slack == BRIDGE_TAU]
    nq = int(reg.qualifies.sum())
    say(f'\n  GATE: {nq}/{N_ZOOS} zoos have bridge structure in >= {MIN_QUALIFYING} regimes at slack (bar {PASS_BAR})')
    if nq < PASS_BAR:
        say('  GATE FAILED -- H_bridge WITHHELD.  Bridge structure does not survive 25% translation')
        say('  slack in most zoos.  Articulation at eps_c is near-definitional (the graph is a')
        say('  near-tree there, and near-trees are made of cut vertices), so the finding is not')
        say('  that bridges are regime-dependent but that BRIDGES ARE A KNIFE-EDGE PHENOMENON.')
        say('  This is the reported result.')
    else:
        nrd = int(reg.regime_dependent.sum())
        say(f'  H_bridge: {nrd}/{N_ZOOS} (bar {PASS_BAR}) -> '
            f'{"PASS: betweenness rankings differ across regimes -- the bridge is a property of the regime" if nrd >= PASS_BAR else "FAIL (null holds): betweenness rankings are regime-invariant"}')

    # ---- TEST 3 ----
    say('\nTEST 3 -- Is there a topological transition, or only smooth drift?')
    jumps = []
    for z in range(N_ZOOS):
        s = d3[d3.zoo == z].sort_values('wind_accel')
        dj = np.abs(np.diff(s.n_components.values))
        jumps.append((int(dj.max()) if len(dj) else 0, int(np.argmax(dj)) if len(dj) else -1))
    big = [(m, l) for m, l in jumps if m >= JUMP_MIN]
    locs = [l for _, l in big]
    consistent = 0
    if locs:
        mode = max(set(locs), key=locs.count)
        consistent = sum(1 for l in locs if abs(l-mode) <= 1)
    mean_step = np.mean([np.abs(np.diff(d3[d3.zoo==z].sort_values('wind_accel').n_components.values)).mean()
                         for z in range(N_ZOOS)])
    say(f'  zoos with a jump >= {JUMP_MIN} in n_components : {len(big)}/{N_ZOOS}')
    say(f'  of those, at a consistent location (+/-1)  : {consistent}')
    say(f'  mean |d n_components| per sweep step       : {mean_step:.3f}')
    say(f'  H_transition: {consistent}/{N_ZOOS} (bar {PASS_BAR}) -> '
        f'{"PASS: a topological transition exists" if consistent >= PASS_BAR else "FAIL (null holds): invariants drift smoothly"}')

    say('\nNOTE: eps_c is the MST bottleneck edge by construction (single linkage), so any')
    say('claim resting on eps_c varying across regimes is a claim about distance MAGNITUDE,')
    say('not about geometry.  Test 1 is what adjudicates that, and Test 2 is why no result')
    say('is registered at eps_c.\n')

    open(f'{OUT}/xxiii_summary.txt','w').write('\n'.join(L))
    print(f'Wrote {OUT}/xxiii_summary.txt')

if __name__ == '__main__':
    main()
