#!/usr/bin/env python3
"""
paper_xix-1-role_triad_replication.py
==============================
Registered replication for Paper XIX per 19-0-role-triad-preregistration.md.

For each seed (= one independently trained zoo):
  1. retrain all 7 zoo models on their regime-specific data,
  2. build the fixed 6-segment regime-shift stream,
  3. evaluate the 5 architectures (P1),
  4. compute per-model governor / sentinel / bridge scores (P2, P4),
  5. compute diverse vs top-utility sentinel portfolios (P3).

Resumable: per-seed results are appended to CSVs and cached to disk; a seed
whose row already exists in role_triad_results.csv is skipped. Trained model
weights are cached under models_cache/seed_XX/ so an interrupted seed can
resume without retraining completed models.

CPU-sized (tuned for an 8-core / 16-thread Ryzen, no CUDA). Stream evaluation
is batched (all timesteps in one forward pass per model) rather than looped,
which is the main speedup over the pilot script.

Usage:
    python 19-1-role_triad_replication.py                # seeds 0-19, then analyze
    python 19-1-role_triad_replication.py --seeds 5      # smoke: seeds 0-4
    python 19-1-role_triad_replication.py --start 10 --seeds 20
    python 19-1-role_triad_replication.py --analyze-only
"""

import os, csv, gc, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage import convolve
from scipy.stats import spearmanr

try:
    import networkx as nx
    HAVE_NX = True
except ImportError:
    HAVE_NX = False

torch.set_num_threads(os.cpu_count() or 8)
DEVICE = "cpu"

# ------------------------------------------------------------------
# Registered configuration
# ------------------------------------------------------------------
OFFSETS = [5, 10, 20]
INPUT_LEN = 20
RENDER = 16
DIM = RENDER * RENDER

# Training (CPU-sized; lighter than pilot's 2000-seq/30-epoch, per prereg budget)
NUM_SEQ = 500
SEQ_LEN = 200
WINDOWS_PER_SEQ = 12
EPOCHS = 20
PATIENCE = 4
LR = 1e-3
BATCH = 128

# Stream
SEG_LEN = 500
REGIME_SEQUENCE = ["normal", "wind", "damped", "blur", "normal", "wind"]

# Detector (registered, robust variant from pilot)
ROLL = 20
MARGIN_FRAC = 0.10
HORIZON = 50
LEAD_MIN = 10
SPIKE_MULT = 2.0
SPIKE_BASE_WIN = 10

# Zoo (fixed across seeds)
ZOO = [
    ("normal_h8",       8,  "normal", "normal",   False),
    ("normal_h16",      16, "normal", "normal",   False),
    ("compressed_h2",   2,  "normal", "normal",   False),
    ("wind_h8",         8,  "wind",   "normal",   False),
    ("damped_h8",       8,  "damped", "normal",   False),
    ("blur_h8",         8,  "blur",   "normal",   False),
    ("velocity_aux_h8", 8,  "normal", "velocity", True),
]
MODEL_NAMES = [z[0] for z in ZOO]

OUT = "outputs"
CACHE = "models_cache"
RESULTS_CSV = os.path.join(OUT, "role_triad_results.csv")
PORT_CSV = os.path.join(OUT, "portfolio_results_xix.csv")
RES_FIELDS = ["seed", "model", "governor_mse", "sentinel_cov", "sentinel_prec",
              "betweenness", "is_articulation_any"]
PORT_FIELDS = ["seed", "K", "diverse_cov", "top_utility_cov"]
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
class BouncingDot:
    def __init__(self, rng, regime="normal", noise=0.05,
                 wind=0.3, damp=0.6):
        self.rng, self.regime = rng, regime
        self.noise, self.wind, self.damp = noise, wind, damp
        self.dt = 0.05
        self.reset()

    def reset(self):
        self.x, self.y = self.rng.random(), self.rng.random()
        self.vx = (self.rng.random() - 0.5) * 2
        self.vy = (self.rng.random() - 0.5) * 2

    def step(self):
        if self.regime == "wind":
            self.vx += self.wind * self.dt
            self.vy += self.wind * self.dt
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        if self.x <= 0 or self.x >= 1:
            self.vx *= -self.damp if self.regime == "damped" else -1
            self.x = min(max(self.x, 0), 1)
        if self.y <= 0 or self.y >= 1:
            self.vy *= -self.damp if self.regime == "damped" else -1
            self.y = min(max(self.y, 0), 1)
        img = np.zeros((RENDER, RENDER))
        px = int(min(max(self.x * (RENDER - 1), 0), RENDER - 1))
        py = int(min(max(self.y * (RENDER - 1), 0), RENDER - 1))
        img[max(0, px-1):min(RENDER, px+2), max(0, py-1):min(RENDER, py+2)] = 1.0
        img += self.rng.standard_normal((RENDER, RENDER)) * self.noise
        img = np.clip(img, 0, 1)
        if self.regime == "blur":
            img = convolve(img, np.ones((3, 3)) / 9.0, mode="constant")
        return img.flatten().astype(np.float32), np.array(
            [self.x, self.y, self.vx, self.vy], dtype=np.float32)


def make_dataset(rng, regime, velocity=False):
    env = BouncingDot(rng, regime)
    X, Ypos, Yvel = [], [], []
    maxoff = max(OFFSETS)
    for _ in range(NUM_SEQ):
        env.reset()
        frames, states = [], []
        for _ in range(SEQ_LEN):
            f, s = env.step()
            frames.append(f); states.append(s)
        frames = np.array(frames); states = np.array(states)
        starts = list(range(SEQ_LEN - INPUT_LEN - maxoff))
        if WINDOWS_PER_SEQ < len(starts):
            starts = rng.choice(starts, WINDOWS_PER_SEQ, replace=False)
        for st in starts:
            en = st + INPUT_LEN
            X.append(frames[st:en])
            Ypos.append(np.stack([states[en + o - 1, :2] for o in OFFSETS]))
            if velocity:
                Yvel.append(states[en - 1, 2:])
    X = torch.tensor(np.array(X))
    Ypos = torch.tensor(np.array(Ypos))
    if velocity:
        return X, Ypos, torch.tensor(np.array(Yvel))
    return X, Ypos


def make_stream(rng):
    """Fixed 6-segment regime-shift stream; returns frames, positions, seg_bounds."""
    frames, positions, bounds = [], [], []
    env = BouncingDot(rng, REGIME_SEQUENCE[0])
    env.reset()
    t = 0
    for reg in REGIME_SEQUENCE:
        env.set_regime = None  # no-op guard
        env.regime = reg
        start = t
        for _ in range(SEG_LEN):
            f, s = env.step()
            frames.append(f); positions.append(s[:2]); t += 1
        bounds.append((reg, start, t))
    return np.array(frames, dtype=np.float32), np.array(positions, dtype=np.float32), bounds


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
class Base(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.gru = nn.GRU(DIM, hid, batch_first=True)
        self.dec = nn.Linear(hid, 2 * len(OFFSETS))

    def forward(self, x):
        _, h = self.gru(x)
        return self.dec(h.squeeze(0)).view(x.shape[0], len(OFFSETS), 2)


class VelAux(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.gru = nn.GRU(DIM, hid, batch_first=True)
        self.dec = nn.Linear(hid, 2 * len(OFFSETS))
        self.spd = nn.Linear(hid, 2)

    def forward(self, x):
        _, h = self.gru(x); h = h.squeeze(0)
        return self.dec(h).view(x.shape[0], len(OFFSETS), 2), self.spd(h)


def build(spec):
    _, hid, _, cls, _ = spec
    return VelAux(hid) if cls == "velocity" else Base(hid)


def train_model(model, data, velocity=False):
    n = len(data[0])
    perm = torch.randperm(n)
    sp = int(0.8 * n)
    tr = [a[perm[:sp]] for a in data]
    va = [a[perm[sp:]] for a in data]
    tl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tr),
                                     batch_size=BATCH, shuffle=True)
    vl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*va),
                                     batch_size=BATCH)
    opt = optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    best, best_state, since = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train()
        for batch in tl:
            opt.zero_grad()
            if velocity:
                xb, yp, yv = batch
                pp, sp_ = model(xb)
                loss = mse(pp, yp) + 0.1 * mse(sp_, yv)
            else:
                xb, yb = batch
                loss = mse(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in vl:
                if velocity:
                    xb, yp, yv = batch
                    pp, sp_ = model(xb)
                    l = mse(pp, yp) + 0.1 * mse(sp_, yv)
                else:
                    xb, yb = batch
                    l = mse(model(xb), yb)
                vloss += l.item() * xb.size(0)
        vloss /= len(va[0])
        if vloss < best:
            best, best_state, since = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best


# ------------------------------------------------------------------
# Batched stream evaluation: per-timestep error for one model
# ------------------------------------------------------------------
@torch.no_grad()
def stream_errors(model, frames, positions, velocity=False):
    model.eval()
    maxoff = max(OFFSETS)
    T = len(frames)
    idx = list(range(INPUT_LEN, T - maxoff))
    windows = np.stack([frames[t - INPUT_LEN:t] for t in idx])  # (N, INPUT_LEN, DIM)
    targets = np.stack([[positions[t + o - 1] for o in OFFSETS] for t in idx])
    errs = np.empty(len(idx), dtype=np.float64)
    xb = torch.tensor(windows)
    for i in range(0, len(idx), 512):
        out = model(xb[i:i+512])
        pp = out[0] if isinstance(out, tuple) else out
        pp = pp.numpy()
        errs[i:i+512] = np.mean((pp - targets[i:i+512]) ** 2, axis=(1, 2))
    return np.array(idx), errs


# ------------------------------------------------------------------
# Architectures (P1) — operate on precomputed per-model error arrays
# ------------------------------------------------------------------
def arch_mse(all_errs, val_losses):
    """all_errs: dict name->error array (aligned). Returns dict arch->mean MSE."""
    names = MODEL_NAMES
    E = np.stack([all_errs[n] for n in names])  # (M, N)
    M, N = E.shape
    out = {}
    # Monoculture: best model by validation loss, used throughout
    mono = int(np.argmin(val_losses))
    out["monoculture"] = E[mono].mean()
    # Full pluralism: mean prediction ~ approximated by mean of errors' lower bound?
    # Proper mean-prediction MSE differs from mean error; recompute below.
    out["full_pluralism"] = None  # filled by caller with true averaged preds
    # Oracle WTA: per-step min error (upper-bounds an oracle that switches freely)
    out["wta_oracle"] = E.min(axis=0).mean()
    # Closed WTA: pick best-by-val at start, switch only at reviews
    active = mono
    review_interval = 500
    closed = np.empty(N)
    win = 30
    for t in range(N):
        closed[t] = E[active, t]
        if t > 0 and t % review_interval == 0:
            lo = max(0, t - win)
            active = int(np.argmin(E[:, lo:t].mean(axis=1)))
    out["wta_closed"] = closed.mean()
    # Adaptive audit: rolling-window best with persistence + switch margin
    win_a, margin, persist = 50, 0.05, 5
    active = mono
    cnt = 0
    adap = np.empty(N)
    for t in range(N):
        adap[t] = E[active, t]
        lo = max(0, t - win_a)
        avg = E[:, lo:t+1].mean(axis=1)
        best = int(np.argmin(avg))
        if best != active and avg[best] < avg[active] * (1 - margin):
            cnt += 1
            if cnt >= persist:
                active = best; cnt = 0
        else:
            cnt = 0
    out["adaptive"] = adap.mean()
    return out, mono


# ------------------------------------------------------------------
# Sentinel detector (P2, P3) — returns per-model warning events
# ------------------------------------------------------------------
def sentinel_events(all_errs, active_series, bounds, idx):
    """Return dict model -> set of spike-episode ids it warned for, and TP/FP counts.
    active_series: index (into MODEL_NAMES) active at each aligned step."""
    names = MODEL_NAMES
    E = {n: all_errs[n] for n in names}
    N = len(idx)
    active_err = np.array([all_errs[names[active_series[t]]][t] for t in range(N)])

    # spike episodes: contiguous runs where active error exceeds regime threshold
    step_regime = np.empty(N, dtype=int)
    thr = np.empty(N)
    for ri, (reg, s, e) in enumerate(bounds):
        mask = (idx >= s) & (idx < e)
        step_regime[mask] = ri
        base = active_err[mask][:SPIKE_BASE_WIN]
        thr[mask] = (np.median(base) if len(base) else 0) * SPIKE_MULT
    spike = active_err > thr
    # episode ids
    epis = np.full(N, -1, dtype=int)
    eid = 0
    for t in range(N):
        if spike[t] and (t == 0 or not spike[t-1]):
            eid += 1
        if spike[t]:
            epis[t] = eid
    spike_starts = {}
    for t in range(N):
        if epis[t] > 0 and epis[t] not in spike_starts:
            spike_starts[epis[t]] = t

    warned = {n: set() for n in names}
    tp = {n: 0 for n in names}
    fp = {n: 0 for n in names}
    margin_series = MARGIN_FRAC * active_err
    for n in names:
        delta = active_err - E[n]  # positive when model n beats active
        roll = np.convolve(delta, np.ones(ROLL)/ROLL, mode="same")
        consistent = roll > margin_series
        suppressed = np.array([names[active_series[t]] != n for t in range(N)])
        cand = consistent & suppressed
        for t in range(N):
            if not cand[t]:
                continue
            # does a spike start within (t+LEAD_MIN, t+HORIZON]?
            hit = None
            for eid2, st in spike_starts.items():
                if t + LEAD_MIN <= st <= t + HORIZON:
                    hit = eid2; break
            if hit is not None:
                if hit not in warned[n]:
                    warned[n].add(hit); tp[n] += 1
            else:
                fp[n] += 1
    total_episodes = len(spike_starts)
    return warned, tp, fp, total_episodes


# ------------------------------------------------------------------
# Bridge scores (P4) — behavioral-distance graph
# ------------------------------------------------------------------
def behavioral_distance(all_errs):
    """Distance between models = RMS difference of their error series (proxy for
    behavioral divergence on the stream). Returns (names, DxD matrix)."""
    names = MODEL_NAMES
    E = np.stack([all_errs[n] for n in names])
    M = len(names)
    D = np.zeros((M, M))
    for i in range(M):
        for j in range(i+1, M):
            d = np.sqrt(np.mean((E[i] - E[j]) ** 2))
            D[i, j] = D[j, i] = d
    return names, D


def connectivity_threshold(D):
    """Minimum edge threshold at which the graph (edges where dist <= thr) is connected."""
    if not HAVE_NX:
        return None
    M = D.shape[0]
    edges = sorted({D[i, j] for i in range(M) for j in range(i+1, M)})
    for thr in edges:
        G = nx.Graph()
        G.add_nodes_from(range(M))
        for i in range(M):
            for j in range(i+1, M):
                if D[i, j] <= thr:
                    G.add_edge(i, j)
        if nx.is_connected(G):
            return thr, G
    return edges[-1], None


def bridge_scores(D):
    names = MODEL_NAMES
    if not HAVE_NX:
        return {n: 0.0 for n in names}, set()
    res = connectivity_threshold(D)
    if res is None or res[1] is None:
        return {n: 0.0 for n in names}, set()
    thr, G = res
    bet = nx.betweenness_centrality(G)
    arts = set(nx.articulation_points(G))
    return ({names[i]: bet.get(i, 0.0) for i in range(len(names))},
            {names[i] for i in arts})


# ------------------------------------------------------------------
# Portfolio (P3)
# ------------------------------------------------------------------
def portfolios(warned, governor_mse, total_episodes, Kmax=4):
    names = MODEL_NAMES
    rows = []
    # top-utility: best governors (lowest mse)
    gov_order = sorted(names, key=lambda n: governor_mse[n])
    # diverse: greedy marginal coverage
    for K in range(1, Kmax+1):
        tu = gov_order[:K]
        tu_cov = len(set().union(*[warned[n] for n in tu])) if tu else 0
        chosen, covered = [], set()
        for _ in range(K):
            best_n, best_gain = None, -1
            for n in names:
                if n in chosen:
                    continue
                gain = len(warned[n] - covered)
                if gain > best_gain:
                    best_gain, best_n = gain, n
            chosen.append(best_n); covered |= warned[best_n]
        rows.append({"K": K, "diverse_cov": len(covered), "top_utility_cov": tu_cov})
    return rows


# ------------------------------------------------------------------
# One seed
# ------------------------------------------------------------------
def run_seed(seed):
    if seed_done(seed):
        print(f"[seed {seed}] already complete, skipping.")
        return
    print(f"\n=== seed {seed}: retraining zoo ===")
    seed_dir = os.path.join(CACHE, f"seed_{seed:02d}")
    os.makedirs(seed_dir, exist_ok=True)

    models, val_losses = {}, {}
    for spec in ZOO:
        name, hid, regime, cls, velaux = spec
        path = os.path.join(seed_dir, f"{name}.pt")
        model = build(spec)
        if os.path.exists(path):
            ck = torch.load(path, map_location="cpu")
            model.load_state_dict(ck["state"]); vloss = ck["val"]
        else:
            rng = np.random.default_rng(seed * 100 + hash(name) % 97)
            torch.manual_seed(seed * 100 + len(name))
            data = make_dataset(rng, regime, velocity=velaux)
            model, vloss = train_model(model, data, velocity=velaux)
            torch.save({"state": model.state_dict(), "val": vloss}, path)
            del data; gc.collect()
        models[name] = model; val_losses[name] = vloss
        print(f"  {name:16s} val={vloss:.5f}")

    # stream
    srng = np.random.default_rng(seed * 100 + 7)
    frames, positions, bounds = make_stream(srng)
    idx, _ = stream_errors(models[MODEL_NAMES[0]], frames, positions)
    all_errs = {}
    for spec in ZOO:
        name, _, _, _, velaux = spec
        _, e = stream_errors(models[name], frames, positions, velocity=velaux)
        all_errs[name] = e

    # true full-pluralism MSE (mean of predictions, not mean of errors)
    full_mse = full_pluralism_mse(models, frames, positions, idx)

    vl = [val_losses[n] for n in MODEL_NAMES]
    arch, mono = arch_mse(all_errs, vl)
    arch["full_pluralism"] = full_mse

    # active series for the adaptive architecture (recompute to log sentinels against it)
    active_series = adaptive_active_series(all_errs, mono)

    warned, tp, fp, total_ep = sentinel_events(all_errs, active_series, bounds, idx)
    names, D = behavioral_distance(all_errs)
    bet, arts = bridge_scores(D)

    # write per-model rows
    for n in MODEL_NAMES:
        prec = tp[n] / (tp[n] + fp[n]) if (tp[n] + fp[n]) else 0.0
        append(RESULTS_CSV, RES_FIELDS, {
            "seed": seed, "model": n,
            "governor_mse": round(all_errs[n].mean(), 6),
            "sentinel_cov": len(warned[n]),
            "sentinel_prec": round(prec, 4),
            "betweenness": round(bet[n], 4),
            "is_articulation_any": int(n in arts),
        })
    for row in portfolios(warned, {n: all_errs[n].mean() for n in MODEL_NAMES}, total_ep):
        append(PORT_CSV, PORT_FIELDS, {"seed": seed, **row})

    print(f"  arch MSE: oracle={arch['wta_oracle']:.5f} adaptive={arch['adaptive']:.5f} "
          f"full={arch['full_pluralism']:.5f} mono={arch['monoculture']:.5f} "
          f"closed={arch['wta_closed']:.5f}")
    append(os.path.join(OUT, "arch_results.csv"),
           ["seed", "monoculture", "full_pluralism", "wta_oracle", "wta_closed", "adaptive"],
           {"seed": seed, **{k: round(v, 6) for k, v in arch.items()}})
    del models; gc.collect()


@torch.no_grad()
def full_pluralism_mse(models, frames, positions, idx):
    maxoff = max(OFFSETS)
    windows = np.stack([frames[t - INPUT_LEN:t] for t in idx])
    targets = np.stack([[positions[t + o - 1] for o in OFFSETS] for t in idx])
    xb = torch.tensor(windows)
    preds_sum = np.zeros_like(targets, dtype=np.float64)
    for spec in ZOO:
        name = spec[0]
        acc = []
        for i in range(0, len(idx), 512):
            out = models[name](xb[i:i+512])
            pp = out[0] if isinstance(out, tuple) else out
            acc.append(pp.numpy())
        preds_sum += np.concatenate(acc)
    preds = preds_sum / len(ZOO)
    return float(np.mean((preds - targets) ** 2))


def adaptive_active_series(all_errs, mono):
    names = MODEL_NAMES
    E = np.stack([all_errs[n] for n in names])
    N = E.shape[1]
    win_a, margin, persist = 50, 0.05, 5
    active, cnt = mono, 0
    series = np.empty(N, dtype=int)
    for t in range(N):
        series[t] = active
        lo = max(0, t - win_a)
        avg = E[:, lo:t+1].mean(axis=1)
        best = int(np.argmin(avg))
        if best != active and avg[best] < avg[active] * (1 - margin):
            cnt += 1
            if cnt >= persist:
                active = best; cnt = 0
        else:
            cnt = 0
    return series


# ------------------------------------------------------------------
# CSV helpers
# ------------------------------------------------------------------
def append(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def seed_done(seed):
    return any(int(r["seed"]) == seed for r in load(RESULTS_CSV))


# ------------------------------------------------------------------
# Registered analysis (P1-P4)
# ------------------------------------------------------------------
def analyze():
    res = load(RESULTS_CSV)
    port = load(PORT_CSV)
    arch = load(os.path.join(OUT, "arch_results.csv"))
    if not res:
        print("No results."); return
    seeds = sorted({int(r["seed"]) for r in res})
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("=" * 60)
    out(f"ROLE-TRIAD REPLICATION — {len(seeds)} zoos")
    out("=" * 60)

    # ---- P1 ----
    def af(r, k): return float(r[k])
    order_ok = 0; adaptive_close = 0; closed_worst = 0; adaptive_beats_mono = 0
    for a in arch:
        vals = {k: af(a, k) for k in ["monoculture", "full_pluralism",
                                      "wta_oracle", "wta_closed", "adaptive"]}
        if (vals["wta_oracle"] <= vals["adaptive"] <= vals["full_pluralism"] <=
                vals["monoculture"] and vals["wta_closed"] == max(vals.values())):
            order_ok += 1
        if abs(vals["adaptive"] - vals["wta_oracle"]) < abs(vals["adaptive"] - vals["full_pluralism"]):
            adaptive_close += 1
        if vals["wta_closed"] == max(vals.values()):
            closed_worst += 1
        if vals["adaptive"] < vals["monoculture"]:
            adaptive_beats_mono += 1
    n = len(arch)
    # P1 split into two independently-reported sub-claims.
    # P1a (registered core): adaptive tracks the oracle and beats monoculture.
    # P1b (registered ordering): closed-WTA is the worst architecture.
    p1a = adaptive_close >= 12 and adaptive_beats_mono > n/2
    p1b = closed_worst > n/2
    out(f"\nP1a (adaptive tracks oracle, beats monoculture):")
    out(f"  adaptive closer to oracle than to full: {adaptive_close}/{n} (pass >= 12)")
    out(f"  adaptive beats monoculture: {adaptive_beats_mono}/{n} (pass > {n//2})")
    out(f"  -> {'PASS' if p1a else 'FAIL'}")
    out(f"\nP1b (closed-WTA is the worst architecture):")
    out(f"  closed-WTA worst: {closed_worst}/{n} (pass > {n//2})")
    out(f"  full strict ordering held: {order_ok}/{n} (descriptive)")
    out(f"  -> {'PASS' if p1b else 'FAIL'}")

    # ---- P2 ----
    by_seed = {}
    for r in res:
        by_seed.setdefault(int(r["seed"]), []).append(r)
    diff_top = 0
    G_all, S_all = [], []
    for s, rows in by_seed.items():
        top_gov = min(rows, key=lambda r: float(r["governor_mse"]))["model"]
        top_sen = max(rows, key=lambda r: int(r["sentinel_cov"]))["model"]
        if top_gov != top_sen:
            diff_top += 1
        for r in rows:
            G_all.append(-float(r["governor_mse"]))  # higher = better governor
            S_all.append(int(r["sentinel_cov"]))
    rho, _ = spearmanr(G_all, S_all)
    out(f"\nP2 (governor != sentinel):")
    out(f"  top-gov != top-sentinel: {diff_top}/{len(by_seed)} (pass >= 14)")
    out(f"  Spearman(governor, sentinel) pooled: {rho:.3f} (pass < 0.5)")
    p2 = diff_top >= 14 and rho < 0.5
    out(f"  -> {'PASS' if p2 else 'FAIL'}")

    # ---- P3 ----
    p3_wins = 0; advantages = []
    for s in by_seed:
        r4 = [p for p in port if p.get("seed") and int(p["seed"]) == s
              and int(p.get("K", -1)) == 4]
        if not r4:
            continue
        dv = int(r4[0]["diverse_cov"]); tu = int(r4[0]["top_utility_cov"])
        advantages.append(dv - tu)
        if dv > tu:
            p3_wins += 1
    out(f"\nP3 (diverse > top-utility coverage, K=4):")
    out(f"  diverse wins: {p3_wins}/{len(advantages)} (pass >= 12)")
    out(f"  mean coverage advantage: {np.mean(advantages):.2f}")
    p3 = p3_wins >= 12 and np.mean(advantages) > 0
    out(f"  -> {'PASS' if p3 else 'FAIL'}")

    # ---- P4 ----
    top_bridge_diff = 0; nontop_articulation = 0
    for s, rows in by_seed.items():
        gov_sorted = sorted(rows, key=lambda r: float(r["governor_mse"]))
        top_gov = gov_sorted[0]["model"]
        top2 = {gov_sorted[0]["model"], gov_sorted[1]["model"]}
        top_bridge = max(rows, key=lambda r: float(r["betweenness"]))["model"]
        if top_bridge != top_gov:
            top_bridge_diff += 1
        if any(int(r["is_articulation_any"]) and r["model"] not in top2 for r in rows):
            nontop_articulation += 1
    out(f"\nP4 (governor != bridge):")
    out(f"  top-bridge != top-governor: {top_bridge_diff}/{len(by_seed)} (pass >= 14)")
    out(f"  non-top-2 governor is articulation pt: {nontop_articulation}/{len(by_seed)} (pass >= 12)")
    p4 = top_bridge_diff >= 14 and nontop_articulation >= 12
    out(f"  -> {'PASS' if p4 else 'FAIL'}")
    if not HAVE_NX:
        out("  [networkx not installed: P4 bridge metrics are zeroed — install networkx]")

    with open(os.path.join(OUT, "role_triad_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary -> {os.path.join(OUT, 'role_triad_summary.txt')}")


def backfill_portfolio(seed):
    """Recompute only portfolio rows for one seed from cached weights."""
    if any(p.get("seed") and int(p["seed"]) == seed for p in load(PORT_CSV)):
        print(f"[seed {seed}] portfolio already present, skipping.")
        return
    seed_dir = os.path.join(CACHE, f"seed_{seed:02d}")
    models = {}
    for spec in ZOO:
        name = spec[0]
        path = os.path.join(seed_dir, f"{name}.pt")
        if not os.path.exists(path):
            print(f"[seed {seed}] missing cache {name}, cannot backfill.")
            return
        m = build(spec)
        m.load_state_dict(torch.load(path, map_location="cpu")["state"])
        models[name] = m
    srng = np.random.default_rng(seed * 100 + 7)
    frames, positions, bounds = make_stream(srng)
    idx, _ = stream_errors(models[MODEL_NAMES[0]], frames, positions)
    all_errs = {}
    for spec in ZOO:
        name, _, _, _, velaux = spec
        _, e = stream_errors(models[name], frames, positions, velocity=velaux)
        all_errs[name] = e
    vl_row = {r["model"]: float(r["governor_mse"])
              for r in load(RESULTS_CSV) if int(r["seed"]) == seed}
    mono = int(np.argmin([vl_row[n] for n in MODEL_NAMES]))
    active_series = adaptive_active_series(all_errs, mono)
    warned, _, _, total_ep = sentinel_events(all_errs, active_series, bounds, idx)
    for row in portfolios(warned, {n: all_errs[n].mean() for n in MODEL_NAMES}, total_ep):
        append(PORT_CSV, PORT_FIELDS, {"seed": seed, **row})
    print(f"[seed {seed}] portfolio backfilled.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--backfill-portfolio", action="store_true",
                    help="recompute only the portfolio CSV from cached zoos")
    args = ap.parse_args()
    if args.backfill_portfolio:
        for s in sorted({int(r["seed"]) for r in load(RESULTS_CSV)}):
            backfill_portfolio(s)
        analyze()
    elif not args.analyze_only:
        for s in range(args.start, args.seeds):
            run_seed(s)
    analyze()
