"""
01-4-multiseed_factorization.py
===============================
Multi-seed replication of the capacity-sweep factorization experiment
(01-1-minimal_passive_factorization.py), per the registered protocol in
01-3-multiseed-preregistration.md.

Each seed re-generates the environment, the train/val split, and the model
initialization. Results are appended to outputs/multiseed_results.csv after
each (seed, hidden_size) combination, so the run can be interrupted and
resumed: completed combinations are skipped on restart.

Usage:
    python 01-4-multiseed_factorization.py                 # run seeds 0-19, then analyze
    python 01-4-multiseed_factorization.py --seeds 5       # quick partial run (seeds 0-4)
    python 01-4-multiseed_factorization.py --start-seed 10 --seeds 20   # seeds 10-19
    python 01-4-multiseed_factorization.py --analyze-only  # re-run analysis on existing CSV

Author: Claude / Björn (adapted from DeepSeek / Björn original)
"""

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ------------------------------
# 0. Registered configuration (do not change after preregistration)
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data (reduced from pilot's 2000 trajectories to make 20 seeds tractable)
NUM_TRAJECTORIES = 600
STEPS_PER_TRAJ = 200
BOX_SIZE = 1.0
DT = 0.05
RENDER_SIZE = 16
NOISE_STD = 0.1

# Task
INPUT_LEN = 20
FUTURE_OFFSETS = [5, 10, 20]
NUM_OFFSETS = len(FUTURE_OFFSETS)

# Training
BATCH_SIZE = 128
EPOCHS = 25
PATIENCE = 5           # early stopping on val loss
LEARNING_RATE = 1e-3

# Sweep
HIDDEN_SIZES = [2, 4, 8, 16]
DEFAULT_NUM_SEEDS = 20

# Analysis
PROBE_SUBSET = 2000    # validation samples used for linear probes
ASYM_THRESHOLD = 0.3   # registered threshold for "structured blindness"

OUTPUT_DIR = "outputs"
CSV_PATH = os.path.join(OUTPUT_DIR, "multiseed_results.csv")
CSV_FIELDS = ["seed", "hidden_size", "best_val_loss", "epochs_run",
              "r2_x", "r2_y", "r2_vx", "r2_vy",
              "A_x", "A_y", "asym", "dropped_axis"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------
# 1. Environment (identical to pilot)
# ------------------------------
def generate_bouncing_dot_sequence(rng, num_steps=200, box_size=1.0, dt=0.05,
                                   noise_std=0.1, render_size=16):
    x = rng.random() * box_size
    y = rng.random() * box_size
    vx = (rng.random() - 0.5) * 2.0
    vy = (rng.random() - 0.5) * 2.0

    frames, states = [], []
    for _ in range(num_steps):
        x += vx * dt
        y += vy * dt
        if x <= 0 or x >= box_size:
            vx *= -1
            x = float(np.clip(x, 0, box_size))
        if y <= 0 or y >= box_size:
            vy *= -1
            y = float(np.clip(y, 0, box_size))

        img = np.zeros((render_size, render_size))
        px = int(np.clip(int(x / box_size * (render_size - 1)), 0, render_size - 1))
        py = int(np.clip(int(y / box_size * (render_size - 1)), 0, render_size - 1))
        img[max(0, px - 1):min(render_size, px + 2),
            max(0, py - 1):min(render_size, py + 2)] = 1.0
        img += rng.standard_normal((render_size, render_size)) * noise_std
        img = np.clip(img, 0, 1)
        frames.append(img.flatten())
        states.append([x, y, vx, vy])

    return (np.array(frames, dtype=np.float32),
            np.array(states, dtype=np.float32))


def generate_dataset(rng):
    all_frames, all_states = [], []
    for _ in range(NUM_TRAJECTORIES):
        f, s = generate_bouncing_dot_sequence(
            rng, num_steps=STEPS_PER_TRAJ, box_size=BOX_SIZE, dt=DT,
            noise_std=NOISE_STD, render_size=RENDER_SIZE)
        all_frames.append(f)
        all_states.append(s)
    return (np.array(all_frames, dtype=np.float32),
            np.array(all_states, dtype=np.float32))


def prepare_dataset_with_state(frames, states):
    N, T, _ = frames.shape
    max_offset = max(FUTURE_OFFSETS)
    X, Y, S_last = [], [], []
    for n in range(N):
        for start in range(T - INPUT_LEN - max_offset):
            end = start + INPUT_LEN
            future_pos = [states[n, end + off - 1, :2] for off in FUTURE_OFFSETS]
            X.append(frames[n, start:end])
            Y.append(np.stack(future_pos, axis=0))
            S_last.append(states[n, end - 1, :])
    return (torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(Y), dtype=torch.float32),
            torch.tensor(np.array(S_last), dtype=torch.float32))


# ------------------------------
# 2. Model (identical to pilot)
# ------------------------------
class FuturePredictorGRU(nn.Module):
    def __init__(self, input_dim=RENDER_SIZE * RENDER_SIZE, hidden_dim=4,
                 output_dim=2, num_offsets=NUM_OFFSETS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim * num_offsets)

    def forward(self, x, h0=None):
        batch_size = x.shape[0]
        if h0 is None:
            h0 = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)
        _, h_final = self.gru(x, h0)
        preds = self.decoder(h_final.squeeze(0)).view(batch_size, -1, 2)
        return preds, h_final


# ------------------------------
# 3. Train / probe utilities
# ------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total = 0.0
    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds, _ = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total = 0.0
    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds, _ = model(x)
        total += loss_fn(preds, y).item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def get_hidden_states(model, loader):
    model.eval()
    h_list, s_list = [], []
    for x, _, s_last in loader:
        x = x.to(DEVICE)
        _, h = model(x)
        h_list.append(h.squeeze(0).cpu().numpy())
        s_list.append(s_last.numpy())
    return np.concatenate(h_list), np.concatenate(s_list)


def compute_r2_scores(h, s):
    scores = {}
    for i, var in enumerate(["x", "y", "vx", "vy"]):
        reg = LinearRegression().fit(h, s[:, i])
        scores[var] = r2_score(s[:, i], reg.predict(h))
    return scores


def axis_metrics(r2):
    """Registered definitions: clip negative R2 to 0, per-axis recovery, asymmetry."""
    c = {k: max(0.0, v) for k, v in r2.items()}
    A_x = (c["x"] + c["vx"]) / 2.0
    A_y = (c["y"] + c["vy"]) / 2.0
    asym = abs(A_x - A_y)
    if asym > ASYM_THRESHOLD:
        dropped = "y" if A_y < A_x else "x"
    else:
        dropped = "uniform"
    return A_x, A_y, asym, dropped


# ------------------------------
# 4. One (seed, hidden) run
# ------------------------------
def run_seed(seed):
    """Runs all hidden sizes for one seed. Returns list of result dicts."""
    # Seed everything: environment, split, torch init
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    print(f"\n[seed {seed}] generating dataset...")
    frames, states = generate_dataset(rng)
    X_all, Y_all, S_all = prepare_dataset_with_state(frames, states)
    n = len(X_all)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    split = int(0.8 * n)
    tr, va = perm[:split], perm[split:]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_all[tr], Y_all[tr], S_all[tr]),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_all[va], Y_all[va], S_all[va]),
        batch_size=BATCH_SIZE, shuffle=False)

    probe_idx = va[-PROBE_SUBSET:]
    probe_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_all[probe_idx], Y_all[probe_idx], S_all[probe_idx]),
        batch_size=BATCH_SIZE, shuffle=False)

    loss_fn = nn.MSELoss()
    results = []

    for hid in HIDDEN_SIZES:
        if is_done(seed, hid):
            print(f"[seed {seed}] hidden={hid} already in CSV, skipping.")
            continue
        model = FuturePredictorGRU(hidden_dim=hid).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val, best_state, since_best = float("inf"), None, 0
        epochs_run = 0
        for epoch in range(EPOCHS):
            train_one_epoch(model, train_loader, optimizer, loss_fn)
            val_loss = evaluate(model, val_loader, loss_fn)
            epochs_run = epoch + 1
            if val_loss < best_val:
                best_val, best_state, since_best = val_loss, model.state_dict(), 0
            else:
                since_best += 1
                if since_best >= PATIENCE:
                    break
        model.load_state_dict(best_state)

        h_val, s_val = get_hidden_states(model, probe_loader)
        r2 = compute_r2_scores(h_val, s_val)
        A_x, A_y, asym, dropped = axis_metrics(r2)

        row = {"seed": seed, "hidden_size": hid,
               "best_val_loss": round(best_val, 6), "epochs_run": epochs_run,
               "r2_x": round(r2["x"], 4), "r2_y": round(r2["y"], 4),
               "r2_vx": round(r2["vx"], 4), "r2_vy": round(r2["vy"], 4),
               "A_x": round(A_x, 4), "A_y": round(A_y, 4),
               "asym": round(asym, 4), "dropped_axis": dropped}
        append_row(row)
        results.append(row)
        print(f"[seed {seed}] hidden={hid}: val={best_val:.5f} "
              f"R2(x,y,vx,vy)=({r2['x']:.2f},{r2['y']:.2f},{r2['vx']:.2f},{r2['vy']:.2f}) "
              f"asym={asym:.2f} dropped={dropped}")
    return results


# ------------------------------
# 5. CSV persistence (resumable)
# ------------------------------
def append_row(row):
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def load_rows():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH) as f:
        return [dict(r) for r in csv.DictReader(f)]


def is_done(seed, hid):
    return any(int(r["seed"]) == seed and int(r["hidden_size"]) == hid
               for r in load_rows())


# ------------------------------
# 6. Registered analysis
# ------------------------------
def analyze():
    rows = load_rows()
    if not rows:
        print("No results found.")
        return
    for r in rows:
        for k in ["r2_x", "r2_y", "r2_vx", "r2_vy", "A_x", "A_y", "asym", "best_val_loss"]:
            r[k] = float(r[k])
        r["seed"] = int(r["seed"])
        r["hidden_size"] = int(r["hidden_size"])

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    def med_iqr(vals):
        v = np.array(vals)
        return np.median(v), np.percentile(v, 25), np.percentile(v, 75)

    out("=" * 64)
    out("MULTI-SEED SUMMARY (median [IQR] across seeds)")
    out("=" * 64)
    for hid in HIDDEN_SIZES:
        sub = [r for r in rows if r["hidden_size"] == hid]
        if not sub:
            continue
        out(f"\nHidden {hid}  (n={len(sub)} seeds)")
        m, lo, hi = med_iqr([r["best_val_loss"] for r in sub])
        out(f"  val_loss: {m:.5f} [{lo:.5f}, {hi:.5f}]")
        for var in ["r2_x", "r2_y", "r2_vx", "r2_vy"]:
            m, lo, hi = med_iqr([r[var] for r in sub])
            out(f"  {var:5s}: {m:.3f} [{lo:.3f}, {hi:.3f}]")

    # ---- P1: emergence at h=8 ----
    out("\n" + "=" * 64)
    out("REGISTERED PREDICTIONS")
    out("=" * 64)
    h8 = [r for r in rows if r["hidden_size"] == 8]
    if h8:
        meds = {v: np.median([r[v] for r in h8]) for v in ["r2_x", "r2_y", "r2_vx", "r2_vy"]}
        p1 = all(m >= 0.7 for m in meds.values())
        p1_null = any(m < 0.5 for m in meds.values())
        out(f"\nP1 (emergence, h=8): medians = "
            + ", ".join(f"{k}={v:.3f}" for k, v in meds.items()))
        out(f"  -> {'PASS' if p1 else ('FAIL (null triggered)' if p1_null else 'PARTIAL (between 0.5 and 0.7)')}")

    # ---- P2: structured blindness at h=2 ----
    h2 = [r for r in rows if r["hidden_size"] == 2]
    if h2:
        n_struct = sum(1 for r in h2 if r["asym"] > ASYM_THRESHOLD)
        frac = n_struct / len(h2)
        out(f"\nP2 (structured blindness, h=2): {n_struct}/{len(h2)} seeds "
            f"with asym > {ASYM_THRESHOLD} ({100*frac:.0f}%)")
        out(f"  -> {'PASS' if frac >= 0.8 else 'FAIL'} (threshold: >= 80%)")

        # ---- P3: symmetry breaking ----
        droppers = [r["dropped_axis"] for r in h2 if r["dropped_axis"] != "uniform"]
        n_x = droppers.count("x")
        n_y = droppers.count("y")
        out(f"\nP3 (symmetry breaking, h=2): dropped x in {n_x}, dropped y in {n_y} "
            f"(of {len(droppers)} dropping seeds)")
        if len(droppers) == 0:
            out("  -> NOT EVALUABLE (no dropping seeds)")
        else:
            p3 = n_x >= 3 and n_y >= 3
            dominant = max(n_x, n_y) / len(droppers)
            p3_null = dominant >= 0.9
            out(f"  -> {'PASS' if p3 else ('FAIL (one axis >= 90%: asymmetry suspected)' if p3_null else 'FAIL (threshold: each axis >= 3)')}")

    # ---- P4: diminishing abstraction (secondary) ----
    h16 = [r for r in rows if r["hidden_size"] == 16]
    if h8 and h16:
        vl8 = np.median([r["best_val_loss"] for r in h8])
        vl16 = np.median([r["best_val_loss"] for r in h16])
        loss_better = vl16 < vl8
        no_r2_gain = all(
            np.median([r[v] for r in h16]) <= np.median([r[v] for r in h8]) + 0.05
            for v in ["r2_x", "r2_y", "r2_vx", "r2_vy"])
        out(f"\nP4 (secondary, h=16 vs h=8): median val_loss {vl16:.5f} vs {vl8:.5f}; "
            f"loss improved: {loss_better}; no latent R2 gain > 0.05: {no_r2_gain}")
        out(f"  -> {'PASS' if (loss_better and no_r2_gain) else 'FAIL'}")

    with open(os.path.join(OUTPUT_DIR, "multiseed_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- Plots ----
    # Boxplots of R2 per variable per hidden size
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, var in zip(axes, ["r2_x", "r2_y", "r2_vx", "r2_vy"]):
        data = [[r[var] for r in rows if r["hidden_size"] == h] for h in HIDDEN_SIZES]
        ax.boxplot(data, tick_labels=[str(h) for h in HIDDEN_SIZES])
        ax.set_title(var)
        ax.set_xlabel("hidden size")
        ax.axhline(0.7, color="gray", ls="--", lw=0.8)
    axes[0].set_ylabel("R² (linear probe)")
    fig.suptitle("Latent recovery across seeds")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "multiseed_r2_boxplots.png"), dpi=150)
    plt.close(fig)

    # Dropped-axis histogram + A_x vs A_y scatter at h=2
    if h2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        cats = ["x", "y", "uniform"]
        counts = [sum(1 for r in h2 if r["dropped_axis"] == c) for c in cats]
        ax1.bar(cats, counts, color=["#4477aa", "#ee6677", "#bbbbbb"])
        ax1.set_title("Dropped axis at h=2")
        ax1.set_ylabel("seeds")
        ax2.scatter([r["A_x"] for r in h2], [r["A_y"] for r in h2], c="black")
        lim = max([r["A_x"] for r in h2] + [r["A_y"] for r in h2] + [1.0])
        ax2.plot([0, lim], [0, lim], "--", color="gray", lw=0.8)
        ax2.set_xlabel("A_x (recovery of x-axis subspace)")
        ax2.set_ylabel("A_y (recovery of y-axis subspace)")
        ax2.set_title("Per-axis recovery at h=2 (diagonal = uniform)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "multiseed_h2_symmetry.png"), dpi=150)
        plt.close(fig)

    print(f"\nSummary written to {os.path.join(OUTPUT_DIR, 'multiseed_summary.txt')}")
    print("Plots: multiseed_r2_boxplots.png, multiseed_h2_symmetry.png")


# ------------------------------
# 7. Main
# ------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=DEFAULT_NUM_SEEDS,
                    help="run seeds in [start_seed, seeds)")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    print(f"Using device: {DEVICE}")
    if not args.analyze_only:
        for seed in range(args.start_seed, args.seeds):
            run_seed(seed)
    analyze()
