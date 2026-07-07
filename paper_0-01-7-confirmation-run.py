"""
01-7-confirmation_run.py
========================
Confirmation run per 01-6-confirmation-preregistration.md:
seeds 20-39, hidden sizes {2, 3}, identical environment/model/training/probe
configuration to the first registered run (01-4-multiseed_factorization.py).

Registered predictions evaluated:
  P2'  type-structured blindness at h=2
  P2'' no uniform blur at h=2 (union metric)
  P3'  forced velocity tie-break at h=3

Appends to outputs/confirmation_results.csv (resumable; completed
(seed, hidden) combos are skipped on restart).

Usage:
    python 01-7-confirmation_run.py                  # seeds 20-39, then analyze
    python 01-7-confirmation_run.py --analyze-only
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
# Registered configuration (identical to run 1 except SEEDS / HIDDEN_SIZES)
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_TRAJECTORIES = 600
STEPS_PER_TRAJ = 200
BOX_SIZE = 1.0
DT = 0.05
RENDER_SIZE = 16
NOISE_STD = 0.1

INPUT_LEN = 20
FUTURE_OFFSETS = [5, 10, 20]
NUM_OFFSETS = len(FUTURE_OFFSETS)

BATCH_SIZE = 128
EPOCHS = 25
PATIENCE = 5
LEARNING_RATE = 1e-3

HIDDEN_SIZES = [2, 3]
SEED_START, SEED_END = 20, 40   # seeds 20..39

PROBE_SUBSET = 2000
ASYM_THRESHOLD = 0.3            # registered, both metrics
TIEBREAK_THRESHOLD = 0.25       # registered, P3' within-seed velocity asymmetry

OUTPUT_DIR = "outputs"
CSV_PATH = os.path.join(OUTPUT_DIR, "confirmation_results.csv")
CSV_FIELDS = ["seed", "hidden_size", "best_val_loss", "epochs_run",
              "r2_x", "r2_y", "r2_vx", "r2_vy",
              "A_x", "A_y", "axis_asym",
              "A_pos", "A_vel", "type_asym", "mode"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------
# Environment / dataset (identical to run 1)
# ------------------------------
def generate_bouncing_dot_sequence(rng, num_steps=200):
    x = rng.random() * BOX_SIZE
    y = rng.random() * BOX_SIZE
    vx = (rng.random() - 0.5) * 2.0
    vy = (rng.random() - 0.5) * 2.0
    frames, states = [], []
    for _ in range(num_steps):
        x += vx * DT
        y += vy * DT
        if x <= 0 or x >= BOX_SIZE:
            vx *= -1
            x = float(np.clip(x, 0, BOX_SIZE))
        if y <= 0 or y >= BOX_SIZE:
            vy *= -1
            y = float(np.clip(y, 0, BOX_SIZE))
        img = np.zeros((RENDER_SIZE, RENDER_SIZE))
        px = int(np.clip(int(x / BOX_SIZE * (RENDER_SIZE - 1)), 0, RENDER_SIZE - 1))
        py = int(np.clip(int(y / BOX_SIZE * (RENDER_SIZE - 1)), 0, RENDER_SIZE - 1))
        img[max(0, px - 1):min(RENDER_SIZE, px + 2),
            max(0, py - 1):min(RENDER_SIZE, py + 2)] = 1.0
        img += rng.standard_normal((RENDER_SIZE, RENDER_SIZE)) * NOISE_STD
        img = np.clip(img, 0, 1)
        frames.append(img.flatten())
        states.append([x, y, vx, vy])
    return (np.array(frames, dtype=np.float32),
            np.array(states, dtype=np.float32))


def generate_dataset(rng):
    fs, ss = [], []
    for _ in range(NUM_TRAJECTORIES):
        f, s = generate_bouncing_dot_sequence(rng, num_steps=STEPS_PER_TRAJ)
        fs.append(f)
        ss.append(s)
    return np.array(fs, dtype=np.float32), np.array(ss, dtype=np.float32)


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
# Model / training / probes (identical to run 1)
# ------------------------------
class FuturePredictorGRU(nn.Module):
    def __init__(self, input_dim=RENDER_SIZE * RENDER_SIZE, hidden_dim=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, 2 * NUM_OFFSETS)

    def forward(self, x, h0=None):
        b = x.shape[0]
        if h0 is None:
            h0 = torch.zeros(1, b, self.hidden_dim, device=x.device)
        _, h_final = self.gru(x, h0)
        preds = self.decoder(h_final.squeeze(0)).view(b, -1, 2)
        return preds, h_final


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
    hs, ss = [], []
    for x, _, s_last in loader:
        _, h = model(x.to(DEVICE))
        hs.append(h.squeeze(0).cpu().numpy())
        ss.append(s_last.numpy())
    return np.concatenate(hs), np.concatenate(ss)


def compute_r2_scores(h, s):
    out = {}
    for i, var in enumerate(["x", "y", "vx", "vy"]):
        reg = LinearRegression().fit(h, s[:, i])
        out[var] = r2_score(s[:, i], reg.predict(h))
    return out


def metrics(r2):
    """Registered axis + type metrics; mode classification."""
    c = {k: max(0.0, v) for k, v in r2.items()}
    A_x = (c["x"] + c["vx"]) / 2.0
    A_y = (c["y"] + c["vy"]) / 2.0
    A_pos = (c["x"] + c["y"]) / 2.0
    A_vel = (c["vx"] + c["vy"]) / 2.0
    axis_asym = abs(A_x - A_y)
    type_asym = abs(A_pos - A_vel)
    # Larger-asymmetry-wins (registered): descriptive mode only; the P2'/P2''
    # decisions use the raw metrics, not this label.
    if max(axis_asym, type_asym) <= ASYM_THRESHOLD:
        mode = "uniform"
    elif axis_asym >= type_asym:
        mode = "axis"
    else:
        mode = "type"
    return A_x, A_y, axis_asym, A_pos, A_vel, type_asym, mode


# ------------------------------
# CSV persistence (resumable)
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
# One seed
# ------------------------------
def run_seed(seed):
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

    for hid in HIDDEN_SIZES:
        if is_done(seed, hid):
            print(f"[seed {seed}] hidden={hid} already in CSV, skipping.")
            continue
        model = FuturePredictorGRU(hidden_dim=hid).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        best_val, best_state, since_best, epochs_run = float("inf"), None, 0, 0
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
        A_x, A_y, axis_asym, A_pos, A_vel, type_asym, mode = metrics(r2)

        row = {"seed": seed, "hidden_size": hid,
               "best_val_loss": round(best_val, 6), "epochs_run": epochs_run,
               "r2_x": round(r2["x"], 4), "r2_y": round(r2["y"], 4),
               "r2_vx": round(r2["vx"], 4), "r2_vy": round(r2["vy"], 4),
               "A_x": round(A_x, 4), "A_y": round(A_y, 4),
               "axis_asym": round(axis_asym, 4),
               "A_pos": round(A_pos, 4), "A_vel": round(A_vel, 4),
               "type_asym": round(type_asym, 4), "mode": mode}
        append_row(row)
        print(f"[seed {seed}] hidden={hid}: val={best_val:.5f} "
              f"R2(x,y,vx,vy)=({r2['x']:.2f},{r2['y']:.2f},{r2['vx']:.2f},{r2['vy']:.2f}) "
              f"axis={axis_asym:.2f} type={type_asym:.2f} mode={mode}")


# ------------------------------
# Registered analysis
# ------------------------------
def analyze():
    rows = load_rows()
    if not rows:
        print("No results found.")
        return
    for r in rows:
        for k in ["r2_x", "r2_y", "r2_vx", "r2_vy", "axis_asym", "type_asym",
                  "A_pos", "A_vel", "best_val_loss"]:
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
    out("CONFIRMATION RUN SUMMARY (seeds 20-39)")
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

    out("\n" + "=" * 64)
    out("REGISTERED PREDICTIONS (01-6-confirmation-preregistration.md)")
    out("=" * 64)

    h2 = [r for r in rows if r["hidden_size"] == 2]
    if h2:
        n = len(h2)
        n_type = sum(1 for r in h2 if r["type_asym"] > ASYM_THRESHOLD)
        out(f"\nP2' (type-structured blindness, h=2): {n_type}/{n} seeds "
            f"with type_asym > {ASYM_THRESHOLD} ({100*n_type/n:.0f}%)")
        if n_type / n >= 0.8:
            verdict = "PASS"
        elif n_type / n < 0.6:
            verdict = "FAIL (null triggered: < 60%)"
        else:
            verdict = "FAIL (between null and pass thresholds)"
        out(f"  -> {verdict}")

        n_struct = sum(1 for r in h2
                       if max(r["axis_asym"], r["type_asym"]) > ASYM_THRESHOLD)
        out(f"\nP2'' (no uniform blur, h=2): {n_struct}/{n} seeds structured "
            f"under union metric ({100*n_struct/n:.0f}%)")
        out(f"  -> {'PASS' if n_struct / n >= 0.9 else 'FAIL'} (threshold: >= 90%)")

        modes = {}
        for r in h2:
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
        out(f"\nExploratory (declared): h=2 mode counts this run: {modes}")

    h3 = [r for r in rows if r["hidden_size"] == 3]
    if h3:
        n = len(h3)
        med_x = np.median([r["r2_x"] for r in h3])
        med_y = np.median([r["r2_y"] for r in h3])
        c1 = med_x >= 0.7 and med_y >= 0.7
        out(f"\nP3' (forced tie-break, h=3):")
        out(f"  (1) positions recovered: median r2_x={med_x:.3f}, r2_y={med_y:.3f} "
            f"-> {'PASS' if c1 else 'FAIL'}")

        choosing = [r for r in h3
                    if abs(max(0, r["r2_vx"]) - max(0, r["r2_vy"])) > TIEBREAK_THRESHOLD]
        frac = len(choosing) / n
        c2 = frac >= 0.6
        out(f"  (2) within-seed velocity asymmetry > {TIEBREAK_THRESHOLD}: "
            f"{len(choosing)}/{n} ({100*frac:.0f}%) -> {'PASS' if c2 else 'FAIL'}")

        fav_vx = sum(1 for r in choosing if max(0, r["r2_vx"]) > max(0, r["r2_vy"]))
        fav_vy = len(choosing) - fav_vx
        out(f"  (3) favored velocity varies: vx favored {fav_vx}x, vy favored {fav_vy}x "
            f"(of {len(choosing)} choosing seeds)")
        if len(choosing) == 0:
            out("      -> NOT EVALUABLE")
        else:
            c3 = fav_vx >= 3 and fav_vy >= 3
            dominant = max(fav_vx, fav_vy) / len(choosing)
            if c3:
                out("      -> PASS")
            elif dominant >= 0.9:
                out("      -> FAIL (one velocity >= 90%: hidden asymmetry suspected; audit code)")
            else:
                out("      -> FAIL (threshold: each velocity favored >= 3x)")

    with open(os.path.join(OUTPUT_DIR, "confirmation_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Plots
    if h2:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        colors = {"type": "#4477aa", "axis": "#ee6677", "uniform": "#bbbbbb"}
        for r in h2:
            ax.scatter(r["axis_asym"], r["type_asym"], c=colors[r["mode"]])
        ax.axvline(ASYM_THRESHOLD, color="gray", ls="--", lw=0.8)
        ax.axhline(ASYM_THRESHOLD, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("axis-asym")
        ax.set_ylabel("type-asym")
        ax.set_title("h=2: structured blindness, two cuts")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "confirmation_h2_modes.png"), dpi=150)
        plt.close(fig)
    if h3:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter([max(0, r["r2_vx"]) for r in h3],
                   [max(0, r["r2_vy"]) for r in h3], c="black")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
        ax.plot([TIEBREAK_THRESHOLD, 1], [0, 1 - TIEBREAK_THRESHOLD],
                ":", color="gray", lw=0.8)
        ax.plot([0, 1 - TIEBREAK_THRESHOLD], [TIEBREAK_THRESHOLD, 1],
                ":", color="gray", lw=0.8)
        ax.set_xlabel("R² vx")
        ax.set_ylabel("R² vy")
        ax.set_title("h=3: which velocity gets the third slot?")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "confirmation_h3_tiebreak.png"), dpi=150)
        plt.close(fig)

    print(f"\nSummary written to {os.path.join(OUTPUT_DIR, 'confirmation_summary.txt')}")


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    print(f"Using device: {DEVICE}")
    if not args.analyze_only:
        for seed in range(SEED_START, SEED_END):
            run_seed(seed)
    analyze()
