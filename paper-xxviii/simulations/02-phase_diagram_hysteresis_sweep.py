import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PARAMS = dict(
    n=0.120, alpha=1.339, beta=0.539,
    s=0.908, lam=2.700, mu=1.841,
    theta=0.196, kb=23.591,
    rhoB=0.154, dB=0.117,
    rhoT=0.546, betaT=0.766, dT=0.067, gamma=0.110,
    rhoE=0.073, eta=2.065, cE=0.489, ke=24.382, dE=0.059
)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

def deriv(x, p, s=None, theta=None):
    U, B, T, E = x
    if s is None:
        s = p["s"]
    if theta is None:
        theta = p["theta"]

    F = s * U / ((1 + p["lam"] * T) * (1 + p["mu"] * B))

    b_drive = sigmoid(p["kb"] * (F - theta))
    e_drive = sigmoid(
        p["ke"] * (p["alpha"] * U / (1 + p["eta"] * B) - p["cE"])
    )

    dU = p["n"] * (1 - U) - p["alpha"] * E * (1 - p["beta"] * B) * U
    dB = p["rhoB"] * b_drive - p["dB"] * B
    dT = (
        p["rhoT"] * E * (1 - p["betaT"] * B)
        - p["dT"] * T
        - p["gamma"] * B * T
    )
    dE = p["rhoE"] * e_drive - p["dE"] * E

    return np.array([dU, dB, dT, dE]), F


def simulate(x0, t_end=200, dt=0.05, s_override=None, theta_override=None, p=PARAMS):
    times = np.arange(0, t_end + dt, dt)
    x = np.array(x0, dtype=float)
    rows = []

    for t in times:
        s = p["s"] if s_override is None else s_override
        theta = p["theta"] if theta_override is None else theta_override
        dx, F = deriv(x, p, s=s, theta=theta)
        rows.append([t, *x, s, theta, F])
        x = np.clip(x + dt * dx, 0.0, 1.0)

    return pd.DataFrame(
        rows, columns=["t", "U", "B", "T", "E", "s", "theta", "F"]
    )


# ----------------------------------------------------------------------
# 1. Phase diagram over (s, theta)
# ----------------------------------------------------------------------

def classify(df, tail=200, osc_thresh=0.05):
    tail_df = df.tail(tail)
    B_mean = tail_df["B"].mean()
    B_std = tail_df["B"].std()

    if B_std > osc_thresh:
        return "oscillatory"
    if B_mean > 0.55:
        return "closed"
    elif B_mean < 0.20:
        return "open"
    else:
        return "intermediate"


open_init = (0.20, 0.02, 0.95, 0.90)
closed_init = (0.80, 0.90, 0.02, 0.05)

s_vals = np.linspace(0.40, 1.60, 25)
theta_vals = np.linspace(0.08, 0.32, 25)

regime_grid = np.empty((len(theta_vals), len(s_vals)), dtype=object)
# For CSV: store classifications and final means
phase_data = []

print("Running phase diagram sweep...")
for i, theta in enumerate(theta_vals):
    for j, s in enumerate(s_vals):
        df_open = simulate(open_init, t_end=180, dt=0.05,
                           s_override=s, theta_override=theta)
        df_closed = simulate(closed_init, t_end=180, dt=0.05,
                             s_override=s, theta_override=theta)

        c_open = classify(df_open)
        c_closed = classify(df_closed)

        if c_open != c_closed and "oscillatory" not in (c_open, c_closed):
            regime_grid[i, j] = "bistable"
        elif "oscillatory" in (c_open, c_closed):
            regime_grid[i, j] = "oscillatory"
        else:
            regime_grid[i, j] = c_open

        # Additional details for CSV
        final_B_open = df_open.tail(200)["B"].mean()
        final_B_closed = df_closed.tail(200)["B"].mean()
        phase_data.append({
            "theta": theta,
            "s": s,
            "open_regime": c_open,
            "closed_regime": c_closed,
            "final_regime": regime_grid[i, j],
            "B_open": final_B_open,
            "B_closed": final_B_closed,
        })

phase_df = pd.DataFrame(phase_data)
phase_df.to_csv("phase_diagram_results.csv", index=False)
print("Saved phase_diagram_results.csv")

# Plot
regime_to_int = {
    "open": 0,
    "intermediate": 1,
    "closed": 2,
    "bistable": 3,
    "oscillatory": 4,
}
grid_int = np.vectorize(lambda x: regime_to_int[x])(regime_grid)

cmap = ListedColormap([
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#ff7f0e",
])

plt.figure(figsize=(9, 7))
plt.imshow(grid_int,
           extent=[s_vals[0], s_vals[-1], theta_vals[0], theta_vals[-1]],
           origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=4)
cbar = plt.colorbar(ticks=range(5))
cbar.ax.set_yticklabels(["open", "intermediate", "closed", "bistable", "oscillatory"])
plt.xlabel("s  (stakes / uncertainty multiplier)")
plt.ylabel("theta  (felt-uncertainty tolerance threshold)")
plt.title("Phase diagram: closure–adaptation regimes")
plt.tight_layout()
plt.savefig("phase_diagram.png", dpi=150)
plt.show()


# ----------------------------------------------------------------------
# 2. Hysteresis sweep: slowly increase then decrease s
# ----------------------------------------------------------------------

def sweep(s_values, start_state, p, t_transient=200, dt=0.05):
    x = np.array(start_state, dtype=float)
    records = []

    for s in s_values:
        df = simulate(x, t_end=t_transient, dt=dt,
                      s_override=s, theta_override=p["theta"])
        x = df[["U", "B", "T", "E"]].iloc[-1].to_numpy(dtype=float)
        mean_B = df.tail(100)["B"].mean()
        records.append((s, x.copy(), mean_B))

    return records


s_low = 0.50
s_high = 1.80
n_steps = 60
s_up = np.linspace(s_low, s_high, n_steps)
s_down = s_up[::-1]

df_open_low = simulate(open_init, t_end=250, dt=0.05,
                       s_override=s_low, theta_override=PARAMS["theta"])
x_start_up = df_open_low[["U", "B", "T", "E"]].iloc[-1].to_numpy(dtype=float)

df_closed_high = simulate(closed_init, t_end=250, dt=0.05,
                          s_override=s_high, theta_override=PARAMS["theta"])
x_start_down = df_closed_high[["U", "B", "T", "E"]].iloc[-1].to_numpy(dtype=float)

print("Running upward s sweep...")
up_records = sweep(s_up, x_start_up, PARAMS)

print("Running downward s sweep...")
down_records = sweep(s_down, x_start_down, PARAMS)

s_up_vals = np.array([r[0] for r in up_records])
B_up_vals = np.array([r[2] for r in up_records])
s_down_vals = np.array([r[0] for r in down_records])
B_down_vals = np.array([r[2] for r in down_records])

# Save hysteresis data
hyst_df = pd.DataFrame({
    "s_up": s_up_vals,
    "B_up": B_up_vals,
    "s_down": s_down_vals,
    "B_down": B_down_vals,
})
hyst_df.to_csv("hysteresis_results.csv", index=False)
print("Saved hysteresis_results.csv")

# Plot
plt.figure(figsize=(8, 6))
plt.plot(s_up_vals, B_up_vals, "o-", color="#1f77b4",
         label="up sweep (open start)")
plt.plot(s_down_vals, B_down_vals, "s-", color="#d62728",
         label="down sweep (closed start)")
plt.xlabel("s  (stakes / uncertainty multiplier)")
plt.ylabel("Boundary strength  B")
plt.title("Hysteresis: closure does not reverse symmetrically")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hysteresis_sweep.png", dpi=150)
plt.show()

# Hysteresis gap at midpoint
s_mid = (s_low + s_high) / 2
idx_mid = np.argmin(np.abs(s_up_vals - s_mid))
up_B_mid = B_up_vals[idx_mid]
down_B_mid = B_down_vals[idx_mid]
print(f"\nAt s ≈ {s_mid:.2f}:")
print(f"  up sweep   B = {up_B_mid:.3f}")
print(f"  down sweep B = {down_B_mid:.3f}")
print(f"  hysteresis gap = {down_B_mid - up_B_mid:.3f}")
