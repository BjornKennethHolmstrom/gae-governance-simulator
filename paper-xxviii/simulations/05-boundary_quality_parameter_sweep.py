import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Model parameters (same as original)
# ------------------------------------------------------------
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

def deriv(x, p, s=None, theta=None, perm=0.0):
    """
    perm : boundary permeability (0..1).
           perm=0: B fully suppresses exploration/trust (original model).
           perm=1: B has no suppressive effect on exploration/trust,
                   but still reduces felt uncertainty.
    """
    U, B, T, E = x
    if s is None:
        s = p["s"]
    if theta is None:
        theta = p["theta"]

    # Felt uncertainty: B reduces it regardless of permeability
    F = s * U / ((1 + p["lam"] * T) * (1 + p["mu"] * B))

    # Effective boundary suppression factor
    block = (1 - perm) * B

    b_drive = sigmoid(p["kb"] * (F - theta))
    e_drive = sigmoid(p["ke"] * (p["alpha"] * U / (1 + p["eta"] * block) - p["cE"]))

    dU = p["n"] * (1 - U) - p["alpha"] * E * (1 - p["beta"] * block) * U
    dB = p["rhoB"] * b_drive - p["dB"] * B
    dT = (p["rhoT"] * E * (1 - p["betaT"] * block)
          - p["dT"] * T
          - p["gamma"] * block * T)
    dE = p["rhoE"] * e_drive - p["dE"] * E

    return np.array([dU, dB, dT, dE]), F


def simulate(x0, t_end=200, dt=0.05, s_override=None,
             theta_override=None, perm=0.0, p=PARAMS):
    """Deterministic simulation for given permeability."""
    times = np.arange(0, t_end + dt, dt)
    x = np.array(x0, dtype=float)
    rows = []
    for t in times:
        s = p["s"] if s_override is None else s_override
        theta = p["theta"] if theta_override is None else theta_override
        dx, F = deriv(x, p, s=s, theta=theta, perm=perm)
        rows.append([t, *x, s, theta, perm, F])
        x = np.clip(x + dt * dx, 0.0, 1.0)
    return pd.DataFrame(rows, columns=["t","U","B","T","E","s","theta","perm","F"])


# ------------------------------------------------------------
# Hysteresis sweep for different permeability values
# ------------------------------------------------------------
def hysteresis_sweep(perm, s_low=0.5, s_high=1.8, n_steps=60, t_transient=200):
    """Run deterministic up/down sweep in s for a fixed permeability."""
    s_vals = np.linspace(s_low, s_high, n_steps)
    open_init = (0.2, 0.02, 0.95, 0.90)
    closed_init = (0.8, 0.90, 0.02, 0.05)

    # Up sweep: start open at low s
    df = simulate(open_init, t_end=250, dt=0.05,
                  s_override=s_low, theta_override=PARAMS["theta"], perm=perm)
    x_start = df[["U","B","T","E"]].iloc[-1].to_numpy()
    up_B = []
    for s in s_vals:
        df = simulate(x_start, t_end=t_transient, dt=0.05,
                      s_override=s, theta_override=PARAMS["theta"], perm=perm)
        x_start = df[["U","B","T","E"]].iloc[-1].to_numpy()
        up_B.append(df.tail(100)["B"].mean())

    # Down sweep: start closed at high s
    df = simulate(closed_init, t_end=250, dt=0.05,
                  s_override=s_high, theta_override=PARAMS["theta"], perm=perm)
    x_start = df[["U","B","T","E"]].iloc[-1].to_numpy()
    down_B = []
    for s in s_vals[::-1]:
        df = simulate(x_start, t_end=t_transient, dt=0.05,
                      s_override=s, theta_override=PARAMS["theta"], perm=perm)
        x_start = df[["U","B","T","E"]].iloc[-1].to_numpy()
        down_B.append(df.tail(100)["B"].mean())

    return s_vals, np.array(up_B), np.array(down_B[::-1])  # down_B in increasing s order

# Run for three permeability values
permeabilities = [0.0, 0.5, 1.0]
results = {}
for perm in permeabilities:
    print(f"Running hysteresis sweep for perm={perm}")
    s_vals, up_B, down_B = hysteresis_sweep(perm)
    results[perm] = (s_vals, up_B, down_B)

# ------------------------------------------------------------
# Plot: hysteresis curves for different permeability
# ------------------------------------------------------------
plt.figure(figsize=(10, 7))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for perm, color in zip(permeabilities, colors):
    s_vals, up_B, down_B = results[perm]
    plt.plot(s_vals, up_B, "o-", color=color, alpha=0.7, label=f"perm={perm} (up)")
    plt.plot(s_vals, down_B, "s--", color=color, alpha=0.7, label=f"perm={perm} (down)")

plt.xlabel("s  (stakes / uncertainty multiplier)")
plt.ylabel("Boundary strength  B")
plt.title("Hysteresis for different boundary permeability")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("boundary_quality_hysteresis.png", dpi=150)
plt.show()

# Save results to CSV
csv_data = []
for perm in permeabilities:
    s_vals, up_B, down_B = results[perm]
    for i, s in enumerate(s_vals):
        csv_data.append({
            "perm": perm,
            "s": s,
            "B_up": up_B[i],
            "B_down": down_B[i],
        })
pd.DataFrame(csv_data).to_csv("boundary_quality_hysteresis_results.csv", index=False)
print("Saved boundary_quality_hysteresis_results.csv")
