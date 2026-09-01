import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARAMS = dict(
    n=0.120, alpha=1.339, beta=0.539,
    s=0.908, lam=2.700, mu=1.841,
    theta=0.196, kb=23.591,
    rhoB=0.154, dB=0.117,
    rhoT=0.546, betaT=0.766, dT=0.067, gamma=0.110,
    rhoE=0.073, eta=2.065, cE=0.489, ke=24.382, dE=0.059,
    # New dynamic permeability parameters
    rhoP=0.02,      # slow adaptation rate
    kP=20.0,        # sharpness of permeability response
    thetaP=0.15,    # felt uncertainty threshold for losing permeability
)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

def deriv(x, p, s=None, theta=None):
    U, B, T, E, P = x
    if s is None:
        s = p["s"]
    if theta is None:
        theta = p["theta"]

    # Felt uncertainty
    F = s * U / ((1 + p["lam"] * T) * (1 + p["mu"] * B))

    # Suppressive block = (1-P) * B
    block = (1.0 - P) * B

    b_drive = sigmoid(p["kb"] * (F - theta))
    e_drive = sigmoid(p["ke"] * (p["alpha"] * U / (1 + p["eta"] * block) - p["cE"]))

    dU = p["n"] * (1 - U) - p["alpha"] * E * (1 - p["beta"] * block) * U
    dB = p["rhoB"] * b_drive - p["dB"] * B
    dT = p["rhoT"] * E * (1 - p["betaT"] * block) - p["dT"] * T - p["gamma"] * block * T
    dE = p["rhoE"] * e_drive - p["dE"] * E

    # Dynamic permeability: tends to drop when felt uncertainty is high
    desired_P = 1.0 - sigmoid(p["kP"] * (F - p["thetaP"]))
    dP = p["rhoP"] * (desired_P - P)

    return np.array([dU, dB, dT, dE, dP]), F


def simulate(x0, t_end=250, dt=0.05, s_override=None, theta_override=None, p=PARAMS):
    times = np.arange(0, t_end + dt, dt)
    x = np.array(x0, dtype=float)
    rows = []
    for t in times:
        s = p["s"] if s_override is None else s_override
        theta = p["theta"] if theta_override is None else theta_override
        dx, F = deriv(x, p, s=s, theta=theta)
        rows.append([t, *x, s, theta, F])
        x = np.clip(x + dt * dx, 0.0, 1.0)
    return pd.DataFrame(rows, columns=["t","U","B","T","E","P","s","theta","F"])


# Hysteresis sweep with dynamic P
def hysteresis_sweep_dynamic(s_low=0.5, s_high=1.8, n_steps=60, t_transient=250):
    s_vals = np.linspace(s_low, s_high, n_steps)
    open_init = (0.2, 0.02, 0.95, 0.90, 0.9)   # high initial permeability
    closed_init = (0.8, 0.90, 0.02, 0.05, 0.1) # low initial permeability

    # Up sweep
    df = simulate(open_init, t_end=300, dt=0.05,
                  s_override=s_low, theta_override=PARAMS["theta"])
    x_start = df[["U","B","T","E","P"]].iloc[-1].to_numpy()
    up_B, up_P = [], []
    for s in s_vals:
        df = simulate(x_start, t_end=t_transient, dt=0.05,
                      s_override=s, theta_override=PARAMS["theta"])
        x_start = df[["U","B","T","E","P"]].iloc[-1].to_numpy()
        up_B.append(df.tail(100)["B"].mean())
        up_P.append(df.tail(100)["P"].mean())

    # Down sweep
    df = simulate(closed_init, t_end=300, dt=0.05,
                  s_override=s_high, theta_override=PARAMS["theta"])
    x_start = df[["U","B","T","E","P"]].iloc[-1].to_numpy()
    down_B, down_P = [], []
    for s in s_vals[::-1]:
        df = simulate(x_start, t_end=t_transient, dt=0.05,
                      s_override=s, theta_override=PARAMS["theta"])
        x_start = df[["U","B","T","E","P"]].iloc[-1].to_numpy()
        down_B.append(df.tail(100)["B"].mean())
        down_P.append(df.tail(100)["P"].mean())

    return s_vals, np.array(up_B), np.array(down_B[::-1]), np.array(up_P), np.array(down_P[::-1])

# Run dynamic P sweep
s_vals, up_B, down_B, up_P, down_P = hysteresis_sweep_dynamic()

# Save results
results_df = pd.DataFrame({
    "s": s_vals,
    "B_up": up_B,
    "B_down": down_B,
    "P_up": up_P,
    "P_down": down_P,
})
results_df.to_csv("dynamic_permeability_hysteresis_results.csv", index=False)
print("Saved dynamic_permeability_hysteresis_results.csv")

# Plot B hysteresis
plt.figure(figsize=(10,6))
plt.plot(s_vals, up_B, "o-", color="#1f77b4", label="up (open start)")
plt.plot(s_vals, down_B, "s--", color="#d62728", label="down (closed start)")
plt.xlabel("s")
plt.ylabel("Boundary strength B")
plt.title("Hysteresis with dynamic permeability")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dynamic_permeability_B_hysteresis.png", dpi=150)
plt.show()

# Plot P hysteresis
plt.figure(figsize=(10,6))
plt.plot(s_vals, up_P, "o-", color="#1f77b4", label="up (open start)")
plt.plot(s_vals, down_P, "s--", color="#d62728", label="down (closed start)")
plt.xlabel("s")
plt.ylabel("Permeability P")
plt.title("Permeability hysteresis")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dynamic_permeability_P_hysteresis.png", dpi=150)
plt.show()
