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
    rhoP=0.02, kP=20.0, thetaP=0.15,
)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

def deriv(x, p, s, theta, force_P=None):
    U, B, T, E, P = x
    F = s * U / ((1 + p["lam"] * T) * (1 + p["mu"] * B))
    block = (1.0 - P) * B
    b_drive = sigmoid(p["kb"] * (F - theta))
    e_drive = sigmoid(p["ke"] * (p["alpha"] * U / (1 + p["eta"] * block) - p["cE"]))
    dU = p["n"] * (1 - U) - p["alpha"] * E * (1 - p["beta"] * block) * U
    dB = p["rhoB"] * b_drive - p["dB"] * B
    dT = p["rhoT"] * E * (1 - p["betaT"] * block) - p["dT"] * T - p["gamma"] * block * T
    dE = p["rhoE"] * e_drive - p["dE"] * E
    desired_P = 1.0 - sigmoid(p["kP"] * (F - p["thetaP"]))
    P_target = force_P if force_P is not None else desired_P
    dP = p["rhoP"] * (P_target - P)
    return np.array([dU, dB, dT, dE, dP]), F

def simulate_combined_shock(x0, s_base, theta, t_end=400, dt=0.05,
                            shock_start=100, shock_duration=30,
                            shock_s=3.0, shock_P=0.02, p=PARAMS):
    times = np.arange(0, t_end + dt, dt)
    x = np.array(x0, dtype=float)
    rows = []
    for t in times:
        in_shock = (shock_start <= t < shock_start + shock_duration)
        s = shock_s if in_shock else s_base
        force_P = shock_P if in_shock else None
        dx, F = deriv(x, p, s, theta, force_P=force_P)
        rows.append([t, *x, s, theta, F])
        x = np.clip(x + dt * dx, 0.0, 1.0)
    return pd.DataFrame(rows, columns=["t","U","B","T","E","P","s","theta","F"])

# Define baseline open attractor at s=1.5
s_base = 1.5
theta = PARAMS["theta"]
df_open = simulate_combined_shock((0.2,0.02,0.95,0.90,0.9), s_base, theta,
                                  t_end=300, dt=0.05,
                                  shock_start=9999, shock_duration=0,
                                  shock_s=s_base, shock_P=0.0)
x_open = df_open[["U","B","T","E","P"]].iloc[-1].to_numpy()
print(f"Open attractor at s={s_base}: B={x_open[1]:.3f}, P={x_open[4]:.3f}")

# Run combined shock
df_shock = simulate_combined_shock(x_open, s_base, theta,
                                   t_end=400, dt=0.05,
                                   shock_start=100, shock_duration=30,
                                   shock_s=3.0, shock_P=0.02)

# Plot
fig, axes = plt.subplots(3,1, figsize=(10,10), sharex=True)
axes[0].plot(df_shock["t"], df_shock["s"], color="black")
axes[0].set_ylabel("s")
axes[0].grid(alpha=0.3)

axes[1].plot(df_shock["t"], df_shock["B"], color="#1f77b4")
axes[1].axvspan(100, 130, color="red", alpha=0.2)
axes[1].set_ylabel("B")
axes[1].grid(alpha=0.3)

axes[2].plot(df_shock["t"], df_shock["P"], color="#2ca02c")
axes[2].axvspan(100, 130, color="red", alpha=0.2)
axes[2].set_ylabel("P")
axes[2].set_xlabel("Time")
axes[2].grid(alpha=0.3)

plt.suptitle("Combined shock: high stakes + forced low permeability")
plt.tight_layout()
plt.savefig("combined_shock_experiment.png", dpi=150)
plt.show()

# Final state
final = df_shock.iloc[-1]
print(f"After combined shock: B={final['B']:.3f}, P={final['P']:.3f}, "
      f"T={final['T']:.3f}, E={final['E']:.3f}, U={final['U']:.3f}")
