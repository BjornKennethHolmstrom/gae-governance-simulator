import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Same model as before, but with ability to temporarily set P to a low value
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
    if force_P is not None:
        P_target = force_P
    else:
        P_target = desired_P
    dP = p["rhoP"] * (P_target - P)
    return np.array([dU, dB, dT, dE, dP]), F

def simulate_with_P_shock(x0, s, theta, t_end=400, dt=0.05,
                          shock_start=100, shock_duration=20,
                          shock_P=0.05, p=PARAMS):
    times = np.arange(0, t_end + dt, dt)
    x = np.array(x0, dtype=float)
    rows = []
    for t in times:
        # Force P during shock window
        force_P = shock_P if (shock_start <= t < shock_start + shock_duration) else None
        dx, F = deriv(x, p, s, theta, force_P=force_P)
        rows.append([t, *x, s, theta, F])
        x = np.clip(x + dt * dx, 0.0, 1.0)
    return pd.DataFrame(rows, columns=["t","U","B","T","E","P","s","theta","F"])

# Find open attractor at s=1.5
s_test = 1.5
theta_test = PARAMS["theta"]
df_open = simulate_with_P_shock((0.2,0.02,0.95,0.90,0.9), s=s_test, theta=theta_test,
                                t_end=300, dt=0.05, shock_start=9999, shock_duration=0,
                                shock_P=0.0)
x_open = df_open[["U","B","T","E","P"]].iloc[-1].to_numpy()
print(f"Open attractor at s={s_test}: B={x_open[1]:.3f}, P={x_open[4]:.3f}")

# Run shock experiment
shock_duration = 20
df_shock = simulate_with_P_shock(x_open, s=s_test, theta=theta_test,
                                 t_end=400, dt=0.05,
                                 shock_start=100, shock_duration=shock_duration,
                                 shock_P=0.05)

# Plot
fig, axes = plt.subplots(2,1, figsize=(10,8), sharex=True)
axes[0].plot(df_shock["t"], df_shock["B"], color="#1f77b4", label="Boundary strength B")
axes[0].axvspan(100, 100+shock_duration, color="red", alpha=0.2, label="P shock")
axes[0].set_ylabel("B")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(df_shock["t"], df_shock["P"], color="#2ca02c", label="Permeability P")
axes[1].axvspan(100, 100+shock_duration, color="red", alpha=0.2)
axes[1].set_ylabel("P")
axes[1].set_xlabel("Time")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.suptitle(f"Temporary permeability shock at s={s_test}")
plt.tight_layout()
plt.savefig("permeability_shock_experiment.png", dpi=150)
plt.show()

# Record final state
final_row = df_shock.iloc[-1]
print(f"After shock: B={final_row['B']:.3f}, P={final_row['P']:.3f}, "
      f"T={final_row['T']:.3f}, E={final_row['E']:.3f}")
