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

def deriv(x, p, s, theta, P_min=None):
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
    dP = p["rhoP"] * (desired_P - P)

    # Optional floor: enforce P >= P_min
    if P_min is not None and P < P_min:
        P = P_min
        dP = max(0.0, dP)

    return np.array([dU, dB, dT, dE, dP]), F

def simulate_with_floor(x0, s_base, theta, P_min,
                        t_end=400, dt=0.05,
                        shock_start=100, shock_duration=30,
                        shock_s=3.0, shock_P=0.02,
                        p=PARAMS):
    times = np.arange(0, t_end + dt, dt)
    x = np.array(x0, dtype=float)
    rows = []

    for t in times:
        in_shock = (shock_start <= t < shock_start + shock_duration)
        s = shock_s if in_shock else s_base

        # Determine desired P and forcing
        if in_shock:
            target_P = max(shock_P, P_min)   # force P low but respect floor
            forced_P = True
        else:
            target_P = None
            forced_P = False

        # Compute derivatives with current state and parameters
        F = s * x[0] / ((1 + p["lam"] * x[2]) * (1 + p["mu"] * x[1]))
        block = (1.0 - x[4]) * x[1]

        b_drive = sigmoid(p["kb"] * (F - theta))
        e_drive = sigmoid(p["ke"] * (p["alpha"] * x[0] / (1 + p["eta"] * block) - p["cE"]))

        dU = p["n"] * (1 - x[0]) - p["alpha"] * x[3] * (1 - p["beta"] * block) * x[0]
        dB = p["rhoB"] * b_drive - p["dB"] * x[1]
        dT = p["rhoT"] * x[3] * (1 - p["betaT"] * block) - p["dT"] * x[2] - p["gamma"] * block * x[2]
        dE = p["rhoE"] * e_drive - p["dE"] * x[3]

        # Permeability dynamics
        if forced_P:
            dP = 0.0
            new_P = target_P
        else:
            desired_P = 1.0 - sigmoid(p["kP"] * (F - p["thetaP"]))
            dP = p["rhoP"] * (desired_P - x[4])
            new_P = x[4] + dt * dP
            # Apply floor
            new_P = max(new_P, P_min)

        # Update all variables
        x[0] = np.clip(x[0] + dt * dU, 0, 1)
        x[1] = np.clip(x[1] + dt * dB, 0, 1)
        x[2] = np.clip(x[2] + dt * dT, 0, 1)
        x[3] = np.clip(x[3] + dt * dE, 0, 1)
        x[4] = np.clip(new_P, 0, 1)

        rows.append([t, *x, s, theta, P_min])

    return pd.DataFrame(rows, columns=["t","U","B","T","E","P","s","theta","P_min"])

# Find open attractor at s=1.5 with no floor
s_base = 1.5
theta = PARAMS["theta"]
df_open = simulate_with_floor((0.2,0.02,0.95,0.90,0.9), s_base, theta, P_min=0.0,
                              t_end=300, dt=0.05,
                              shock_start=9999, shock_duration=0,
                              shock_s=s_base, shock_P=0.0)
x_open = df_open[["U","B","T","E","P"]].iloc[-1].to_numpy()
print(f"Open attractor at s={s_base}: B={x_open[1]:.3f}, P={x_open[4]:.3f}")

# Run intervention sweep over P_min
P_min_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
results = []
for P_min in P_min_values:
    df = simulate_with_floor(x_open, s_base, theta, P_min,
                             t_end=400, dt=0.05,
                             shock_start=100, shock_duration=30,
                             shock_s=3.0, shock_P=0.02)
    final = df.iloc[-1]
    results.append({
        "P_min": P_min,
        "final_B": final["B"],
        "final_P": final["P"],
        "final_T": final["T"],
        "final_E": final["E"],
        "final_U": final["U"],
        "recovered": final["B"] < 0.2,
    })
    print(f"P_min={P_min:.1f}: B={final['B']:.3f}, P={final['P']:.3f}, "
          f"T={final['T']:.3f}, recovered={final['B'] < 0.2}")

res_df = pd.DataFrame(results)
res_df.to_csv("P_min_intervention_results.csv", index=False)

# Plot final B vs P_min
plt.figure(figsize=(8,5))
plt.plot(res_df["P_min"], res_df["final_B"], "o-", color="#1f77b4")
plt.axhline(y=0.55, color="red", linestyle="--", label="closure threshold")
plt.axhline(y=0.20, color="green", linestyle="--", label="open threshold")
plt.xlabel("Minimum permeability floor  P_min")
plt.ylabel("Final boundary strength  B")
plt.title("Effect of constitutional transparency floor on crisis recovery")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("P_min_intervention.png", dpi=150)
plt.show()
