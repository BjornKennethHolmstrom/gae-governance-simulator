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

def deriv_two(x, p, s, theta, force_P1=None, force_P2=None):
    # x = [U, B1, T1, E1, P1, B2, T2, E2, P2]
    U = x[0]
    B1, T1, E1, P1 = x[1], x[2], x[3], x[4]
    B2, T2, E2, P2 = x[5], x[6], x[7], x[8]

    block1 = (1.0 - P1) * B1
    block2 = (1.0 - P2) * B2

    # Felt uncertainty for each population
    F1 = s * U / ((1 + p["lam"] * T1) * (1 + p["mu"] * B1))
    F2 = s * U / ((1 + p["lam"] * T2) * (1 + p["mu"] * B2))

    # Boundary drives
    b_drive1 = sigmoid(p["kb"] * (F1 - theta))
    b_drive2 = sigmoid(p["kb"] * (F2 - theta))

    # Exploration drives
    e_drive1 = sigmoid(p["ke"] * (p["alpha"] * U / (1 + p["eta"] * block1) - p["cE"]))
    e_drive2 = sigmoid(p["ke"] * (p["alpha"] * U / (1 + p["eta"] * block2) - p["cE"]))

    # Effective exploration contributions to reducing global U
    E1_eff = E1 * (1 - p["beta"] * block1)
    E2_eff = E2 * (1 - p["beta"] * block2)

    # Global uncertainty dynamics
    dU = p["n"] * (1 - U) - p["alpha"] * (E1_eff + E2_eff) * U

    # Population 1 dynamics
    dB1 = p["rhoB"] * b_drive1 - p["dB"] * B1
    dT1 = p["rhoT"] * E1 * (1 - p["betaT"] * block1) - p["dT"] * T1 - p["gamma"] * block1 * T1
    dE1 = p["rhoE"] * e_drive1 - p["dE"] * E1
    desired_P1 = 1.0 - sigmoid(p["kP"] * (F1 - p["thetaP"]))
    P_target1 = force_P1 if force_P1 is not None else desired_P1
    dP1 = p["rhoP"] * (P_target1 - P1)

    # Population 2 dynamics
    dB2 = p["rhoB"] * b_drive2 - p["dB"] * B2
    dT2 = p["rhoT"] * E2 * (1 - p["betaT"] * block2) - p["dT"] * T2 - p["gamma"] * block2 * T2
    dE2 = p["rhoE"] * e_drive2 - p["dE"] * E2
    desired_P2 = 1.0 - sigmoid(p["kP"] * (F2 - p["thetaP"]))
    P_target2 = force_P2 if force_P2 is not None else desired_P2
    dP2 = p["rhoP"] * (P_target2 - P2)

    return np.array([dU, dB1, dT1, dE1, dP1, dB2, dT2, dE2, dP2]), F1, F2


def simulate_two(x0, t_end=400, dt=0.05, s_override=None, theta_override=None,
                 shock_start=None, shock_duration=None,
                 shock_s=None, shock_P1=None, shock_P2=None, p=PARAMS):
    times = np.arange(0, t_end + dt, dt)
    x = np.array(x0, dtype=float)
    rows = []
    for t in times:
        s = p["s"] if s_override is None else s_override
        theta = p["theta"] if theta_override is None else theta_override

        # Determine if currently in shock
        in_shock = False
        if shock_start is not None and shock_duration is not None:
            in_shock = (shock_start <= t < shock_start + shock_duration)

        # Apply shock parameters if in shock
        if in_shock:
            s_eff = shock_s if shock_s is not None else s
            force_P1 = shock_P1 if shock_P1 is not None else None
            force_P2 = shock_P2 if shock_P2 is not None else None
        else:
            s_eff = s
            force_P1 = None
            force_P2 = None

        dx, F1, F2 = deriv_two(x, p, s_eff, theta, force_P1=force_P1, force_P2=force_P2)
        rows.append([t, *x, s_eff, theta, F1, F2])
        x = np.clip(x + dt * dx, 0.0, 1.0)

    return pd.DataFrame(rows, columns=[
        "t", "U", "B1", "T1", "E1", "P1", "B2", "T2", "E2", "P2",
        "s", "theta", "F1", "F2"
    ])


# ----------------------------------------------------------------------
# Experiment 1: Polarization baseline - open vs closed initial states
# ----------------------------------------------------------------------
s_test = 1.5
theta_test = PARAMS["theta"]

# Initial state: population 1 open, population 2 closed
x0_polar = [0.5, 0.02, 0.95, 0.90, 0.9, 0.90, 0.02, 0.05, 0.1]

print("Running polarization baseline...")
df_polar = simulate_two(x0_polar, t_end=500, dt=0.05,
                        s_override=s_test, theta_override=theta_test)

# Plot results
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
axes[0].plot(df_polar["t"], df_polar["U"], color="black", label="Global U")
axes[0].set_ylabel("U")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(df_polar["t"], df_polar["B1"], color="#1f77b4", label="B1 (open start)")
axes[1].plot(df_polar["t"], df_polar["B2"], color="#d62728", label="B2 (closed start)")
axes[1].set_ylabel("Boundary strength")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(df_polar["t"], df_polar["T1"], color="#1f77b4", linestyle="--", label="T1")
axes[2].plot(df_polar["t"], df_polar["T2"], color="#d62728", linestyle="--", label="T2")
axes[2].set_ylabel("Trust")
axes[2].legend()
axes[2].grid(alpha=0.3)

axes[3].plot(df_polar["t"], df_polar["P1"], color="#1f77b4", label="P1")
axes[3].plot(df_polar["t"], df_polar["P2"], color="#d62728", label="P2")
axes[3].set_ylabel("Permeability")
axes[3].set_xlabel("Time")
axes[3].legend()
axes[3].grid(alpha=0.3)

plt.suptitle(f"Polarization: open vs closed initial states (s={s_test})")
plt.tight_layout()
plt.savefig("two_pop_polarization_baseline.png", dpi=150)
plt.show()

# Final states
final = df_polar.iloc[-1]
print(f"Final state: B1={final['B1']:.3f}, B2={final['B2']:.3f}, "
      f"T1={final['T1']:.3f}, T2={final['T2']:.3f}, "
      f"P1={final['P1']:.3f}, P2={final['P2']:.3f}, U={final['U']:.3f}")

# Save results
df_polar.to_csv("two_pop_polarization_baseline.csv", index=False)
print("Saved two_pop_polarization_baseline.csv")


# ----------------------------------------------------------------------
# Experiment 2: Cascade collapse - shock population 1 only, observe population 2
# ----------------------------------------------------------------------
print("Running cascade collapse experiment...")
# Start both open at s=1.5
x0_both_open = [0.3, 0.02, 0.95, 0.90, 0.9, 0.02, 0.95, 0.90, 0.9]

# Shock parameters: only population 1 experiences combined shock
shock_s = 3.0
shock_P1 = 0.02
shock_P2 = None  # population 2 is not forced low
shock_start = 100
shock_duration = 30

df_cascade = simulate_two(x0_both_open, t_end=500, dt=0.05,
                          s_override=s_test, theta_override=theta_test,
                          shock_start=shock_start, shock_duration=shock_duration,
                          shock_s=shock_s, shock_P1=shock_P1, shock_P2=shock_P2)

# Plot
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
axes[0].plot(df_cascade["t"], df_cascade["s"], color="black", label="s")
axes[0].axvspan(shock_start, shock_start+shock_duration, color="red", alpha=0.2)
axes[0].set_ylabel("s")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(df_cascade["t"], df_cascade["B1"], color="#1f77b4", label="B1 (shocked)")
axes[1].plot(df_cascade["t"], df_cascade["B2"], color="#d62728", label="B2 (unshocked)")
axes[1].axvspan(shock_start, shock_start+shock_duration, color="red", alpha=0.2)
axes[1].set_ylabel("Boundary strength")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(df_cascade["t"], df_cascade["T1"], color="#1f77b4", linestyle="--", label="T1")
axes[2].plot(df_cascade["t"], df_cascade["T2"], color="#d62728", linestyle="--", label="T2")
axes[2].axvspan(shock_start, shock_start+shock_duration, color="red", alpha=0.2)
axes[2].set_ylabel("Trust")
axes[2].legend()
axes[2].grid(alpha=0.3)

axes[3].plot(df_cascade["t"], df_cascade["P1"], color="#1f77b4", label="P1")
axes[3].plot(df_cascade["t"], df_cascade["P2"], color="#d62728", label="P2")
axes[3].axvspan(shock_start, shock_start+shock_duration, color="red", alpha=0.2)
axes[3].set_ylabel("Permeability")
axes[3].set_xlabel("Time")
axes[3].legend()
axes[3].grid(alpha=0.3)

plt.suptitle(f"Cascade collapse: shock only population 1 (s={s_test})")
plt.tight_layout()
plt.savefig("two_pop_cascade_collapse.png", dpi=150)
plt.show()

# Final states
final = df_cascade.iloc[-1]
print(f"Final after cascade shock: B1={final['B1']:.3f}, B2={final['B2']:.3f}, "
      f"T1={final['T1']:.3f}, T2={final['T2']:.3f}, "
      f"P1={final['P1']:.3f}, P2={final['P2']:.3f}, U={final['U']:.3f}")

# Save results
df_cascade.to_csv("two_pop_cascade_collapse.csv", index=False)
print("Saved two_pop_cascade_collapse.csv")
