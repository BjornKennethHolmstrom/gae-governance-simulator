import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Model parameters (same as before)
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

def simulate_noisy(x0, t_end=150, dt=0.05, s_override=None,
                   theta_override=None, sigma=0.05, seed=None, p=PARAMS):
    """
    Euler integration with additive Gaussian noise on perceived F.
    sigma=0 gives deterministic dynamics.
    """
    if seed is not None:
        np.random.seed(seed)
    times = np.arange(0, t_end + dt, dt)
    x = np.array(x0, dtype=float)
    rows = []

    for t in times:
        s = p["s"] if s_override is None else s_override
        theta = p["theta"] if theta_override is None else theta_override
        U, B, T, E = x
        F = s * U / ((1 + p["lam"] * T) * (1 + p["mu"] * B))
        F_noisy = F + sigma * np.random.randn()
        b_drive = sigmoid(p["kb"] * (F_noisy - theta))
        e_drive = sigmoid(p["ke"] * (p["alpha"] * U / (1 + p["eta"] * B) - p["cE"]))

        dU = p["n"] * (1 - U) - p["alpha"] * E * (1 - p["beta"] * B) * U
        dB = p["rhoB"] * b_drive - p["dB"] * B
        dT = p["rhoT"] * E * (1 - p["betaT"] * B) - p["dT"] * T - p["gamma"] * B * T
        dE = p["rhoE"] * e_drive - p["dE"] * E

        rows.append([t, U, B, T, E, s, theta, F, F_noisy])
        x = np.clip(x + dt * np.array([dU, dB, dT, dE]), 0.0, 1.0)

    return pd.DataFrame(rows, columns=["t", "U", "B", "T", "E", "s", "theta", "F", "F_noisy"])

# ------------------------------------------------------------
# Stochastic hysteresis sweep
# ------------------------------------------------------------
def run_stochastic_sweep(s_values, theta, sigma, n_runs=50,
                         open_init=(0.2, 0.02, 0.95, 0.90),
                         closed_init=(0.8, 0.9, 0.02, 0.05),
                         t_end=150, dt=0.05, base_seed=42):
    """
    For each s, run n_runs noisy simulations from open_init and closed_init.
    Record fraction ending closed (B>0.5) and mean final B.
    """
    records = []
    for s in s_values:
        # open start
        final_B_open = []
        for r in range(n_runs):
            df = simulate_noisy(open_init, t_end=t_end, dt=dt,
                                s_override=s, theta_override=theta,
                                sigma=sigma, seed=base_seed + r * 1000)
            final_B_open.append(df["B"].iloc[-1])
        prob_open = np.mean(np.array(final_B_open) > 0.5)
        mean_B_open = np.mean(final_B_open)

        # closed start
        final_B_closed = []
        for r in range(n_runs):
            df = simulate_noisy(closed_init, t_end=t_end, dt=dt,
                                s_override=s, theta_override=theta,
                                sigma=sigma, seed=base_seed + r * 1000 + 500000)
            final_B_closed.append(df["B"].iloc[-1])
        prob_closed = np.mean(np.array(final_B_closed) > 0.5)
        mean_B_closed = np.mean(final_B_closed)

        records.append({
            "s": s,
            "prob_closed_open_start": prob_open,
            "mean_B_open_start": mean_B_open,
            "prob_closed_closed_start": prob_closed,
            "mean_B_closed_start": mean_B_closed,
        })
        print(f"s={s:.3f}  P(closed|open start)={prob_open:.3f}  "
              f"P(closed|closed start)={prob_closed:.3f}")

    return pd.DataFrame(records)

# Sweep parameters
s_low = 0.60
s_high = 1.60
n_s = 21                      # number of s points
s_values = np.linspace(s_low, s_high, n_s)
theta = PARAMS["theta"]
sigma = 0.05                  # noise level (same as the one that caused tipping at s=1.15)
n_runs = 50                   # Monte Carlo runs per s per start

print("Running stochastic hysteresis sweep...")
sweep_df = run_stochastic_sweep(s_values, theta, sigma, n_runs=n_runs)

# Save results
sweep_df.to_csv("stochastic_hysteresis_results.csv", index=False)
print("Saved stochastic_hysteresis_results.csv")

# ------------------------------------------------------------
# Plot probability of closure vs s
# ------------------------------------------------------------
plt.figure(figsize=(9,6))
plt.plot(sweep_df["s"], sweep_df["prob_closed_open_start"], "o-",
         color="#1f77b4", label="open start")
plt.plot(sweep_df["s"], sweep_df["prob_closed_closed_start"], "s-",
         color="#d62728", label="closed start")
plt.axvline(x=1.15, color="gray", linestyle="--", alpha=0.6,
            label="previous single-s test (s=1.15)")
plt.xlabel("s  (stakes / uncertainty multiplier)")
plt.ylabel("Probability of ending closed")
plt.title(f"Stochastic hysteresis (σ={sigma})")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("stochastic_hysteresis_probability.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# Plot mean final B vs s
# ------------------------------------------------------------
plt.figure(figsize=(9,6))
plt.plot(sweep_df["s"], sweep_df["mean_B_open_start"], "o-",
         color="#1f77b4", label="open start")
plt.plot(sweep_df["s"], sweep_df["mean_B_closed_start"], "s-",
         color="#d62728", label="closed start")
plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.6,
            label="closure threshold")
plt.xlabel("s  (stakes / uncertainty multiplier)")
plt.ylabel("Mean final boundary strength  B")
plt.title(f"Stochastic hysteresis: mean boundary strength (σ={sigma})")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("stochastic_hysteresis_meanB.png", dpi=150)
plt.show()
