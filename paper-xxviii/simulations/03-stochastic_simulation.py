import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    e_drive = sigmoid(p["ke"] * (p["alpha"] * U / (1 + p["eta"] * B) - p["cE"]))

    dU = p["n"] * (1 - U) - p["alpha"] * E * (1 - p["beta"] * B) * U
    dB = p["rhoB"] * b_drive - p["dB"] * B
    dT = p["rhoT"] * E * (1 - p["betaT"] * B) - p["dT"] * T - p["gamma"] * B * T
    dE = p["rhoE"] * e_drive - p["dE"] * E

    return np.array([dU, dB, dT, dE]), F

# ----------------------------------------------------------------------
# Stochastic simulation: noise enters through perceived felt uncertainty
# ----------------------------------------------------------------------
def simulate_noisy(x0, t_end=200, dt=0.05, s_override=None,
                   theta_override=None, sigma=0.0, seed=None, p=PARAMS):
    """
    Euler integration with additive Gaussian noise on the perceived F
    (standard deviation = sigma). sigma=0 gives deterministic dynamics.
    """
    if seed is not None:
        np.random.seed(seed)
    times = np.arange(0, t_end + dt, dt)
    x = np.array(x0, dtype=float)
    rows = []

    for t in times:
        s = p["s"] if s_override is None else s_override
        theta = p["theta"] if theta_override is None else theta_override
        # Compute deterministic derivative, but with noise in the boundary drive
        U, B, T, E = x
        F = s * U / ((1 + p["lam"] * T) * (1 + p["mu"] * B))
        # Noisy perception of F
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

# ----------------------------------------------------------------------
# Experiment 1: Noise-induced transitions near the deterministic tipping point
# ----------------------------------------------------------------------
def probability_closure(s, theta, sigma, n_runs=100, t_end=150, dt=0.05,
                        open_init=(0.2, 0.02, 0.95, 0.90), seed=42):
    """Run n_runs noisy simulations from open_init, return fraction ending closed (B > 0.5)."""
    closed_count = 0
    final_Bs = []
    for i in range(n_runs):
        df = simulate_noisy(open_init, t_end=t_end, dt=dt,
                            s_override=s, theta_override=theta,
                            sigma=sigma, seed=seed+i)
        final_B = df["B"].iloc[-1]
        final_Bs.append(final_B)
        if final_B > 0.5:
            closed_count += 1
    return closed_count / n_runs, final_Bs

# Choose parameter near the deterministic tipping point
s_test = 1.15          # from up-sweep, deterministic open stays open until s ~1.2
theta_test = PARAMS["theta"]  # 0.196

sigma_values = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
n_runs = 200            # number of Monte Carlo runs per sigma

print("Running noise-induced transition experiment...")
results = []
for sigma in sigma_values:
    prob, Bs = probability_closure(s_test, theta_test, sigma, n_runs=n_runs, seed=12345)
    results.append({"sigma": sigma, "prob_closed": prob, "final_B_mean": np.mean(Bs),
                    "final_B_std": np.std(Bs)})
    print(f"sigma={sigma:4.2f}  P(closed)={prob:.3f}  mean(B)={np.mean(Bs):.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv("noise_transition_results.csv", index=False)

# Plot probability vs sigma
plt.figure(figsize=(8,5))
plt.plot(results_df["sigma"], results_df["prob_closed"], "o-", color="#d62728")
plt.xlabel("Noise intensity sigma")
plt.ylabel("Probability of ending closed")
plt.title(f"Noise-induced closure at s={s_test}, theta={theta_test:.3f}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("noise_probability_vs_sigma.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------
# Experiment 2: Sample trajectories for three noise levels
# ----------------------------------------------------------------------
print("Generating sample trajectories...")
sigmas_plot = [0.0, 0.05, 0.2]
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, sigma in zip(axes, sigmas_plot):
    df = simulate_noisy((0.2, 0.02, 0.95, 0.90), t_end=150, dt=0.05,
                        s_override=s_test, theta_override=theta_test,
                        sigma=sigma, seed=7)
    ax.plot(df["t"], df["B"], color="#1f77b4", lw=1.5)
    ax.set_ylabel(f"B (sigma={sigma})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Time")
plt.suptitle("Boundary strength trajectories under noise (s=1.15)")
plt.tight_layout()
plt.savefig("noise_sample_trajectories.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------
# Experiment 3: Histogram of final B for a fixed noise level
# ----------------------------------------------------------------------
sigma_hist = 0.1
_, final_Bs = probability_closure(s_test, theta_test, sigma_hist, n_runs=1000, seed=999)
plt.figure(figsize=(8,5))
plt.hist(final_Bs, bins=30, color="#2ca02c", alpha=0.7, edgecolor="k")
plt.xlabel("Final boundary strength B")
plt.ylabel("Frequency")
plt.title(f"Distribution of final B (sigma={sigma_hist})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("noise_histogram_finalB.png", dpi=150)
plt.show()

print("\nNoise extension analysis complete.")
