import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============================================================
# Model parameters (same as before)
# ============================================================
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
    # stable sigmoid
    if z > 60:
        return 1.0
    if z < -60:
        return 0.0
    return 1.0 / (1.0 + np.exp(-z))

# ============================================================
# Fast integration for a single run, returns final mean B and std
# ============================================================
def integrate_to_final(x0, s, theta, rhoP, sigma=0.0, t_max=200, dt=0.05,
                       tol=1e-4, patience=100):
    """
    Integrate the 5D system until convergence or t_max.
    Returns (final_B_mean, final_B_std, final_P, final_U)
    """
    p = PARAMS.copy()
    p["rhoP"] = rhoP
    p["theta"] = theta

    U, B, T, E, P = x0
    step = 0
    last_B = B
    stable_count = 0
    B_tail = []

    while step * dt < t_max:
        # felt uncertainty
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

        # Add noise to perceived F (affects only boundary drive)
        if sigma > 0:
            F_noisy = F + sigma * np.random.randn()
            b_drive = sigmoid(p["kb"] * (F_noisy - theta))
            dB = p["rhoB"] * b_drive - p["dB"] * B

        U = np.clip(U + dt * dU, 0, 1)
        B = np.clip(B + dt * dB, 0, 1)
        T = np.clip(T + dt * dT, 0, 1)
        E = np.clip(E + dt * dE, 0, 1)
        P = np.clip(P + dt * dP, 0, 1)

        step += 1
        # record tail for mean
        if step > (t_max/dt) - 200:
            B_tail.append(B)

        # early stopping: if B changes less than tol for patience steps
        if abs(B - last_B) < tol:
            stable_count += 1
        else:
            stable_count = 0
        if stable_count > patience:
            break
        last_B = B

    if len(B_tail) < 50:
        B_tail = [B] * 50
    return np.mean(B_tail), np.std(B_tail), P, U

# ============================================================
# Deterministic phase diagram over (s, theta) for fixed rhoP
# ============================================================
def phase_diagram_s_theta(rhoP, s_vals, theta_vals, sigma=0.0):
    records = []
    for theta in theta_vals:
        for s in s_vals:
            # open start
            B_open_mean, B_open_std, P_open, U_open = integrate_to_final(
                [0.2, 0.02, 0.95, 0.90, 0.9], s, theta, rhoP, sigma)
            # closed start
            B_closed_mean, B_closed_std, P_closed, U_closed = integrate_to_final(
                [0.8, 0.90, 0.02, 0.05, 0.1], s, theta, rhoP, sigma)

            # classify
            def classify(mean, std):
                if std > 0.05:
                    return "oscillatory"
                if mean > 0.55:
                    return "closed"
                elif mean < 0.20:
                    return "open"
                else:
                    return "intermediate"

            c_open = classify(B_open_mean, B_open_std)
            c_closed = classify(B_closed_mean, B_closed_std)
            if c_open != c_closed and "oscillatory" not in (c_open, c_closed):
                final = "bistable"
            elif "oscillatory" in (c_open, c_closed):
                final = "oscillatory"
            else:
                final = c_open

            records.append({
                "rhoP": rhoP,
                "theta": theta,
                "s": s,
                "open_regime": c_open,
                "closed_regime": c_closed,
                "final_regime": final,
                "B_open": B_open_mean,
                "B_closed": B_closed_mean,
                "P_open": P_open,
                "P_closed": P_closed,
                "U_open": U_open,
                "U_closed": U_closed,
            })
            # print progress occasionally
    return pd.DataFrame(records)

# ============================================================
# Stochastic sweep over (s, sigma) for fixed theta, rhoP
# ============================================================
def stochastic_sweep(s_vals, sigma_vals, theta, rhoP, n_runs=30, base_seed=42):
    records = []
    for sigma in sigma_vals:
        for s in s_vals:
            closed_count = 0
            B_finals = []
            for r in range(n_runs):
                np.random.seed(base_seed + r * 1000)
                B_mean, B_std, P, U = integrate_to_final(
                    [0.2, 0.02, 0.95, 0.90, 0.9], s, theta, rhoP, sigma=sigma)
                B_finals.append(B_mean)
                if B_mean > 0.5:
                    closed_count += 1
            prob_closed = closed_count / n_runs
            records.append({
                "sigma": sigma,
                "s": s,
                "theta": theta,
                "rhoP": rhoP,
                "prob_closed_open_start": prob_closed,
                "mean_B": np.mean(B_finals),
                "std_B": np.std(B_finals),
            })
    return pd.DataFrame(records)

# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    # 1. Deterministic phase diagrams for several rhoP values
    rhoP_values = [0.01, 0.02, 0.05, 0.10]
    s_grid = np.linspace(0.5, 1.8, 20)
    theta_grid = np.linspace(0.08, 0.32, 20)

    all_phase_data = []
    for rhoP in rhoP_values:
        print(f"Running phase diagram for rhoP={rhoP}")
        df = phase_diagram_s_theta(rhoP, s_grid, theta_grid)
        all_phase_data.append(df)

    # Combine and save
    full_phase_df = pd.concat(all_phase_data, ignore_index=True)
    full_phase_df.to_csv("phase_diagram_5D_deterministic.csv", index=False)

    # Plot a 2x2 panel of heatmaps (one per rhoP)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.flatten()
    regime_to_int = {"open": 0, "intermediate": 1, "closed": 2, "bistable": 3, "oscillatory": 4}
    cmap = ListedColormap(["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"])

    for idx, rhoP in enumerate(rhoP_values):
        df_sub = full_phase_df[full_phase_df["rhoP"] == rhoP]
        grid = np.empty((len(theta_grid), len(s_grid)), dtype=int)
        for i, theta in enumerate(theta_grid):
            for j, s in enumerate(s_grid):
                row = df_sub[(df_sub["theta"] == theta) & (df_sub["s"] == s)]
                if not row.empty:
                    grid[i, j] = regime_to_int[row.iloc[0]["final_regime"]]
        im = axes[idx].imshow(grid, extent=[s_grid[0], s_grid[-1], theta_grid[0], theta_grid[-1]],
                              origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=4)
        axes[idx].set_title(f"rhoP = {rhoP}")
        axes[idx].set_xlabel("s")
        axes[idx].set_ylabel("theta")
    fig.colorbar(im, ax=axes, ticks=range(5), label="open, intermediate, closed, bistable, oscillatory")
    plt.savefig("phase_diagram_5D_heatmaps.png", dpi=150)
    plt.show()

    # 2. Stochastic sweep over (s, sigma) for typical theta, rhoP
    theta_fixed = 0.196
    rhoP_fixed = 0.02
    s_grid_stoch = np.linspace(0.7, 1.5, 15)
    sigma_grid = np.linspace(0.0, 0.3, 10)
    print("Running stochastic sweep...")
    stoch_df = stochastic_sweep(s_grid_stoch, sigma_grid, theta_fixed, rhoP_fixed,
                                n_runs=30)
    stoch_df.to_csv("stochastic_sweep_5D.csv", index=False)

    # Plot stochastic heatmap
    grid = np.empty((len(sigma_grid), len(s_grid_stoch)))
    for i, sigma in enumerate(sigma_grid):
        for j, s in enumerate(s_grid_stoch):
            row = stoch_df[(stoch_df["sigma"] == sigma) & (stoch_df["s"] == s)]
            grid[i, j] = row.iloc[0]["prob_closed_open_start"]

    plt.figure(figsize=(10, 6))
    plt.imshow(grid, extent=[s_grid_stoch[0], s_grid_stoch[-1], sigma_grid[0], sigma_grid[-1]],
               origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Probability of closure (open start)")
    plt.xlabel("s (stakes)")
    plt.ylabel("sigma (noise)")
    plt.title(f"Stochastic closure probability (theta={theta_fixed}, rhoP={rhoP_fixed})")
    plt.tight_layout()
    plt.savefig("stochastic_heatmap_5D.png", dpi=150)
    plt.show()
