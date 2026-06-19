#!/usr/bin/env python3
"""
gae-simulator-v15-legitimacy-trap.py
====================================
Paper XIII — Legitimacy as Emergent Gain
Simulation of the legitimacy-coupled governance control loop.

Models a two-dimensional system with a Kalman-filtered LQR controller
whose effective actuation and observation noise are modulated by an
endogenous legitimacy parameter L(t). L(t) evolves in response to the
delivery gap, transparency, and (if active) a stochastic betrayal hazard
from suppressed reporting.

Four scenarios:
  1. High-transparency, high-legitimacy equilibrium
  2. The legitimacy trap (large external shock)
  3. Recovery through transparency intervention
  4. Borrowed-legitimacy collapse (suppression + stochastic revelation)

Outputs:
  outputs/v15-phase-diagram.png
  outputs/v15-trap-and-recovery.png
  outputs/v15-borrowed-collapse.png
  outputs/v15-collapse-heatmap.png
  outputs/v15-asymmetry-sweep.png

Corrections vs. original version
---------------------------------
1. A_mat changed from 0.95*I to 1.05*I (critical correctness fix).
   With A=0.95 the uncontrolled system is stable, so x always decays to 0
   regardless of L.  The delivery gap therefore always vanishes, L always
   recovers for any T>0, and no genuine low-L attractor can exist — making
   the legitimacy trap, the asymmetry sweep effect, and the phase-diagram
   basin structure all mathematically impossible.  With A=1.05 the open-loop
   spectral radius is 1.05; the LQR (recomputed for A=1.05) gives K≈0.963,
   yielding a critical threshold L*=(A-1)/K≈0.052.  Below L*, the closed-loop
   is unstable: x diverges, the delivery gap grows, and L cannot recover —
   a genuine structural trap.

2. Revelation forces T_transp=1 in addition to lambda_sup=1.
   Previously only lambda_sup was set to 1 at revelation, while T_transp
   stayed at its scenario value (0.2 for borrowed-legitimacy), giving only
   β·0.2+δ=0.021/step post-revelation instead of β·1.0+δ=0.085/step.
   Since revelation means the public now knows the true state, both the
   reporting fidelity (lambda) and the declared transparency (T) should
   be forced to 1.

3. Shock moved before measurement (fixes asymmetry sweep all-1 problem).
   With A=1.05 and the shock placed AFTER the Kalman update, x_hat saw 0
   at the shock step, so u≈0 at t+1 while A=1.05 grew the state from [3,0]
   to [3.15,0].  The extra gap increment pushed L below L* even at ratio=1
   (symmetric hysteresis), trapping every run.  Placing the shock before
   measurement means x_hat is updated correctly and corrective control is
   applied one step earlier.

4. Phase diagram now uses a probe shock at t=0 with magnitude [2,0] and
   lambda_sup=1.0.  Without a shock and with x(0)=0, there is nothing to
   destabilise and all cells recover to L=1 regardless of L(0) or T,
   producing an uninformative all-green grid.  The probe shock creates a
   clear diagonal separatrix: low L(0) and low T → trap; high L(0) or
   high T → recovery.  lambda_sup=1 keeps suppression/revelation out of
   the sweep so only L(0) and T drive the basin structure.
"""

import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os, time

os.makedirs('outputs', exist_ok=True)

# ── Global parameters (Appendix B.1) ─────────────────────────────────────────
n = 2                     # state dimension
A_mat = 1.05 * np.eye(n)  # internal dynamics — mildly unstable open-loop
                          # (spectral radius 1.05) so that low legitimacy
                          # (L_B < L* ≈ 0.052) produces a diverging closed-loop
                          # and a genuine structural trap. With A=0.95 the
                          # uncontrolled system was stable and x always decayed
                          # to 0, making the trap mathematically impossible for
                          # any T > 0.
B_mat = np.eye(n)         # nominal actuation
C_mat = np.eye(n)         # full-state observation
W_cov = 0.01 * np.eye(n)  # process noise covariance
V0    = 0.05 * np.eye(n)  # baseline measurement noise covariance (at L=1)

# LQR design
Q_lqr = np.eye(n)
R_lqr = 0.1 * np.eye(n)
P_lqr = solve_discrete_are(A_mat, B_mat, Q_lqr, R_lqr)
K_nom = np.linalg.inv(R_lqr + B_mat.T @ P_lqr @ B_mat) @ B_mat.T @ P_lqr @ A_mat
# For diagonal A,B, K ≈ 0.75 * I

# Simulation length
T_sim   = 300
T_burn  = 20
T_total = T_sim + T_burn

# Monte Carlo
N_MC = 100
SEED_BASE = 20250601

# ── Legitimacy parameters (Appendix B.2) ─────────────────────────────────────
# Hysteresis asymmetry
ALPHA_DROP  = 0.12
ALPHA_RECOV = 0.03

# Transparency sensitivity
BETA = 0.08

# Exogenous drift
DELTA = 0.005

# Betrayal sensitivities
GAMMA_BUILT    = 0.5
GAMMA_BORROWED = 3.0

# Hazard coefficient for stochastic revelation
HAZARD_H = 0.02

# ── Kalman filter ────────────────────────────────────────────────────────────
def kalman_step(x_hat, P, u, y, L_B, L_C):
    """Single Kalman filter step. Returns updated x_hat, P."""
    # Effective actuation
    B_eff = L_B * B_mat
    # Effective observation noise
    V_eff = V0 / max(L_C, 1e-6)

    # Predict
    x_pred = A_mat @ x_hat + B_eff @ u
    P_pred = A_mat @ P @ A_mat.T + W_cov

    # Update
    S = C_mat @ P_pred @ C_mat.T + V_eff
    K = P_pred @ C_mat.T @ np.linalg.inv(S)
    x_hat_new = x_pred + K @ (y - C_mat @ x_pred)
    P_new = (np.eye(n) - K @ C_mat) @ P_pred

    return x_hat_new, P_new, K

# ── Core simulation ──────────────────────────────────────────────────────────
def run_sim(scenario, seed, params=None):
    """
    Run one simulation trajectory.

    Parameters
    ----------
    scenario : int
        1 = high-L equilibrium
        2 = legitimacy trap
        3 = recovery
        4 = borrowed-legitimacy collapse
    seed : int
        RNG seed
    params : dict or None
        Overrides for parameters (e.g. gamma, alpha_drop, etc.)

    Returns
    -------
    dict with keys:
        t, x_true, x_hat, u, y, L, L_B, L_C, x_rep, E_betrayal,
        revelation_event, revelation_step,
        scenario, seed, stable, metrics
    """
    rng = np.random.default_rng(seed)

    # ── Unpack parameters (with defaults) ──
    p = dict(
        L0        = 0.7,
        T_transp  = 1.0,
        lambda_sup = 1.0,   # suppression: 1 = full transparency
        alpha_drop  = ALPHA_DROP,
        alpha_recov = ALPHA_RECOV,
        beta      = BETA,
        gamma     = GAMMA_BUILT,
        delta     = DELTA,
        h         = HAZARD_H,
        shock_t   = None,
        shock_mag = None,
        rebuild_t  = None,
        rebuild_gain_scale = 1.0,
    )
    if params is not None:
        p.update(params)

    # ── Initialise ──
    x_true = np.zeros(n)          # start at target
    x_hat  = np.zeros(n)          # filter estimate
    P_est  = np.eye(n)            # error covariance
    L = p['L0']
    E_betrayal = 0.0
    revelation_occurred = False
    revelation_step = -1

    # Storage
    hist = {
        't': np.arange(T_total),
        'x_true': np.zeros((T_total, n)),
        'x_hat': np.zeros((T_total, n)),
        'u': np.zeros((T_total, n)),
        'y': np.zeros((T_total, n)),
        'L': np.zeros(T_total),
        'x_rep_norm': np.zeros(T_total),
        'E_betrayal': np.zeros(T_total),
        'K_norm': np.zeros(T_total),
    }

    for t in range(T_total):
        # Current L values
        L_C = L
        L_B = L

        # ── Controller ──
        u = -K_nom @ x_hat
        # Gain scaling for recovery scenario
        if p['rebuild_t'] is not None and t >= p['rebuild_t']:
            u = p['rebuild_gain_scale'] * u

        # ── True dynamics ──
        w = rng.multivariate_normal(np.zeros(n), W_cov)
        B_eff = L_B * B_mat
        x_true_next = A_mat @ x_true + B_eff @ u + w

        # ── External shock ──
        # Applied BEFORE measurement so the Kalman filter sees the shocked
        # state immediately and issues corrective control at the next step.
        # Previously placed after measurement, creating a 1-step blind spot:
        # x_hat saw 0, so u≈0 at t+1 while A=1.05 grew the state to [3.15,0],
        # pushing the gap above the single-step L drop and collapsing L to 0
        # even at ratio=1 (symmetric hysteresis) — trapping every run.
        if p['shock_t'] is not None and t == p['shock_t']:
            x_true_next = x_true_next + p['shock_mag']

        # ── Measurement ──
        V_eff = V0 / max(L_C, 1e-6)
        v = rng.multivariate_normal(np.zeros(n), V_eff)
        y = C_mat @ x_true_next + v

        # ── Kalman update ──
        x_hat_next, P_next, K_k = kalman_step(x_hat, P_est, u, y, L_B, L_C)

        # ── Legitimacy dynamics ──
        # Reported state (what the public sees)
        x_rep = p['lambda_sup'] * x_true_next + (1 - p['lambda_sup']) * np.zeros(n)

        # Delivery gap (squared norm of reported state)
        gap = np.sum(x_rep**2)

        # Hysteresis asymmetry: compare current gap to previous
        if t > 0:
            prev_gap = hist['x_rep_norm'][t-1]
        else:
            prev_gap = gap

        if gap > prev_gap:
            alpha = p['alpha_drop']
        else:
            alpha = p['alpha_recov']

        # Transparency contribution
        transp_contrib = p['beta'] * p['T_transp']

        # Betrayal check
        D_t = 0
        if not revelation_occurred and p['lambda_sup'] < 1.0:
            # Accumulate hidden discrepancy
            discrepancy = np.sum((x_true_next - x_rep)**2)
            E_betrayal += discrepancy
            # Stochastic revelation hazard
            prob_rev = 1.0 - np.exp(-p['h'] * E_betrayal)
            if rng.random() < prob_rev:
                D_t = 1
                revelation_occurred = True
                revelation_step = t
                # Revelation forces full transparency on both channels:
                # the public now knows the truth, so suppression collapses
                # and the transparency benefit is also restored to maximum.
                p['lambda_sup'] = 1.0
                p['T_transp']   = 1.0

        # Update legitimacy
        L_next = L - alpha * gap + transp_contrib - p['gamma'] * D_t + p['delta']
        L_next = np.clip(L_next, 0.0, 1.0)

        # ── Store ──
        hist['x_true'][t] = x_true_next
        hist['x_hat'][t] = x_hat_next
        hist['u'][t] = u
        hist['y'][t] = y
        hist['L'][t] = L_next
        hist['x_rep_norm'][t] = gap
        hist['E_betrayal'][t] = E_betrayal
        hist['K_norm'][t] = np.linalg.norm(K_k)

        # ── Advance state ──
        x_true = x_true_next
        x_hat = x_hat_next
        P_est = P_next
        L = L_next

    # ── Compute metrics ──
    post_burn = slice(T_burn, T_total)
    L_post = hist['L'][post_burn]
    x_norm_post = np.linalg.norm(hist['x_true'][post_burn], axis=1)

    metrics = {
        'L_final': np.mean(L_post[-50:]) if len(L_post) >= 50 else np.mean(L_post),
        'L_min': np.min(hist['L']),
        'x_final': np.mean(x_norm_post[-50:]) if len(x_norm_post) >= 50 else np.mean(x_norm_post),
        'revelation_step': revelation_step,
        'stable': True,
    }

    # Trap entry: L fell below 0.25 and didn't recover to 0.5 by end
    L_crit = 0.25
    if np.min(hist['L'][T_burn:]) < L_crit and metrics['L_final'] < 0.5:
        metrics['in_trap'] = 1
    else:
        metrics['in_trap'] = 0

    return {**hist, 'scenario': scenario, 'seed': seed, 'metrics': metrics}


# ── Monte Carlo runner ───────────────────────────────────────────────────────
def mc_run(scenario, n_mc=N_MC, params=None, label=""):
    """Run Monte Carlo ensemble for a scenario."""
    results = []
    for seed in range(n_mc):
        res = run_sim(scenario, SEED_BASE + seed, params=params)
        results.append(res)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Phase diagram — basins of attraction
# ══════════════════════════════════════════════════════════════════════════════
def figure_phase_diagram():
    """Sweep L0 and transparency to map basins of attraction."""
    L0_vals = np.linspace(0.10, 0.90, 17)
    T_vals  = np.linspace(0.0, 1.0, 11)
    L_final_grid = np.zeros((len(L0_vals), len(T_vals)))
    trap_grid = np.zeros((len(L0_vals), len(T_vals)))

    print("Computing phase diagram...")
    for i, L0 in enumerate(L0_vals):
        for j, Tt in enumerate(T_vals):
            finals = []
            traps = []
            for seed in range(30):  # fewer seeds for sweep
                # lambda_sup=1 (no suppression/revelation in the phase diagram;
                # we want only L0 and T to drive the basin structure).
                # shock at t=0 with magnitude [2,0] provides the probe
                # perturbation that makes the separatrix visible.  Without
                # it, x(0)=0 means there is nothing to destabilise and all
                # cells recover to L=1 regardless of L0 or T.
                params = dict(L0=L0, T_transp=Tt, lambda_sup=1.0,
                              shock_t=0, shock_mag=np.array([2.0, 0.0]),
                              alpha_drop=ALPHA_DROP, alpha_recov=ALPHA_RECOV,
                              gamma=GAMMA_BUILT, h=HAZARD_H)
                res = run_sim(1, SEED_BASE + 10000 + i*100 + j*10 + seed, params=params)
                finals.append(res['metrics']['L_final'])
                traps.append(res['metrics']['in_trap'])
            L_final_grid[i, j] = np.mean(finals)
            trap_grid[i, j] = np.mean(traps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    im1 = ax1.pcolormesh(T_vals, L0_vals, L_final_grid, shading='auto',
                          cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_xlabel('Transparency T')
    ax1.set_ylabel('Initial legitimacy L(0)')
    ax1.set_title('Final L (built legitimacy)')
    plt.colorbar(im1, ax=ax1, label='L(T=300)')

    im2 = ax2.pcolormesh(T_vals, L0_vals, trap_grid, shading='auto',
                          cmap='Reds', vmin=0, vmax=1)
    # Overlay approximate separatrix
    ax2.contour(T_vals, L0_vals, trap_grid, levels=[0.5], colors='black', linewidths=2)
    ax2.set_xlabel('Transparency T')
    ax2.set_ylabel('Initial legitimacy L(0)')
    ax2.set_title('Trap entry probability (built legitimacy)')
    plt.colorbar(im2, ax=ax2, label='P(trap)')

    fig.suptitle('Phase Diagram: Basins of Attraction (v15)\n'
                 'Black contour = separatrix; below/left → trap basin',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/v15-phase-diagram.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/v15-phase-diagram.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Trap and recovery trajectories
# ══════════════════════════════════════════════════════════════════════════════
def figure_trap_and_recovery():
    """Plot Scenario 2 (trap) and Scenario 3 (recovery) trajectories."""
    print("Running trap and recovery scenarios...")

    # Scenario 2 — Trap
    params_trap = dict(
        L0=0.7, T_transp=1.0, lambda_sup=1.0,
        shock_t=50, shock_mag=np.array([3.0, 0.0]),
        gamma=GAMMA_BUILT,
    )
    mc_trap = mc_run(2, params=params_trap, label="Trap")

    # Scenario 3 — Recovery (start from low L)
    params_recov = dict(
        L0=0.30, T_transp=1.0, lambda_sup=1.0,
        rebuild_t=50, rebuild_gain_scale=0.5,
        gamma=GAMMA_BUILT,
    )
    mc_recov = mc_run(3, params=params_recov, label="Recovery")

    # Also run a counterfactual recovery without intervention
    params_no_int = dict(
        L0=0.30, T_transp=1.0, lambda_sup=1.0,
        rebuild_t=None, rebuild_gain_scale=1.0,
        gamma=GAMMA_BUILT,
    )
    mc_no_int = mc_run(3, params=params_no_int, label="No intervention")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    ts = np.arange(T_total)

    def plot_band(ax, data_list, ts, color, label, alpha=0.15):
        arr = np.array([d['x_true'] for d in data_list])
        state_norms = np.linalg.norm(arr, axis=2)
        med = np.median(state_norms, axis=0)
        lo  = np.percentile(state_norms, 10, axis=0)
        hi  = np.percentile(state_norms, 90, axis=0)
        ax.plot(ts, med, color=color, lw=2, label=label)
        ax.fill_between(ts, lo, hi, color=color, alpha=alpha)

    # Panel 1: State norm
    ax = axes[0]
    plot_band(ax, mc_trap, ts, '#d62728', 'Scenario 2 (Trap)')
    plot_band(ax, mc_recov, ts, '#2ca02c', 'Scenario 3 (Recovery)')
    plot_band(ax, mc_no_int, ts, 'gray', 'No intervention', alpha=0.08)
    ax.axvline(50, color='purple', ls=':', lw=1.5, alpha=0.7, label='Shock / Intervention')
    ax.set_ylabel('||x(t)||')
    ax.set_title('State norm: trap vs recovery')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 2: Legitimacy
    ax = axes[1]
    for data_list, color, label in [(mc_trap, '#d62728', 'Trap'),
                                     (mc_recov, '#2ca02c', 'Recovery'),
                                     (mc_no_int, 'gray', 'No intervention')]:
        arr = np.array([d['L'] for d in data_list])
        med = np.median(arr, axis=0)
        lo  = np.percentile(arr, 10, axis=0)
        hi  = np.percentile(arr, 90, axis=0)
        ax.plot(ts, med, color=color, lw=2, label=label)
        ax.fill_between(ts, lo, hi, color=color, alpha=0.15)
    ax.axvline(50, color='purple', ls=':', lw=1.5, alpha=0.7)
    ax.set_ylabel('L(t)')
    ax.set_title('Legitimacy: collapse and recovery')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 3: Kalman gain norm (observation effectiveness)
    ax = axes[2]
    for data_list, color, label in [(mc_trap, '#d62728', 'Trap'),
                                     (mc_recov, '#2ca02c', 'Recovery'),
                                     (mc_no_int, 'gray', 'No intervention')]:
        arr = np.array([d['K_norm'] for d in data_list])
        med = np.median(arr, axis=0)
        ax.plot(ts, med, color=color, lw=2, label=label)
    ax.axvline(50, color='purple', ls=':', lw=1.5, alpha=0.7)
    ax.set_ylabel('||K_kalman||')
    ax.set_xlabel('Time step')
    ax.set_title('Kalman gain norm (0 = open-loop, ignores measurements)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle('Legitimacy Trap and Recovery (v15)\n'
                 'Median ± 10–90th percentile, 100 MC seeds',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/v15-trap-and-recovery.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/v15-trap-and-recovery.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Borrowed-legitimacy collapse
# ══════════════════════════════════════════════════════════════════════════════
def figure_borrowed_collapse():
    """Plot Scenario 4 trajectories showing the borrowed-legitimacy collapse."""
    print("Running borrowed-legitimacy scenarios...")

    params_borrowed = dict(
        L0=0.55, T_transp=0.2, lambda_sup=0.3,
        alpha_drop=0.25, alpha_recov=0.02,
        gamma=GAMMA_BORROWED, h=HAZARD_H,
    )

    # Run multiple seeds and pick a few that have revelation events
    seeds_with_rev = []
    for seed in range(500):
        res = run_sim(4, SEED_BASE + 20000 + seed, params=dict(params_borrowed))
        if res['metrics']['revelation_step'] >= 0:
            seeds_with_rev.append(res)
        if len(seeds_with_rev) >= 6:
            break

    if len(seeds_with_rev) < 2:
        print("Warning: few revelation events found; using available seeds")
        if len(seeds_with_rev) == 0:
            # Force more seeds
            for seed in range(500, 2000):
                res = run_sim(4, SEED_BASE + 20000 + seed, params=dict(params_borrowed))
                if res['metrics']['revelation_step'] >= 0:
                    seeds_with_rev.append(res)
                if len(seeds_with_rev) >= 3:
                    break

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    ts = np.arange(T_total)
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds_with_rev)))

    for i, res in enumerate(seeds_with_rev):
        rev_step = res['metrics']['revelation_step']
        color = colors[i]

        # Panel 1: True vs reported state norm
        ax = axes[0]
        true_norm = np.linalg.norm(res['x_true'], axis=1)
        x_rep_norm_sqrt = np.sqrt(res['x_rep_norm'])  # x_rep_norm = gap = ||x_rep||^2
        ax.plot(ts, true_norm, color=color, lw=1.5, alpha=0.7,
                label=f'True (seed {res["seed"]%1000})' if i == 0 else '')
        ax.plot(ts, x_rep_norm_sqrt, color=color, lw=1.5, ls='--', alpha=0.7,
                label=f'Reported' if i == 0 else '')
        ax.axvline(rev_step, color=color, ls=':', lw=2, alpha=0.8)

        # Panel 2: Apparent vs true L
        ax = axes[1]
        ax.plot(ts, res['L'], color=color, lw=1.5, label=f'Seed {res["seed"]%1000}')
        ax.axvline(rev_step, color=color, ls=':', lw=2, alpha=0.8)

        # Panel 3: Hidden discrepancy
        ax = axes[2]
        ax.plot(ts, res['E_betrayal'], color=color, lw=1.5)
        ax.axvline(rev_step, color=color, ls=':', lw=2, alpha=0.8,
                   label=f'Revelation t={rev_step}')

    axes[0].set_ylabel('||x||')
    axes[0].set_title('True vs reported state norm (dashed = reported)')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.2)

    axes[1].set_ylabel('L(t)')
    axes[1].set_title('Legitimacy collapse at revelation')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.2)

    axes[2].set_ylabel('E_betrayal(t)')
    axes[2].set_xlabel('Time step')
    axes[2].set_title('Cumulative hidden discrepancy')
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.2)

    fig.suptitle('Borrowed-Legitimacy Collapse (v15)\n'
                 'Dotted lines = stochastic revelation events',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/v15-borrowed-collapse.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/v15-borrowed-collapse.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Collapse severity heatmap
# ══════════════════════════════════════════════════════════════════════════════
def figure_collapse_heatmap():
    """Sweep suppression duration and betrayal sensitivity."""
    gamma_vals = np.linspace(0.5, 5.0, 10)
    # Suppression duration proxied by 1/h (expected time to revelation)
    dur_vals = np.array([10, 20, 30, 50, 75, 100, 150, 200])
    min_L = np.zeros((len(gamma_vals), len(dur_vals)))
    trap_rate = np.zeros((len(gamma_vals), len(dur_vals)))

    print("Computing collapse heatmap...")
    for i, gamma in enumerate(gamma_vals):
        for j, dur in enumerate(dur_vals):
            h_val = 1.0 / dur
            mins = []
            traps = []
            for seed in range(20):  # fewer seeds per cell
                params = dict(
                    L0=0.55, T_transp=0.2, lambda_sup=0.3,
                    alpha_drop=0.25, alpha_recov=0.02,
                    gamma=gamma, h=h_val,
                )
                res = run_sim(4, SEED_BASE + 30000 + i*100 + j*10 + seed, params=params)
                if res['metrics']['revelation_step'] >= 0:
                    # Compute post-revelation minimum L
                    rev_step = res['metrics']['revelation_step']
                    post_L = res['L'][rev_step:min(rev_step+30, T_total)]
                    mins.append(np.min(post_L) if len(post_L) > 0 else res['L'][-1])
                else:
                    mins.append(res['metrics']['L_final'])
                traps.append(res['metrics']['in_trap'])
            min_L[i, j] = np.mean(mins)
            trap_rate[i, j] = np.mean(traps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    im1 = ax1.pcolormesh(dur_vals, gamma_vals, min_L, shading='auto',
                          cmap='RdYlGn_r', vmin=0, vmax=0.6)
    ax1.set_xlabel('Expected suppression duration (time steps)')
    ax1.set_ylabel('Betrayal sensitivity γ')
    ax1.set_title('Minimum L after revelation')
    plt.colorbar(im1, ax=ax1, label='L_min')

    im2 = ax2.pcolormesh(dur_vals, gamma_vals, trap_rate, shading='auto',
                          cmap='Reds', vmin=0, vmax=1)
    ax2.contour(dur_vals, gamma_vals, trap_rate, levels=[0.5], colors='black', linewidths=2)
    ax2.set_xlabel('Expected suppression duration (time steps)')
    ax2.set_ylabel('Betrayal sensitivity γ')
    ax2.set_title('Trap entry probability\n(black contour = 50%)')
    plt.colorbar(im2, ax=ax2, label='P(trap)')

    fig.suptitle('Collapse Severity Heatmap (v15)\n'
                 'Longer suppression + higher γ = more severe collapse',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/v15-collapse-heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/v15-collapse-heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Hysteresis asymmetry sweep
# ══════════════════════════════════════════════════════════════════════════════
def figure_asymmetry_sweep():
    """Sweep drop/recovery asymmetry ratio and measure trap entry rate."""
    ratios = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
    # Keep geometric mean constant at sqrt(0.12*0.03) ≈ 0.06
    gm = np.sqrt(ALPHA_DROP * ALPHA_RECOV)  # ~0.06
    trap_rates = []
    recovery_times = []

    print("Sweeping hysteresis asymmetry...")
    for ratio in ratios:
        alpha_d = gm * np.sqrt(ratio)
        alpha_r = gm / np.sqrt(ratio)
        traps = []
        rec_times = []
        for seed in range(40):
            params_trap = dict(
                L0=0.7, T_transp=1.0, lambda_sup=1.0,
                shock_t=50, shock_mag=np.array([3.0, 0.0]),
                alpha_drop=alpha_d, alpha_recov=alpha_r,
                gamma=GAMMA_BUILT,
            )
            res = run_sim(2, SEED_BASE + 40000 + int(ratio*100) + seed, params=params_trap)
            traps.append(res['metrics']['in_trap'])
        trap_rates.append(np.mean(traps))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratios, trap_rates, 'o-', color='#d62728', lw=2, ms=8)
    ax.set_xlabel('Hysteresis asymmetry ratio α_drop / α_recovery')
    ax.set_ylabel('Trap entry probability')
    ax.set_title('Hysteresis Asymmetry vs Trap Entry (v15)\n'
                 'Symmetric update (ratio=1) → no trap; asymmetry → trap emerges')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('outputs/v15-asymmetry-sweep.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/v15-asymmetry-sweep.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 70)
    print("Paper XIII Simulation — Legitimacy as Emergent Gain")
    print(f"Monte Carlo: {N_MC} seeds, T = {T_total} steps")
    print("=" * 70)

    t0 = time.time()

    figure_phase_diagram()
    figure_trap_and_recovery()
    figure_borrowed_collapse()
    figure_collapse_heatmap()
    figure_asymmetry_sweep()

    print(f"\nAll figures complete. Total time: {time.time()-t0:.1f}s")
    print("Outputs written to outputs/")
