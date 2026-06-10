#!/usr/bin/env python3
"""
gae-simulator-v11-epistemic-monoculture.py
============================================
Sim D — Epistemic Monoculture Collapse
Paper X: Requisite Observer Diversity

Environment
-----------
X ∈ ℝ⁵  (economic, social, ecological, technological, structural fragility)
Stationary until t=100, then dimension 5 begins a slow, persistent drift
(toward failure) while dimensions 1‑4 remain stable.

Observers
---------
N = 20 organisations.  Each can be:

  • Independent (I)  – sees 3 random dimensions (may include dim 5),
      noise σ²=0.5, errors uncorrelated (ρ=0 across I’s).
      Cost c_ind = 1.0 + liability penalty L(f) (not used in fixed‑fraction runs).

  • Shared (S)      – sees dims 1‑4 only, noise σ²=0.2,
      errors perfectly correlated (ρ=1 across all S’s).
      Cost c_shared = 0.5, no liability penalty.

Governance controller
-----------------------
Computes ensemble mean & spread for each dimension.
For dim 5: if spread > S_high = 1.5 → Precautionary Gate triggers,
gradually improving independent observers’ resolution on dim 5.

Scenarios (all organisations are held fixed to the given counts)
---------------------------------------------------------------
  • Diverse   : 20 I, 0 S
  • Monoculture: 0 I, 20 S
  • Mixed     : 5 I, 15 S  (protected I’s)

Outputs
-------
  outputs/v11‑monoculture‑trajectories.png   – X₅ truth, estimate & spread
  outputs/v11‑monoculture‑phase.png          – (X₁, X₅) phase portrait
  outputs/v11‑monoculture‑sweep.png          – failure probability vs. protected fraction

"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────

T          = 200
N_OBS      = 20
DIMS       = 5
DIM_BLIND  = 4          # dimension 5 = index 4

# Dynamics
A_DIAG       = 0.95      # for dims 0‑3 (post‑shift dim4 is random walk)
Q_SCALE      = 0.1
MU_DRIFT     = 0.10      # per‑step drift of dim‑5 after t=100 (random walk)
COUPLING     = 0.08      # X₅ → X₁ coupling (increased for visible divergence)
CONTROL_EFF  = 0.70      # fraction of drift offset when gate active
CONTROL_RAMP = 0.25      # additional effectiveness as sensing improves (raised)
X0           = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

# Observers
SIGMA_IND   = np.sqrt(0.5)   # ≈0.707
SIGMA_SHR   = np.sqrt(0.2)   # ≈0.447
C_SHR       = np.zeros((4, DIMS))
C_SHR[:4, :4] = np.eye(4)    # sees dims 1‑4, blind to dim‑5
DIM5_IMPROVE = 0.8
SIGMA_FLOOR  = np.sqrt(0.1)

# Precautionary gate
S_HIGH       = 0.8    # kept for spread visualization
DETECT_THRESH = -1.5  # gate fires when independent mean-estimate of dim5 drops below this

# Failure
X5_FAIL = -8.0

# Monte Carlo
N_MC    = 100
SEEDS   = np.arange(N_MC)


# ── Simulation ─────────────────────────────────────────────────────────────────

def init_observer(strategy: str, rng: np.random.Generator):
    obs = {'strategy': strategy}
    if strategy == 'independent':
        dims = rng.choice(DIMS, size=3, replace=False)
        C = np.zeros((3, DIMS))
        for i, d in enumerate(dims):
            C[i, d] = 0.70
            others = [j for j in range(DIMS) if j != d]
            n_other = rng.integers(1, 3)
            chosen = rng.choice(others, size=n_other, replace=False)
            for j in chosen:
                C[i, j] += 0.30 / n_other
        obs['C'] = C
        obs['sigma'] = np.full(3, SIGMA_IND)
        obs['dims'] = dims
    else:
        obs['C'] = C_SHR.copy()
        obs['sigma'] = np.full(4, SIGMA_SHR)
        obs['dims'] = np.arange(4)
    return obs


def run_sim_D(n_I: int, n_S: int, seed: int) -> dict:
    assert n_I + n_S == N_OBS
    rng = np.random.default_rng(seed)

    # ── Initialise observers BEFORE the main loop ──────────────────────────
    observers = [init_observer('independent', rng) for _ in range(n_I)]
    observers += [init_observer('shared', rng) for _ in range(n_S)]

    # shared observers all receive the *same* noise → zero internal spread
    shared_noise = rng.normal(0, SIGMA_SHR, (T, 4))

    X = np.zeros((T, DIMS))
    X[0] = X0.copy()
    est_mean        = np.full((T, DIMS), np.nan)
    est_spread      = np.full((T, DIMS), np.nan)
    gate_active     = np.zeros(T, dtype=bool)
    control_history = np.zeros(T)
    control_eff     = 0.0

    for t in range(T):
        # ── 1. Observe at time t ───────────────────────────────────────────
        all_by_dim = {d: [] for d in range(DIMS)}   # all observers → mean
        ind_by_dim = {d: [] for d in range(DIMS)}   # independent only → spread

        for obs in observers:
            if obs['strategy'] == 'independent':
                y = obs['C'] @ X[t] + rng.normal(0, obs['sigma'])
                for i, d in enumerate(obs['dims']):
                    all_by_dim[d].append(y[i])
                    ind_by_dim[d].append(y[i])   # spread signal from independent only
            else:
                y = obs['C'] @ X[t] + shared_noise[t]
                for i, d in enumerate(obs['dims']):
                    all_by_dim[d].append(y[i])   # contributes to mean but NOT spread

        for d in range(DIMS):
            if all_by_dim[d]:
                est_mean[t, d] = np.mean(all_by_dim[d])
            if len(ind_by_dim[d]) >= 2:          # need ≥2 for std to be defined
                est_spread[t, d] = np.std(ind_by_dim[d])

        # ── 2. Precautionary gate ──────────────────────────────────────────────
        # Triggered by drift in independent-ensemble mean of dim5.
        # est_mean[t, DIM_BLIND] is NaN for monoculture (no one observes dim5)
        # → gate can never fire → drift goes uncontrolled → monoculture fails.
        # For diverse/mixed: fires once the independent ensemble detects X5 below
        # DETECT_THRESH.  est_spread is kept for visualization (not the trigger).
        x5_ind_mean = est_mean[t, DIM_BLIND]
        gate_fires = (not np.isnan(x5_ind_mean)) and (x5_ind_mean < DETECT_THRESH)
        if gate_fires:
            gate_active[t] = True
            control_eff = min(CONTROL_EFF + CONTROL_RAMP, control_eff + 0.12)
            for obs in observers:
                if obs['strategy'] == 'independent' and DIM_BLIND in obs['dims']:
                    idx = list(obs['dims']).index(DIM_BLIND)
                    obs['sigma'][idx] = max(SIGMA_FLOOR,
                                            obs['sigma'][idx] * DIM5_IMPROVE)
        else:
            control_eff = max(0.0, control_eff - 0.01)

        control_history[t] = control_eff

        # ── 3. Simulate environment t → t+1 using current control_eff ─────
        if t < T - 1:
            w = rng.normal(0, Q_SCALE, DIMS)
            X[t+1, :4] = A_DIAG * X[t, :4] + w[:4]
            if t < 99:
                X[t+1, DIM_BLIND] = A_DIAG * X[t, DIM_BLIND] + w[DIM_BLIND]
            else:
                # control_eff from this step's observations now offsets drift
                X[t+1, DIM_BLIND] = (X[t, DIM_BLIND]
                                      - MU_DRIFT * (1.0 - control_eff)
                                      + w[DIM_BLIND])
                X[t+1, 0] += COUPLING * X[t, DIM_BLIND]

    collapsed = np.any(X[:, DIM_BLIND] < X5_FAIL)

    return {
        'X':               X,
        'est_mean':        est_mean,
        'est_spread':      est_spread,
        'gate_active':     gate_active,
        'control_history': control_history,
        'collapsed':       collapsed,
    }


# ── Monte Carlo ────────────────────────────────────────────────────────────────

scenarios = {
    'Diverse':    (20, 0),
    'Mixed':      (5, 15),
    'Monoculture': (0, 20),
}

mc = {}
for name, (ni, ns) in scenarios.items():
    print(f"Running {name} ({ni}I, {ns}S) × {N_MC} seeds ...")
    Xs, means, spreads, gates, controls, coll = [], [], [], [], [], []
    for seed in SEEDS:
        res = run_sim_D(ni, ns, seed)
        Xs.append(res['X'])
        means.append(res['est_mean'])
        spreads.append(res['est_spread'])
        gates.append(res['gate_active'])
        controls.append(res['control_history'])
        coll.append(res['collapsed'])
    mc[name] = {
        'X': np.array(Xs),
        'est_mean': np.array(means),
        'est_spread': np.array(spreads),
        'gate': np.array(gates),
        'control': np.array(controls),
        'collapsed': np.array(coll),
    }
    print(f"  → failure prob: {np.mean(coll):.3f}")

print("Done.\n")


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_band(ax, data, color, label, ts, alpha=0.2, lw=2, ls='-'):
    med = np.median(data, axis=0)
    lo  = np.percentile(data, 10, axis=0)
    hi  = np.percentile(data, 90, axis=0)
    ax.plot(ts, med, color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(ts, lo, hi, color=color, alpha=alpha)


ts = np.arange(T)

# ── Figure D1: X₅ truth & estimate ────────────────────────────────────────────

fig1, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

true5_mono    = mc['Monoculture']['X'][0, :, DIM_BLIND]   # uncontrolled — drifts to failure
true5_diverse = mc['Diverse']['X'][0, :, DIM_BLIND]       # controlled   — stabilised
axes[0].plot(ts, true5_mono,    color='#d62728', lw=2, label='X₅ — monoculture (uncontrolled drift)')
axes[0].plot(ts, true5_diverse, color='#1f77b4', lw=2, label='X₅ — diverse (control applied at gate)')
axes[0].axhline(X5_FAIL, color='red', ls='--', lw=1, alpha=0.6, label=f'Failure threshold ({X5_FAIL})')
axes[0].axvline(100, color='gray', ls=':', alpha=0.5, label='Regime shift (t=100)')
axes[0].set_ylabel('X₅')
axes[0].set_title('Dimension 5 (structural fragility) — truth', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.2)

ax = axes[1]
for name, color in zip(['Monoculture', 'Mixed', 'Diverse'],
                        ['#d62728', '#2ca02c', '#1f77b4']):
    mean5 = mc[name]['est_mean'][:, :, DIM_BLIND]
    if name == 'Monoculture':
        ax.text(120, -1.5, 'Monoculture: no estimate\n(dim‑5 unobserved)',
                color='#d62728', fontsize=9, fontstyle='italic')
        continue
    plot_band(ax, mean5, color, name, ts, alpha=0.15)
ax.plot(ts, true5_diverse, color='gray', lw=1.5, alpha=0.5, label='True X₅ (diverse seed 0)')
ax.axhline(X5_FAIL, color='red', ls='--', lw=1, label=f'Failure threshold ({X5_FAIL})')
ax.set_xlabel('Time step')
ax.set_ylabel('Controller estimate of X₅')
ax.set_title('Governance controller\'s estimate of X₅  (median & 10–90% band)',
             fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.2)

fig1.suptitle('v11 — Epistemic Monoculture Collapse: X₅ trajectory',
              fontsize=11, y=1.01)
plt.tight_layout()
fig1.savefig('outputs/v11-monoculture-trajectories.png', dpi=150)
plt.close()


# ── Figure D2: Max ensemble spread (trigger signal) ──────────────────────────

fig2, ax2 = plt.subplots(figsize=(12, 5))
for name, color in zip(['Diverse', 'Mixed', 'Monoculture'],
                        ['#1f77b4', '#2ca02c', '#d62728']):
    if name == 'Monoculture':
        # No independent observers → spread is always zero (shared noise is correlated)
        ax2.plot(ts, np.zeros(T), color=color, lw=2,
                 label=f'{name} (spread = 0: no independent observers)')
    else:
        max_sp = np.nanmax(mc[name]['est_spread'], axis=2)  # N_MC × T
        plot_band(ax2, max_sp, color, name, ts, alpha=0.15)
ax2.axhline(S_HIGH, color='red', ls='--', lw=1.5, label=f'S_high = {S_HIGH} (visualization reference)')
ax2.set_xlabel('Time step')
ax2.set_ylabel('Max ensemble spread across dimensions\n(independent observers only)')
ax2.set_title('Independent-observer ensemble spread — epistemic divergence signal\n'
              '(Monoculture has zero independent observers: spread = 0, threat invisible;\n'
              'gate now triggers on dim-5 mean drift, not spread)',
              fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.2)
plt.tight_layout()
fig2.savefig('outputs/v11-monoculture-spread.png', dpi=150)
plt.close()


# ── Figure D3: Phase portrait (X₁, X₅) ──────────────────────────────────────

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
for ax, (name, color) in zip(axes3,
    [('Diverse', '#1f77b4'), ('Mixed', '#2ca02c'), ('Monoculture', '#d62728')]):
    X_arr = mc[name]['X']
    n_show = min(15, N_MC)
    for i in range(n_show):
        ax.plot(X_arr[i, :, 0], X_arr[i, :, DIM_BLIND],
                color=color, alpha=0.3, lw=0.8)
    ax.axhline(X5_FAIL, color='red', ls='--', lw=1)
    ax.set_title(f'{name}\n(collapse: {np.mean(mc[name]["collapsed"]):.0%})',
                 fontweight='bold')
    ax.set_xlabel('X₁ (visible dimension)')
    if ax is axes3[0]:
        ax.set_ylabel('X₅ (structural fragility)')
    ax.grid(True, alpha=0.2)
fig3.suptitle('Phase portrait: visible dimension vs. hidden fragility\n'
              '(Coupling: X₅ drift → X₁ drag after t=100)',
              fontsize=11)
plt.tight_layout()
fig3.savefig('outputs/v11-monoculture-phase.png', dpi=150)
plt.close()


# ── Figure D4: Parameter sweep (protected independent fraction) ────────────────

prot_fracs = np.arange(0, 21) / N_OBS   # 0.0 to 1.0
fail_probs = np.zeros_like(prot_fracs)
N_SWEEP = 200

for idx, frac in enumerate(prot_fracs):
    n_I = int(frac * N_OBS)
    n_S = N_OBS - n_I
    fails = 0
    for s in range(N_SWEEP):
        res = run_sim_D(n_I, n_S, seed=40000 + s)
        if res['collapsed']:
            fails += 1
    fail_probs[idx] = fails / N_SWEEP
    print(f"  frac {frac:.2f} ({n_I:2d}I) → fail prob {fail_probs[idx]:.3f}")

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(prot_fracs, fail_probs, 'o-', color='#d62728', lw=2, markersize=6)
ax4.axhline(1.0, color='gray', ls=':', alpha=0.5)
ax4.axvline(0.15, color='blue', ls='--', alpha=0.4,
            label='Paper X threshold: ~15% protected')
ax4.set_xlabel('Protected independent observer fraction')
ax4.set_ylabel('Failure probability')
ax4.set_title(f'Failure probability vs. protected observer fraction\n'
              f'(μ={MU_DRIFT}, coupling={COUPLING}, '
              f'detect_thresh={DETECT_THRESH}, '
              f'fail threshold={X5_FAIL}, n={N_SWEEP} seeds/point)',
              fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.2)
ax4.set_ylim(0, 1.05)
plt.tight_layout()
fig4.savefig('outputs/v11-monoculture-sweep.png', dpi=150)
plt.close()

print("All figures saved to outputs/")
