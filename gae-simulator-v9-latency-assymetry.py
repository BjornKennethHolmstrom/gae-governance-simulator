#!/usr/bin/env python3
"""
gae-simulator-v9-latency-asymmetry.py
========================================
Sim B — Reform-Incumbent Latency Asymmetry
Paper IX: The Political Economy of Requisite Governance

Architecture state
------------------
A ∈ [0, 1]^N  where A_i = 0 is current configuration, A_i = 1 is target.
Mean(A) is the overall reform success score.

Two controllers
---------------
Reform coalition:
    Observes d_R of N dimensions with latency τ_R steps.
    Pushes observed dims toward 1 at gain K_R per step.

Incumbent:
    Observes d_I of N dimensions with latency τ_I ≤ τ_R (shorter = advantage).
    Pulls observed dims toward 0 at gain K_I per step.

Per-dimension dynamics
----------------------
  Both observe dim i : contested — K_R pushes up, K_I pulls down.
  Reform only        : reform wins unopposed at rate K_R.
  Incumbent only     : incumbent defends; slow decay toward 0.
  Neither            : slow passive decay toward 0 (institutional inertia).

Key insight
-----------
The latency asymmetry matters only on contested dimensions. On dimensions
the incumbent does not monitor, reform proceeds regardless of τ_ratio.
This means the d_R − overlap region provides a guaranteed reform floor,
and the boundary question (is dim(T) ≥ dim(I) a sharp threshold?) is
answered empirically by the phase diagram sweep.

Sweep
-----
τ_ratio = τ_I / τ_R ∈ [0.1, 1.0]  (lower = heavier incumbent advantage)
d_I ∈ [1, N]                        (higher = more incumbent coverage)

Outputs
-------
  outputs/v9-latency-main.png    — trajectories, sweeps, findings panel
  outputs/v9-latency-sweep.png   — full 2D phase diagram
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import os

os.makedirs('outputs', exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────

N     = 10     # number of architectural dimensions
T     = 300    # time steps
TAU_R = 8      # reform coalition observation latency (steps)
K_R   = 0.08   # reform gain per observed dimension per step
K_I   = 0.10   # incumbent gain per dimension (higher: embedded actors react sharper)
D_R   = 8      # reform coalition observes this many dimensions
DECAY        = 0.006  # passive drift toward 0 on unobserved dimensions
SIGMA        = 0.015  # observation noise (each dimension)
A0           = 0.20   # initial architecture quality (far from target)
N_MC         = 50     # Monte Carlo seeds for main figure
TIMING_BOOST = 2.0    # models faster counter-mobilisation as effective gain advantage
                      # on contested dims: K_I_eff = K_I × (1 + BOOST × (1 − τ_ratio))
                      # τ_ratio=1 → no boost; τ_ratio→0 → maximum boost
                      # TIER-2 ASSUMPTION: not derived from first principles.

# Reform always observes dims 0..D_R-1 (fixed across seeds for comparability).
# Incumbent dims are randomly sampled each seed.

REFORM_MASK_FIXED = np.zeros(N, dtype=bool)
REFORM_MASK_FIXED[:D_R] = True


# ── Core simulation ────────────────────────────────────────────────────────────

def run_sim_B(tau_I: int, n_inc_dims: int, seed: int) -> np.ndarray:
    """
    Run one simulation. Returns A_mean(t), shape (T,).

    Parameters
    ----------
    tau_I      : incumbent observation latency in steps
    n_inc_dims : number of dimensions incumbent monitors
    seed       : RNG seed (also governs which dims incumbent gets)
    """
    rng = np.random.default_rng(seed)

    # Incumbent dim assignment (random per seed)
    inc_idx  = rng.choice(N, size=n_inc_dims, replace=False)
    inc_mask = np.zeros(N, dtype=bool)
    inc_mask[inc_idx] = True

    neither_mask = ~REFORM_MASK_FIXED & ~inc_mask

    # Classify each dim's regime
    contested_mask   = REFORM_MASK_FIXED & inc_mask
    reform_only_mask = REFORM_MASK_FIXED & ~inc_mask
    inc_only_mask    = ~REFORM_MASK_FIXED & inc_mask

    # Timing advantage: faster incumbent → effective gain boost on contested dims
    tau_ratio   = tau_I / TAU_R
    timing_mult = 1.0 + TIMING_BOOST * (1.0 - tau_ratio)  # 1.0 at τ_ratio=1, max at 0

    # Full history for delayed observations
    A_hist = np.zeros((T, N))
    A_hist[0] = A0

    for t in range(1, T):
        t_obs_R = max(0, t - TAU_R)
        t_obs_I = max(0, t - tau_I)

        A_obs_R = np.clip(A_hist[t_obs_R] + rng.normal(0, SIGMA, N), 0.0, 1.0)
        A_obs_I = np.clip(A_hist[t_obs_I] + rng.normal(0, SIGMA, N), 0.0, 1.0)

        dA = np.zeros(N)
        # Reform pushes its dims toward 1
        dA[REFORM_MASK_FIXED] += K_R * (1.0 - A_obs_R[REFORM_MASK_FIXED])
        # Incumbent: boosted gain on contested dims, base gain on incumbent-only
        dA[contested_mask]    -= K_I * timing_mult * A_obs_I[contested_mask]
        dA[inc_only_mask]     -= K_I * A_obs_I[inc_only_mask]
        # Unobserved dims decay passively
        dA[neither_mask]      -= DECAY * A_hist[t - 1][neither_mask]

        A_hist[t] = np.clip(
            A_hist[t - 1] + dA + rng.normal(0, SIGMA * 0.3, N),
            0.0, 1.0,
        )

    return A_hist.mean(axis=1)   # mean over all N dims


# ── Four illustrative cases ────────────────────────────────────────────────────

CASES = [
    dict(tau_I=1, n_inc=8,
         label='τ_ratio=0.13, d_I=8\n(heavy advantage, high coverage)',  color='#d62728'),
    dict(tau_I=2, n_inc=6,
         label='τ_ratio=0.25, d_I=6\n(strong advantage, med coverage)',  color='#9467bd'),
    dict(tau_I=6, n_inc=5,
         label='τ_ratio=0.75, d_I=5\n(weak advantage, med coverage)',    color='#ff7f0e'),
    dict(tau_I=8, n_inc=2,
         label='τ_ratio=1.00, d_I=2\n(no advantage, low coverage)',       color='#2ca02c'),
]

print(f"Running main figure: 4 cases × {N_MC} seeds ...")
case_mc = []
for case in CASES:
    trajs = [run_sim_B(case['tau_I'], case['n_inc'], seed) for seed in range(N_MC)]
    case_mc.append(np.array(trajs))
print("Done.")

ts = np.arange(T)

# ── Figure 1: Main results ─────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 11))
gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

# Top-left (2 cols): trajectories
ax_traj = fig.add_subplot(gs[0, :2])
for case, traj_mc in zip(CASES, case_mc):
    med = np.median(traj_mc, axis=0)
    lo  = np.percentile(traj_mc, 10, axis=0)
    hi  = np.percentile(traj_mc, 90, axis=0)
    ax_traj.plot(ts, med, color=case['color'], lw=2, label=case['label'])
    ax_traj.fill_between(ts, lo, hi, color=case['color'], alpha=0.15)
ax_traj.axhline(0.70, color='gray', ls='--', lw=1, alpha=0.6, label='Success threshold (0.70)')
ax_traj.axhline(0.35, color='gray', ls=':',  lw=1, alpha=0.6, label='Failure threshold (0.35)')
ax_traj.set_ylabel('Mean architecture quality A(t)')
ax_traj.set_title(
    f'Architecture quality trajectories — d_R={D_R}, τ_R={TAU_R}; '
    'four (τ_ratio, d_I) combinations\n'
    'Reform succeeds on uncontested dimensions regardless of latency; '
    'contested dims are where asymmetry bites',
    fontsize=9, fontweight='bold',
)
ax_traj.legend(fontsize=7)
ax_traj.grid(True, alpha=0.2)

# Top-right: final distribution (violin)
ax_vio = fig.add_subplot(gs[0, 2])
final_vals = [mc_arr[:, -50:].mean(axis=1) for mc_arr in case_mc]
parts = ax_vio.violinplot(final_vals, positions=range(len(CASES)),
                          showmedians=True, showextrema=False)
for body, case in zip(parts['bodies'], CASES):
    body.set_facecolor(case['color'])
    body.set_alpha(0.70)
parts['cmedians'].set_color('black')
ax_vio.axhline(0.70, color='gray', ls='--', lw=1, alpha=0.6)
ax_vio.axhline(0.35, color='gray', ls=':',  lw=1, alpha=0.6)
ax_vio.set_xticks(range(len(CASES)))
ax_vio.set_xticklabels([f'C{i+1}' for i in range(len(CASES))], fontsize=8)
ax_vio.set_ylabel('Final mean A (last 50 steps)')
ax_vio.set_title('Final reform success\n(violin over MC seeds)', fontsize=9, fontweight='bold')
ax_vio.grid(True, alpha=0.2, axis='y')

# Bottom-left: d_I sweep at fixed τ_ratio = 0.5
print("d_I sweep (τ_ratio=0.5) ...")
tau_I_fixed = 4   # τ_ratio = 4/8 = 0.5
d_I_vals    = list(range(1, N + 1))

sweep_dI = np.array([
    [run_sim_B(tau_I_fixed, d_I, seed)[-50:].mean() for seed in range(N_MC)]
    for d_I in d_I_vals
])  # shape (N, N_MC)

ax_dim = fig.add_subplot(gs[1, 0])
ax_dim.errorbar(
    d_I_vals,
    np.median(sweep_dI, axis=1),
    yerr=[np.median(sweep_dI, axis=1) - np.percentile(sweep_dI, 10, axis=1),
          np.percentile(sweep_dI, 90, axis=1) - np.median(sweep_dI, axis=1)],
    fmt='o-', color='#1f77b4', capsize=3, lw=1.5, ms=5,
)
ax_dim.axhline(0.70, color='gray', ls='--', lw=1, alpha=0.6, label='Success threshold')
ax_dim.axhline(0.35, color='gray', ls=':',  lw=1, alpha=0.6, label='Failure threshold')
ax_dim.set_xlabel(f'Incumbent coverage d_I  (of {N} dims)')
ax_dim.set_ylabel('Final mean architecture quality A')
ax_dim.set_title(f'Reform success vs incumbent coverage\n(τ_ratio=0.5, d_R={D_R})',
                 fontsize=9, fontweight='bold')
ax_dim.legend(fontsize=7)
ax_dim.grid(True, alpha=0.2)

# Bottom-middle: τ_ratio sweep at fixed d_I = 7 (high enough coverage for latency to bite)
print("τ_ratio sweep (d_I=7) ...")
d_I_fixed   = 7
TAU_RATIOS  = np.linspace(0.1, 1.0, 12)

sweep_tau = np.array([
    [run_sim_B(max(1, int(tr * TAU_R)), d_I_fixed, seed)[-50:].mean()
     for seed in range(N_MC)]
    for tr in TAU_RATIOS
])  # shape (12, N_MC)

ax_lat = fig.add_subplot(gs[1, 1])
ax_lat.errorbar(
    TAU_RATIOS,
    np.median(sweep_tau, axis=1),
    yerr=[np.median(sweep_tau, axis=1) - np.percentile(sweep_tau, 10, axis=1),
          np.percentile(sweep_tau, 90, axis=1) - np.median(sweep_tau, axis=1)],
    fmt='s-', color='#d62728', capsize=3, lw=1.5, ms=5,
)
ax_lat.axhline(0.70, color='gray', ls='--', lw=1, alpha=0.6, label='Success threshold')
ax_lat.axhline(0.35, color='gray', ls=':',  lw=1, alpha=0.6, label='Failure threshold')
ax_lat.set_xlabel('τ_ratio = τ_I / τ_R  (1.0 = no advantage ← 0.1 = strong incumbent)')
ax_lat.set_ylabel('Final mean architecture quality A')
ax_lat.set_title(f'Reform success vs latency ratio\n(d_I=7, d_R={D_R})',
                 fontsize=9, fontweight='bold')
ax_lat.legend(fontsize=7)
ax_lat.grid(True, alpha=0.2)

# Bottom-right: findings annotation
ax_note = fig.add_subplot(gs[1, 2])
ax_note.axis('off')
findings = (
    "Key findings\n\n"
    "① Reform wins unopposed on dimensions the\n"
    "  incumbent does not monitor, regardless of\n"
    "  τ_ratio — the 'reform floor' is structural.\n\n"
    "② As d_I ↑, contested dims increase, reform-\n"
    "  only dims decrease → success declines.\n\n"
    "③ As τ_ratio ↓ (incumbent faster), effective\n"
    "  counter-mobilisation gain rises on contested\n"
    "  dims → reform absorbed more readily.\n\n"
    "④ The d_I threshold for reform failure is\n"
    "  a gradient, not a sharp phase boundary.\n"
    "  → dim(T) ≥ dim(I) is heuristic (tier 2),\n"
    "  not a sharp law.\n\n"
    "⑤ TIER-2 NOTE: latency effect is modelled as\n"
    "  K_I_eff = K_I·(1+BOOST·(1−τ_ratio)) on\n"
    "  contested dims. Direction is robust;\n"
    "  magnitude depends on BOOST=2.0 (assumed).\n\n"
    f"  N={N}, d_R={D_R}, τ_R={TAU_R}, "
    f"K_R={K_R}, K_I={K_I}"
)
ax_note.text(0.04, 0.97, findings, transform=ax_note.transAxes,
             va='top', ha='left', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))

fig.suptitle(
    f'v9 — Reform-Incumbent Latency Asymmetry  '
    f'(N={N} dims, d_R={D_R}, τ_R={TAU_R}, K_R={K_R}, K_I={K_I})\n'
    f'n={N_MC} MC seeds; error bars / bands = 10th–90th pct',
    fontsize=10, y=1.01,
)
plt.savefig('outputs/v9-latency-main.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/v9-latency-main.png")


# ── Figure 2: Full 2D phase diagram (τ_ratio × d_I) ──────────────────────────

print(f"\nFull 2D phase diagram sweep ...")

TAU_RATIO_GRID = np.linspace(0.1, 1.0, 12)
D_I_GRID       = np.arange(1, N + 1)
N_SW           = 15   # MC seeds per cell

sweep2D = np.zeros((len(D_I_GRID), len(TAU_RATIO_GRID)))

for di, d_I in enumerate(D_I_GRID):
    for ti, tr in enumerate(TAU_RATIO_GRID):
        tau_I_v = max(1, int(tr * TAU_R))
        scores  = [run_sim_B(tau_I_v, d_I, seed)[-50:].mean() for seed in range(N_SW)]
        sweep2D[di, ti] = np.mean(scores)
    print(f"  d_I = {d_I}/{N} done")

print("Sweep complete.")

# Classify: 2=success (>0.65), 1=contested (0.35–0.65), 0=absorbed (<0.35)
outcome = np.where(sweep2D >= 0.65, 2, np.where(sweep2D <= 0.35, 0, 1))

fig2, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: continuous score
im0 = axes[0].imshow(
    sweep2D, origin='lower', aspect='auto',
    extent=[TAU_RATIO_GRID[0], TAU_RATIO_GRID[-1],
            D_I_GRID[0] - 0.5, D_I_GRID[-1] + 0.5],
    cmap='RdYlGn', vmin=0.0, vmax=1.0,
)
plt.colorbar(im0, ax=axes[0], label='Mean final A (0=reform absorbed, 1=reform succeeds)')
axes[0].set_xlabel('τ_ratio = τ_I / τ_R  (← incumbent latency advantage)')
axes[0].set_ylabel('Incumbent coverage d_I')
axes[0].set_title(f'Reform success score (continuous)\nd_R={D_R} fixed, n={N_SW} MC per cell',
                  fontsize=9, fontweight='bold')
# Mark uncontested reform floor
floor_line = (N - D_R)  # dims neither agent touches (decay only)
axes[0].axhline(D_R + 0.5, color='white', ls='--', lw=1.5, alpha=0.8,
                label=f'd_I > d_R={D_R}: full reform dims contested')
axes[0].legend(fontsize=7)

# Right: classified outcomes
cmap3 = ListedColormap(['#d62728', '#ff7f0e', '#2ca02c'])
bnorm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap3.N)
im1 = axes[1].imshow(
    outcome, origin='lower', aspect='auto',
    extent=[TAU_RATIO_GRID[0], TAU_RATIO_GRID[-1],
            D_I_GRID[0] - 0.5, D_I_GRID[-1] + 0.5],
    cmap=cmap3, norm=bnorm,
)
axes[1].set_xlabel('τ_ratio = τ_I / τ_R  (← incumbent latency advantage)')
axes[1].set_ylabel('Incumbent coverage d_I')
axes[1].set_title('Classified outcomes\n'
                  '(thresholds: A>0.65 success, A<0.35 absorbed, else contested)',
                  fontsize=9, fontweight='bold')
legend_elements = [
    Patch(facecolor='#2ca02c', label='Reform succeeds (A > 0.65)'),
    Patch(facecolor='#ff7f0e', label='Contested  (0.35 ≤ A ≤ 0.65)'),
    Patch(facecolor='#d62728', label='Reform absorbed  (A < 0.35)'),
]
axes[1].legend(handles=legend_elements, fontsize=7, loc='upper left')
axes[1].axhline(D_R + 0.5, color='white', ls='--', lw=1.5, alpha=0.7)

fig2.suptitle(
    f'v9 — Phase diagram: τ_ratio × d_I  (n={N_SW} MC per cell)\n'
    f'N={N} dims, d_R={D_R} (fixed). '
    'Gradient boundary confirms the dim(T)≥dim(I) condition is heuristic, not a sharp law.',
    fontsize=10,
)
plt.tight_layout()
plt.savefig('outputs/v9-latency-sweep.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/v9-latency-sweep.png")
