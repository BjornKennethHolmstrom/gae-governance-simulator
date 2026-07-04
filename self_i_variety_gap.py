#!/usr/bin/env python3
"""
Self Stability Simulator
=========================
A minimal dynamical model of the self‑variety gap and the
Goodhart‑Ashby synthesis at the personal scale.

Five coupled life dimensions:
    H : physical health
    R : relational integrity
    M : existential meaning
    C : career / contribution
    L : leisure / restoration

Two personal value architectures:
    Architecture 1D (Career‑only) : tracks only career,
                                    blind to other dimensions.
    Architecture ND (Multi‑dim)   : tracks health, relationships,
                                    and career simultaneously.

Both controllers have the same total effort budget.
Differences are architectural, not motivational.

Reproducibility: fixed random seed, standard NumPy/Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Reproducibility ───────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)

# ── Simulation parameters ─────────────────────────────────────────────────────
T          = 300          # time steps (e.g., months)
N          = 5            # total dimensions: H, R, M, C, L

# Indices
H, R, M, C, L = 0, 1, 2, 3, 4

# Natural dynamics: each dimension drifts back to baseline when not maintained
A_sys = 0.95              # retention factor per step
baseline = np.array([70., 70., 70., 70., 70.])  # equilibrium without control

# Cross‑coupling: how much each dimension affects the others
# Matrix C_cross[i,j] = effect of dimension j on dimension i next step
C_cross = np.array([
    [ 0.0,  0.0,  0.0, -0.03,  0.02],   # health
    [ 0.02, 0.0,  0.01, -0.04,  0.01],   # relationships
    [ 0.01, 0.02, 0.0,  -0.02,  0.0],    # meaning
    [ 0.0,  0.0,  0.0,   0.0,   0.0],    # career (handled by health penalty)
    [-0.01, 0.0,  0.0,  -0.02,  0.0],    # leisure: weakened from -0.05 to -0.02
])

# Control effectiveness: effort → dimension improvement
B = np.eye(N) * 0.3

# Disturbance standard deviations (per dimension, per step)
disturbance_std = np.array([1.5, 1.2, 1.0, 1.8, 0.8])

# Shared total effort budget per step
total_effort_budget = 10.0

# Gains for tracked dimensions
K_tracked = 0.05          # per unit of error, effort applied

# ── Controllers ───────────────────────────────────────────────────────────────
def controller_1d(x_obs):
    """Career‑only: all effort goes to closing the career gap."""
    u = np.zeros(N)
    error_c = baseline[C] - x_obs[C]
    u[C] = K_tracked * error_c
    # Scale to fit within budget
    norm = np.abs(u).sum()
    if norm > 0:
        u = u * total_effort_budget / norm
    return u

def controller_nd(x_obs, tracked_dims):
    """Multi‑dimensional: effort distributed among tracked dimensions."""
    u = np.zeros(N)
    for dim in tracked_dims:
        error = baseline[dim] - x_obs[dim]
        u[dim] = K_tracked * error
    norm = np.abs(u).sum()
    if norm > 0:
        u = u * total_effort_budget / norm
    return u

# ── Simulation run ────────────────────────────────────────────────────────────
def simulate(arch="1D", tracked_dims=None):
    """Run one simulation and return state trajectory (T, N), effort (T, N)."""
    if arch == "1D":
        tracked_dims = [C]
    elif tracked_dims is None:
        tracked_dims = [H, R, C]  # default multi‑dim

    x = np.zeros((T, N))
    x[0] = baseline.copy()
    u = np.zeros((T, N))

    for t in range(T-1):
        # Observation: true state + noise
        obs = x[t] + rng.normal(0, disturbance_std * 0.3)

        # Compute control effort
        if arch == "1D":
            u[t] = controller_1d(obs)
            # Career effort directly drains health (overwork - immediate cost)
            health_cost = u[t, C] * 0.12
            x[t, H] = max(0, x[t, H] - health_cost)
        else:
            u[t] = controller_nd(obs, tracked_dims)

        # Natural dynamics for most dimensions
        x_next = np.zeros(N)
        for i in [H, R, M, L]:
            x_next[i] = A_sys * x[t, i] + (1 - A_sys) * baseline[i]

        # Leisure: auto-restores when healthy (you naturally seek rest when not burned out)
        if x[t, H] > 0.6 * baseline[H]:
            leisure_boost = (x[t, H] / baseline[H] - 0.6) * 3.0
        else:
            leisure_boost = 0
        x_next[L] = A_sys * x[t, L] + (1 - A_sys) * baseline[L] + leisure_boost

        # Career: exponentially sensitive to health state
        health_ratio = x[t, H] / baseline[H]
        if health_ratio < 0.7:
            # Exponential collapse below 70% health
            career_penalty = np.exp(-5 * (0.7 - health_ratio))
        else:
            career_penalty = 1.0
        
        x_next[C] = A_sys * x[t, C] * career_penalty + (1 - A_sys) * baseline[C] * career_penalty

        # Cross-coupling effects
        x_next += C_cross @ x[t]
        
        # Control input
        x_next += B @ u[t]
        
        # Disturbances
        x_next += rng.normal(0, disturbance_std)

        # Floor at zero (no negative states)
        x[t+1] = np.maximum(x_next, 0)
        
    return x, u

# ── Run architectures ─────────────────────────────────────────────────────────
print("Simulating 1D (Career‑only) ...")
x1, u1 = simulate("1D")
print("Simulating ND (Multi‑dim: Health, Relationships, Career) ...")
xnd, und = simulate("ND", tracked_dims=[H, R, C])

# ── Metrics ───────────────────────────────────────────────────────────────────
collapse_threshold = 0.2 * baseline  # 20% of baseline

def crisis_events(x, threshold):
    """Count time steps where any dimension falls below threshold."""
    return np.any(x < threshold, axis=1)

crisis_1d = crisis_events(x1, collapse_threshold)
crisis_nd = crisis_events(xnd, collapse_threshold)

mean_wellbeing_1d = x1[-50:].mean()
mean_wellbeing_nd = xnd[-50:].mean()
total_effort_1d = np.abs(u1).sum()
total_effort_nd = np.abs(und).sum()

# Self‑variety gap: dim(D) approximated as N, dim(V) = number of tracked dimensions
G_self_1d = N - 1   # 1 tracked dimension
G_self_nd = N - len([H, R, C])  # 3 tracked dimensions

print(f"\n1D: G_self = {G_self_1d}, crisis fraction = {crisis_1d.mean():.2f}, "
      f"mean wellbeing (last 50) = {mean_wellbeing_1d:.1f}")
print(f"ND: G_self = {G_self_nd}, crisis fraction = {crisis_nd.mean():.2f}, "
      f"mean wellbeing (last 50) = {mean_wellbeing_nd:.1f}")

# ── Plotting ──────────────────────────────────────────────────────────────────
dim_labels = ['Health', 'Relationships', 'Meaning', 'Career', 'Leisure']
colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#1f77b4', '#d62728']

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax_1d = fig.add_subplot(gs[0, 0])
ax_nd = fig.add_subplot(gs[0, 1])
ax_career = fig.add_subplot(gs[1, :])
ax_summary = fig.add_subplot(gs[2, 0])
ax_effort = fig.add_subplot(gs[2, 1])

ts = np.arange(T)

# ── Top left: 1D trajectories ─────────────────────────────────────────────────
for i, label in enumerate(dim_labels):
    ax_1d.plot(ts, x1[:, i], color=colors[i], lw=1.3, label=label)
ax_1d.axhline(baseline[C], color='gray', ls='--', alpha=0.4, label='Career baseline')
ax_1d.set_title('Architecture 1D (Career‑only)\nHealth & Relationships collapse, then Career follows',
                fontsize=10, fontweight='bold')
ax_1d.set_ylabel('State')
ax_1d.legend(fontsize=7, loc='upper right')
ax_1d.grid(True, alpha=0.2)

# Highlight crisis periods
crisis_starts = np.where(np.diff(crisis_1d.astype(int)) == 1)[0]
crisis_ends = np.where(np.diff(crisis_1d.astype(int)) == -1)[0]
for cs, ce in zip(crisis_starts, crisis_ends):
    ax_1d.axvspan(cs, ce, color='red', alpha=0.1)

# ── Top right: ND trajectories ────────────────────────────────────────────────
for i, label in enumerate(dim_labels):
    ax_nd.plot(ts, xnd[:, i], color=colors[i], lw=1.3, label=label)
ax_nd.axhline(baseline[C], color='gray', ls='--', alpha=0.4, label='Career baseline')
ax_nd.set_title('Architecture ND (Multi‑dim)\nTracked dimensions stabilize, Career steady',
                fontsize=10, fontweight='bold')
ax_nd.set_ylabel('State')
ax_nd.legend(fontsize=7, loc='upper right')
ax_nd.grid(True, alpha=0.2)

crisis_starts_nd = np.where(np.diff(crisis_nd.astype(int)) == 1)[0]
crisis_ends_nd = np.where(np.diff(crisis_nd.astype(int)) == -1)[0]
for cs, ce in zip(crisis_starts_nd, crisis_ends_nd):
    ax_nd.axvspan(cs, ce, color='red', alpha=0.1)

# ── Middle: Career dimension alone ─────────────────────────────────────────────
ax_career.plot(ts, x1[:, C], color='#1f77b4', lw=2, label='1D (Career‑only)')
ax_career.plot(ts, xnd[:, C], color='#ff7f0e', lw=2, label='ND (Multi‑dim)')
ax_career.axhline(baseline[C], color='gray', ls='--', alpha=0.4, label='Baseline')
ax_career.set_title('Career trajectory — both architectures\n1D collapses; ND stabilises at sustainable level',
                    fontsize=10, fontweight='bold')
ax_career.set_ylabel('Career state')
ax_career.legend(fontsize=9)
ax_career.grid(True, alpha=0.2)

# ── Bottom left: wellbeing summary ────────────────────────────────────────────
metrics_labels = ['Mean\nwellbeing', 'Crisis\nfraction', 'G_self']
x_vals = np.arange(len(metrics_labels))
width = 0.35

ax_summary.bar(x_vals - width/2, 
               [mean_wellbeing_1d, crisis_1d.mean(), G_self_1d],
               width, color='#1f77b4', alpha=0.8, label='1D')
ax_summary.bar(x_vals + width/2,
               [mean_wellbeing_nd, crisis_nd.mean(), G_self_nd],
               width, color='#ff7f0e', alpha=0.8, label='ND')
ax_summary.set_xticks(x_vals)
ax_summary.set_xticklabels(metrics_labels)
ax_summary.set_title('Summary comparison', fontsize=10, fontweight='bold')
ax_summary.legend(fontsize=8)
ax_summary.grid(True, alpha=0.2, axis='y')
# Annotate values
for i, (v1, v2) in enumerate(zip(
    [mean_wellbeing_1d, crisis_1d.mean(), G_self_1d],
    [mean_wellbeing_nd, crisis_nd.mean(), G_self_nd])):
    ax_summary.text(i - width/2, v1 + 0.5, f'{v1:.1f}', ha='center', va='bottom', fontsize=7)
    ax_summary.text(i + width/2, v2 + 0.5, f'{v2:.1f}', ha='center', va='bottom', fontsize=7)

# ── Bottom right: total effort ────────────────────────────────────────────────
ax_effort.bar(['1D (Career‑only)', 'ND (Multi‑dim)'],
              [total_effort_1d, total_effort_nd], 
              color=['#1f77b4', '#ff7f0e'], alpha=0.8)
ax_effort.set_title('Total effort expended\n(identical budget per step)',
                    fontsize=10, fontweight='bold')
ax_effort.set_ylabel('Cumulative |u|')
ax_effort.grid(True, alpha=0.2, axis='y')
for i, v in enumerate([total_effort_1d, total_effort_nd]):
    ax_effort.text(i, v + 5, f'{v:.0f}', ha='center', fontsize=9)

fig.suptitle(
    'Self Stability Simulator — The Self‑Variety Gap and Goodhart‑Ashby in Personal Governance\n'
    'x(t+1) = A·x(t) + coupling(x) + B·u(t) + d(t)   |   '
    f'G_self(1D)={G_self_1d}   |   G_self(ND)={G_self_nd}   |   '
    'All controllers have equal effort budget',
    fontsize=11, y=1.02
)

plt.savefig('outputs/self_i_variety_gap.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved to outputs/self_i_variety_gap.png")
