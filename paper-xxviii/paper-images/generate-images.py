import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Global style settings for publication quality
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

# Ensure figures directory exists
import os
os.makedirs('figures', exist_ok=True)

# ----------------------------------------------------------------------
# Figure 1: Phase diagram heatmap for rhoP = 0.02
# ----------------------------------------------------------------------
# Load the full phase diagram data
df_phase = pd.read_csv('phase_diagram_5D_deterministic.csv')

# Select rhoP = 0.02
rhoP_select = 0.02
df_sub = df_phase[df_phase['rhoP'] == rhoP_select].copy()

# Need to build a grid of final regime classifications
# Regime mapping
regime_to_int = {
    'open': 0,
    'intermediate': 1,
    'closed': 2,
    'bistable': 3,
    'oscillatory': 4
}

# Get unique theta and s values
theta_vals = np.sort(df_sub['theta'].unique())
s_vals = np.sort(df_sub['s'].unique())

# Build integer grid
grid = np.empty((len(theta_vals), len(s_vals)), dtype=int)
for i, theta in enumerate(theta_vals):
    for j, s in enumerate(s_vals):
        row = df_sub[(df_sub['theta'] == theta) & (df_sub['s'] == s)]
        if not row.empty:
            grid[i, j] = regime_to_int[row.iloc[0]['final_regime']]
        else:
            grid[i, j] = -1  # missing

# Define colormap matching earlier plots
cmap = ListedColormap(['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = plt.Normalize(vmin=-0.5, vmax=4.5)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(grid, extent=[s_vals[0], s_vals[-1], theta_vals[0], theta_vals[-1]],
               origin='lower', aspect='auto', cmap=cmap, norm=norm)
cbar = plt.colorbar(im, ax=ax, ticks=[0,1,2,3,4])
cbar.ax.set_yticklabels(['open', 'intermediate', 'closed', 'bistable', 'oscillatory'])
ax.set_xlabel('s (stakes / uncertainty multiplier)')
ax.set_ylabel('theta (felt-uncertainty tolerance threshold)')
ax.set_title(f'Phase diagram: closure–adaptation regimes (rhoP = {rhoP_select})')
plt.savefig('figures/figure1_phase_diagram_rhoP_002.png')
plt.close()

# ----------------------------------------------------------------------
# Figure 2: Bistable fractions vs rhoP
# ----------------------------------------------------------------------
df_summary = pd.read_csv('phase_diagram_strict_summary.csv')

fig, ax = plt.subplots(figsize=(6, 4))
x = df_summary['rhoP']
ax.plot(x, df_summary['weak_bistable_frac'], 'o-', label='Weak bistable (any difference)')
ax.plot(x, df_summary['strong_bistable_frac'], 's--', label='Strong bistable (open vs closed)')
ax.set_xlabel('rhoP (permeability adaptation rate)')
ax.set_ylabel('Fraction of parameter space')
ax.set_ylim(0, 0.7)
ax.legend()
ax.set_title('Bistability across permeability adaptation rates')
plt.savefig('figures/figure2_bistable_fractions.png')
plt.close()

# ----------------------------------------------------------------------
# Figure 3: Hysteresis loop for rhoP = 0.02 (or 0.01/0.05 if available)
# ----------------------------------------------------------------------
# We'll attempt to load boundary_quality_hysteresis_results.csv; if missing,
# we can regenerate from a quick simulation, but assume file exists.
try:
    df_hyst = pd.read_csv('boundary_quality_hysteresis_results.csv')
    # Filter perm=0.0 (or closest) for a clear loop? Actually we used perm=0.0 in original.
    # For clarity, use perm=0.0 if present; otherwise first perm value.
    perms = df_hyst['perm'].unique()
    perm_select = 0.0 if 0.0 in perms else perms[0]
    df_loop = df_hyst[df_hyst['perm'] == perm_select]
    s_vals = df_loop['s'].values
    B_up = df_loop['B_up'].values
    B_down = df_loop['B_down'].values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s_vals, B_up, 'o-', color='#1f77b4', label='Up sweep (open start)')
    ax.plot(s_vals, B_down, 's--', color='#d62728', label='Down sweep (closed start)')
    ax.set_xlabel('s (stakes / uncertainty multiplier)')
    ax.set_ylabel('Boundary strength B')
    ax.set_title(f'Hysteresis: closure does not reverse symmetrically (perm = {perm_select})')
    ax.legend()
    plt.savefig('figures/figure3_hysteresis_loop.png')
    plt.close()
except FileNotFoundError:
    print('boundary_quality_hysteresis_results.csv not found; skipping Figure 3.')
except Exception as e:
    print(f'Error generating Figure 3: {e}')

# ----------------------------------------------------------------------
# Figure 4: Stochastic closure probability heatmap
# ----------------------------------------------------------------------
try:
    df_stoch = pd.read_csv('stochastic_sweep_5D.csv')
    # Build grid of prob_closed_open_start
    sigma_vals = np.sort(df_stoch['sigma'].unique())
    s_vals_stoch = np.sort(df_stoch['s'].unique())
    grid = np.empty((len(sigma_vals), len(s_vals_stoch)))
    for i, sigma in enumerate(sigma_vals):
        for j, s in enumerate(s_vals_stoch):
            row = df_stoch[(df_stoch['sigma'] == sigma) & (df_stoch['s'] == s)]
            if not row.empty:
                grid[i, j] = row.iloc[0]['prob_closed_open_start']
            else:
                grid[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, extent=[s_vals_stoch[0], s_vals_stoch[-1],
                                  sigma_vals[0], sigma_vals[-1]],
                   origin='lower', aspect='auto', cmap='viridis',
                   vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability of closure (open start)')
    ax.set_xlabel('s (stakes)')
    ax.set_ylabel('sigma (noise intensity)')
    ax.set_title('Stochastic closure probability (theta=0.196, rhoP=0.02)')
    plt.savefig('figures/figure4_stochastic_heatmap.png')
    plt.close()
except FileNotFoundError:
    print('stochastic_sweep_5D.csv not found; skipping Figure 4.')

# ----------------------------------------------------------------------
# Figure 5: Two-population polarization time series
# ----------------------------------------------------------------------
try:
    df_pol = pd.read_csv('two_pop_polarization_baseline.csv')
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(df_pol['t'], df_pol['B1'], label='B1 (open start)', color='#1f77b4')
    axes[0].plot(df_pol['t'], df_pol['B2'], label='B2 (closed start)', color='#d62728')
    axes[0].set_ylabel('Boundary strength')
    axes[0].legend()

    axes[1].plot(df_pol['t'], df_pol['T1'], label='T1', color='#1f77b4', linestyle='--')
    axes[1].plot(df_pol['t'], df_pol['T2'], label='T2', color='#d62728', linestyle='--')
    axes[1].set_ylabel('Trust')
    axes[1].legend()

    axes[2].plot(df_pol['t'], df_pol['U'], label='Shared uncertainty U', color='black')
    axes[2].set_ylabel('U')
    axes[2].set_xlabel('Time')
    axes[2].legend()

    fig.suptitle('Polarization: two populations, identical parameters, different initial states')
    plt.savefig('figures/figure5_polarization_time_series.png')
    plt.close()
except FileNotFoundError:
    print('two_pop_polarization_baseline.csv not found; skipping Figure 5.')

# ----------------------------------------------------------------------
# Figure 6: Two-population cascade collapse time series
# ----------------------------------------------------------------------
try:
    df_casc = pd.read_csv('two_pop_cascade_collapse.csv')
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(df_casc['t'], df_casc['B1'], label='B1 (shocked)', color='#1f77b4')
    axes[0].plot(df_casc['t'], df_casc['B2'], label='B2 (unshocked)', color='#d62728')
    axes[0].set_ylabel('Boundary strength')
    axes[0].legend()

    axes[1].plot(df_casc['t'], df_casc['T1'], label='T1', color='#1f77b4', linestyle='--')
    axes[1].plot(df_casc['t'], df_casc['T2'], label='T2', color='#d62728', linestyle='--')
    axes[1].set_ylabel('Trust')
    axes[1].legend()

    axes[2].plot(df_casc['t'], df_casc['U'], label='Shared uncertainty U', color='black')
    axes[2].set_ylabel('U')
    axes[2].set_xlabel('Time')
    axes[2].legend()

    fig.suptitle('Cascade collapse: shock to population 1, both collapse')
    plt.savefig('figures/figure6_cascade_collapse_time_series.png')
    plt.close()
except FileNotFoundError:
    print('two_pop_cascade_collapse.csv not found; skipping Figure 6.')

# ----------------------------------------------------------------------
# Figure 7: Intervention effect of P_min
# ----------------------------------------------------------------------
try:
    df_int = pd.read_csv('P_min_intervention_results_corrected.csv')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df_int['P_min'], df_int['final_B'], 'o-', color='#1f77b4')
    ax.axhline(y=0.55, color='red', linestyle='--', label='Closure threshold (B=0.55)')
    ax.axhline(y=0.20, color='green', linestyle='--', label='Open threshold (B=0.20)')
    ax.set_xlabel('Minimum permeability floor P_min')
    ax.set_ylabel('Final boundary strength B')
    ax.set_title('Effect of constitutional permeability floor on crisis recovery')
    ax.legend()
    plt.savefig('figures/figure7_Pmin_intervention.png')
    plt.close()
except FileNotFoundError:
    print('P_min_intervention_results_corrected.csv not found; skipping Figure 7.')

print('All available figures generated in the figures/ directory.')
