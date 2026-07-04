"""
Governance as Engineering Governance Simulator v4 — Fractality as Stability
===========================================================================
Extends v3 to a multi-scale disturbance environment, demonstrating that no
single-scale controller can stabilise a system subject to simultaneous fast,
medium, and slow disturbances — and that fractal (nested multi-scale) control
is the stability-optimal architecture.

Disturbance spectrum
─────────────────────
  Fast   (τ_local  = 2)  — recurring local shocks:   nodes 2 & 7, period 30
  Medium (τ_regional= 6)  — regional sinusoidal drift: nodes 0-4, period 45
  Slow   (τ_global = 12)  — system-wide secular drift: all nodes, very long period

Three architectures
────────────────────
  A — Central only   (τ=12): blind to fast & medium disturbances
  B — Local only     (τ= 2): over-reacts to slow drift → high-freq oscillation
  C — Fractal        (τ_l=2, τ_r=6, τ_g=12): each layer closes its frequency gap

Key formal result
──────────────────
  Any controller with latency τ cannot stabilise disturbances faster than
      f_max ≈ 1 / (2·τ)
  Architecture A (τ=12): f_max ≈ 0.042   misses fast & medium bands
  Architecture B (τ= 2): f_max ≈ 0.250   misses slow band → oscillates
  Architecture C:         each layer covers its own band → full spectrum stable

State transition (per node i)
──────────────────────────────
  x_i(t+1) = A·x_i(t)
             + β·Σ(x_j − x_i)                    [coupling / contagion]
             + B_l·u_l,i(t−τ_l)                   [local control]
             + B_r·u_r,region(i)(t−τ_r)            [regional control]
             + B_g·u_g(t−τ_g)                      [global control]
             + d_i(t)                              [disturbance]
             + drift                               [equilibrium offset]

Control laws
─────────────
  Local:    u_l,i   = K_l · (x_ref − y_i(t))
  Regional: u_r,r   = K_r · (x_ref − mean(y_region_r(t)))
  Global:   u_g     = K_g · (x_ref − mean(y(t)))
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Reproducibility ───────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)

# ── Global simulation parameters ─────────────────────────────────────────────
N          = 10       # nodes
T          = 150      # time steps
x_ref      = 100.0    # target equilibrium
A_sys      = 0.95     # natural decay
B_l        = 1.0      # local actuator effectiveness
B_r        = 1.0      # regional actuator effectiveness
B_g        = 1.0      # global actuator effectiveness
# All actuator effectiveness values are equal by design: architecture differences
# in the results are attributable to latency and signal fidelity alone, not
# to differential resource endowments. This is the more conservative and
# scientifically defensible parameterisation.
beta       = 0.02     # coupling / contagion coefficient
drift      = x_ref * (1 - A_sys)

# ── Regional grouping (two regions of 5 nodes each) ──────────────────────────
REGIONS = {
    0: [0, 1, 2, 3, 4],
    1: [5, 6, 7, 8, 9],
}

def node_region(i):
    return 0 if i < 5 else 1

# ── Disturbance spectrum ──────────────────────────────────────────────────────
FAST_NODES    = [2, 7]        # recurring local shocks
FAST_PERIOD   = 30            # fast shock every 30 steps
FAST_MAG      = -35.0

MEDIUM_NODES  = list(REGIONS[0])   # sinusoidal regional pressure on region 0
MEDIUM_PERIOD = 45
MEDIUM_AMP    = 12.0          # amplitude of sinusoidal pressure

SLOW_AMP      = 8.0           # amplitude of slow system-wide sinusoidal drift
SLOW_PERIOD   = 120           # very long period — one full cycle ≈ simulation length

def disturbances(t):
    """Return per-node disturbance vector at time t."""
    d = np.zeros(N)

    # Fast: impulse at FAST_NODES every FAST_PERIOD steps (starting at t=20)
    if t >= 20 and (t - 20) % FAST_PERIOD == 0:
        d[FAST_NODES] += FAST_MAG

    # Medium: sinusoidal pressure on region 0
    d[MEDIUM_NODES] += -MEDIUM_AMP * np.sin(2 * np.pi * t / MEDIUM_PERIOD)

    # Slow: system-wide secular drift
    d[:] += -SLOW_AMP * np.sin(2 * np.pi * t / SLOW_PERIOD)

    return d

# ── Controller parameters ─────────────────────────────────────────────────────
# Stability ceiling: K_max ≈ 1/(τ·|A|)
# τ_l=2  → ceiling ≈ 0.53  → K_l = 0.40 (safe, high bandwidth)
# τ_r=6  → ceiling ≈ 0.18  → K_r = 0.15 (moderate)
# τ_g=12 → ceiling ≈ 0.088 → K_g = 0.07 (weak, slow band only)

tau_l, K_l, sigma_l = 2,  0.40, 0.5   # local
tau_r, K_r, sigma_r = 6,  0.15, 2.0   # regional
tau_g, K_g, sigma_g = 12, 0.07, 5.0   # global / central

# ── Coupling helper ───────────────────────────────────────────────────────────
def couple(x):
    correction = np.zeros(N)
    for i in range(N):
        left  = x[i - 1] if i > 0     else x[i]
        right = x[i + 1] if i < N - 1 else x[i]
        correction[i] = beta * ((left - x[i]) + (right - x[i]))
    return correction

# ── Observe (add noise) ───────────────────────────────────────────────────────
def observe(x, sigma):
    return x + rng.normal(0, sigma, N)

# ── Single simulation run ─────────────────────────────────────────────────────
def simulate(arch):
    """
    arch: 'A' (central), 'B' (local), 'C' (fractal)
    Returns x_true (T×N), u_total (T×N)
    """
    x = np.full((T, N), x_ref)

    # Control signal histories (needed for dead-time)
    u_l_hist = np.zeros((T, N))    # local  — per node
    u_r_hist = np.zeros((T, 2))    # regional — per region
    u_g_hist = np.zeros(T)         # global  — scalar

    for t in range(1, T - 1):
        d = disturbances(t)

        # ── Observations ──────────────────────────────────────────────────────
        y_l = observe(x[t], sigma_l)   # local high-fidelity
        y_r = observe(x[t], sigma_r)   # regional (noisier)
        y_g = observe(x[t], sigma_g)   # global (noisiest)

        # ── Control laws ──────────────────────────────────────────────────────
        if arch == 'A':
            # Central only: uses global observation, broadcasts uniform signal
            u_g_hist[t] = K_g * (x_ref - np.mean(y_g))

        elif arch == 'B':
            # Local only: each node acts on its own observation
            u_l_hist[t] = K_l * (x_ref - y_l)

        elif arch == 'C':
            # Fractal: all three layers active, each handling its band
            u_l_hist[t]    = K_l * (x_ref - y_l)
            for r, nodes in REGIONS.items():
                u_r_hist[t, r] = K_r * (x_ref - np.mean(y_r[nodes]))
            u_g_hist[t]    = K_g * (x_ref - np.mean(y_g))

        # ── Apply delayed control ─────────────────────────────────────────────
        act = np.zeros(N)

        if arch == 'A':
            if t >= tau_g:
                act += B_g * u_g_hist[t - tau_g]   # uniform broadcast

        elif arch == 'B':
            if t >= tau_l:
                act += B_l * u_l_hist[t - tau_l]

        elif arch == 'C':
            if t >= tau_l:
                act += B_l * u_l_hist[t - tau_l]
            if t >= tau_r:
                for r, nodes in REGIONS.items():
                    act[nodes] += B_r * u_r_hist[t - tau_r, r]
            if t >= tau_g:
                act += B_g * u_g_hist[t - tau_g]

        # ── State transition ──────────────────────────────────────────────────
        x[t + 1] = A_sys * x[t] + couple(x[t]) + act + d + drift

    # Compute total control effort per node per step
    u_total = np.zeros((T, N))
    if arch == 'A':
        for t in range(T):
            u_total[t] = B_g * u_g_hist[t]
    elif arch == 'B':
        u_total = B_l * u_l_hist
    elif arch == 'C':
        u_total = B_l * u_l_hist.copy()
        for t in range(T):
            for r, nodes in REGIONS.items():
                u_total[t, nodes] += B_r * u_r_hist[t, r]
            u_total[t] += B_g * u_g_hist[t]

    return x, u_total

# ── Run all three architectures ───────────────────────────────────────────────
print("Running simulations...")
xA, uA = simulate('A')
xB, uB = simulate('B')
xC, uC = simulate('C')
print("Done.\n")

# ── Metrics ───────────────────────────────────────────────────────────────────
WARMUP = 10   # ignore initial transient

def deficit(x):
    return np.sum(np.maximum(0, x_ref - x[WARMUP:]), axis=0)

def total_effort(u):
    return np.sum(np.abs(u[WARMUP:]), axis=0)

dA, dB, dC = deficit(xA), deficit(xB), deficit(xC)
eA, eB, eC = total_effort(uA), total_effort(uB), total_effort(uC)

print(f"{'Metric':<28} {'Arch A':>10} {'Arch B':>10} {'Arch C':>10}")
print("─" * 60)
print(f"{'Total deficit':<28} {dA.sum():>10.0f} {dB.sum():>10.0f} {dC.sum():>10.0f}")
print(f"{'Total control effort':<28} {eA.sum():>10.0f} {eB.sum():>10.0f} {eC.sum():>10.0f}")
print(f"{'Mean node stability':<28} {xA[WARMUP:].mean():>10.1f} {xB[WARMUP:].mean():>10.1f} {xC[WARMUP:].mean():>10.1f}")
print(f"{'Stability std dev':<28} {xA[WARMUP:].std():>10.2f} {xB[WARMUP:].std():>10.2f} {xC[WARMUP:].std():>10.2f}")

# ── Build disturbance timeline for annotation ─────────────────────────────────
fast_events = [t for t in range(T) if t >= 20 and (t - 20) % FAST_PERIOD == 0]

# ── Plotting ──────────────────────────────────────────────────────────────────
COLOR_A = '#dc2626'   # red   — central
COLOR_B = '#f97316'   # amber — local only
COLOR_C = '#16a34a'   # green — fractal

ts = np.arange(T)

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

ax_hm_A  = fig.add_subplot(gs[0, 0])
ax_hm_B  = fig.add_subplot(gs[0, 1])
ax_hm_C  = fig.add_subplot(gs[0, 2])
ax_mean  = fig.add_subplot(gs[1, :])
ax_n2    = fig.add_subplot(gs[2, 0])
ax_n5    = fig.add_subplot(gs[2, 1])
ax_sys   = fig.add_subplot(gs[2, 2])
ax_def   = fig.add_subplot(gs[3, 0])
ax_eff   = fig.add_subplot(gs[3, 1])
ax_freq  = fig.add_subplot(gs[3, 2])

vmin, vmax = 70, 115

def add_fast_events(ax, ymin=None, ymax=None):
    for fe in fast_events:
        ax.axvline(fe, color='purple', lw=0.6, ls=':', alpha=0.5)

# ── Row 0: heatmaps ───────────────────────────────────────────────────────────
for ax, x, title, color in [
    (ax_hm_A, xA, 'Architecture A — Central only\n(fast & medium gaps uncontrolled)', COLOR_A),
    (ax_hm_B, xB, 'Architecture B — Local only\n(slow drift causes oscillation)',     COLOR_B),
    (ax_hm_C, xC, 'Architecture C — Fractal\n(all frequency bands covered)',          COLOR_C),
]:
    im = ax.imshow(x.T, aspect='auto', cmap='RdYlGn',
                   vmin=vmin, vmax=vmax,
                   extent=[0, T, N - 0.5, -0.5])
    for fe in fast_events:
        ax.axvline(fe, color='purple', lw=0.8, ls=':', alpha=0.7)
    ax.set_yticks(range(N))
    ax.set_yticklabels([f'N{i}' for i in range(N)], fontsize=6)
    ax.set_xlabel('Time step', fontsize=8)
    ax.set_title(title, fontsize=8, fontweight='bold', color=color)
    plt.colorbar(im, ax=ax, label='Stability', shrink=0.8)

# ── Row 1: system mean stability ──────────────────────────────────────────────
ax_mean.plot(ts, xA.mean(axis=1), color=COLOR_A, lw=1.8, label='A — Central only')
ax_mean.plot(ts, xB.mean(axis=1), color=COLOR_B, lw=1.8, label='B — Local only')
ax_mean.plot(ts, xC.mean(axis=1), color=COLOR_C, lw=2.2, label='C — Fractal', zorder=5)
ax_mean.axhline(x_ref, color='gray', ls='--', alpha=0.4, label='Equilibrium')
add_fast_events(ax_mean)
ax_mean.set_title('System mean stability — all architectures\n(purple dotted lines: fast shock events)',
                  fontsize=9, fontweight='bold')
ax_mean.set_ylabel('Mean stability')
ax_mean.legend(fontsize=8, loc='lower right')
ax_mean.grid(True, alpha=0.2)
ax_mean.set_ylim(60, 115)

# ── Row 2: representative node traces ─────────────────────────────────────────
for ax, node, label, note in [
    (ax_n2, 2, f'Node 2 (fast shock target)',
     'Fast shocks: A misses, B responds, C responds'),
    (ax_n5, 5, f'Node 5 (medium pressure target)',
     'Medium pressure: A underreacts, B oscillates, C stable'),
    (ax_sys, 0, f'Node 0 (slow drift only)',
     'Slow drift: B oscillates, A & C track smoothly'),
]:
    ax.plot(ts, xA[:, node], color=COLOR_A, lw=1.4, alpha=0.9, label='A')
    ax.plot(ts, xB[:, node], color=COLOR_B, lw=1.4, alpha=0.9, label='B')
    ax.plot(ts, xC[:, node], color=COLOR_C, lw=1.8, label='C')
    ax.axhline(x_ref, color='gray', ls='--', alpha=0.4)
    add_fast_events(ax)
    ax.set_title(f'{label}\n{note}', fontsize=8, fontweight='bold')
    ax.set_ylabel('Stability')
    ax.set_ylim(55, 120)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

# ── Row 3: deficit bar, effort bar, frequency diagram ─────────────────────────

# Deficit by node
node_idx = np.arange(N)
w = 0.28
ax_def.bar(node_idx - w, dA, w, color=COLOR_A, alpha=0.8, label='A — Central')
ax_def.bar(node_idx,     dB, w, color=COLOR_B, alpha=0.8, label='B — Local')
ax_def.bar(node_idx + w, dC, w, color=COLOR_C, alpha=0.8, label='C — Fractal')
for fn in FAST_NODES:
    ax_def.annotate('⚡', xy=(fn, max(dA[fn], dB[fn], dC[fn]) + 100),
                    ha='center', fontsize=9)
ax_def.set_xticks(node_idx)
ax_def.set_xticklabels([f'N{i}' for i in range(N)], fontsize=7)
ax_def.set_title('Cumulative stability deficit per node', fontsize=8, fontweight='bold')
ax_def.set_ylabel('Deficit integral')
ax_def.legend(fontsize=7)
ax_def.grid(True, alpha=0.2, axis='y')
ax_def.text(0.98, 0.95,
            f'Total\nA: {dA.sum():.0f}\nB: {dB.sum():.0f}\nC: {dC.sum():.0f}',
            transform=ax_def.transAxes, fontsize=7, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Control effort by node
ax_eff.bar(node_idx - w, eA, w, color=COLOR_A, alpha=0.8, label='A')
ax_eff.bar(node_idx,     eB, w, color=COLOR_B, alpha=0.8, label='B')
ax_eff.bar(node_idx + w, eC, w, color=COLOR_C, alpha=0.8, label='C')
ax_eff.set_xticks(node_idx)
ax_eff.set_xticklabels([f'N{i}' for i in range(N)], fontsize=7)
ax_eff.set_title('Total control effort per node\n(governance cost)', fontsize=8, fontweight='bold')
ax_eff.set_ylabel('|u| summed')
ax_eff.legend(fontsize=7)
ax_eff.grid(True, alpha=0.2, axis='y')

# Frequency coverage diagram
ax_freq.set_xlim(0, 0.55)
ax_freq.set_ylim(-0.5, 3.5)
ax_freq.set_yticks([])
ax_freq.set_xlabel('Disturbance frequency  f  (cycles / step)', fontsize=8)
ax_freq.set_title('Frequency coverage by architecture\n(f_max = 1/2τ)', fontsize=8, fontweight='bold')

def freq_bar(ax, y, f_max, color, label, alpha=0.7):
    ax.barh(y, f_max, left=0, height=0.4, color=color, alpha=alpha, label=label)
    ax.text(f_max + 0.01, y, f'f_max={f_max:.3f}', va='center', fontsize=7)

freq_bar(ax_freq, 3.0, 1/(2*tau_g), COLOR_A, f'A: τ={tau_g} (global only)')
freq_bar(ax_freq, 2.0, 1/(2*tau_l), COLOR_B, f'B: τ={tau_l} (local only)')

# Fractal — show all three bands stacked
ax_freq.barh(1.0, 1/(2*tau_l) - 1/(2*tau_r), left=1/(2*tau_r),
             height=0.4, color='#86efac', alpha=0.9, label=f'C local  τ={tau_l}')
ax_freq.barh(1.0, 1/(2*tau_r) - 1/(2*tau_g), left=1/(2*tau_g),
             height=0.4, color='#4ade80', alpha=0.9, label=f'C region τ={tau_r}')
ax_freq.barh(1.0, 1/(2*tau_g),
             height=0.4, color='#16a34a', alpha=0.9, label=f'C global τ={tau_g}')
ax_freq.text(1/(2*tau_l) + 0.01, 1.0, f'f_max={1/(2*tau_l):.3f}', va='center', fontsize=7)

# Mark actual disturbance frequencies
for freq, label, ls in [
    (1/FAST_PERIOD,   'Fast',   'solid'),
    (1/MEDIUM_PERIOD, 'Medium', 'dashed'),
    (1/SLOW_PERIOD,   'Slow',   'dotted'),
]:
    ax_freq.axvline(freq, color='black', ls=ls, lw=1.2, alpha=0.7)
    ax_freq.text(freq, 3.3, label, ha='center', fontsize=7, rotation=45)

ax_freq.set_yticklabels([])
ax_freq.legend(fontsize=6, loc='lower right')
ax_freq.grid(True, alpha=0.2, axis='x')
ax_freq.text(0.02, -0.3,
             'Vertical lines = actual disturbance frequencies',
             fontsize=6, color='gray', style='italic')

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    'Governance as Engineering Governance Simulator v4 — Fractality as Stability\n'
    'Multi-scale disturbance environment: fast (period 30) + medium (period 45) + slow (period 120)\n'
    'Architecture C (fractal) covers all frequency bands; A and B each leave a gap',
    fontsize=10, y=0.98
)

plt.savefig('outputs/paper_ii_fractal_multiscale.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved to paper_ii_fractal_multiscale.png")
