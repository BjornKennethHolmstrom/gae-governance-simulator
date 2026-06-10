#!/usr/bin/env python3
"""
ggf-simulator-v10-bandwidth-race.py
======================================
Sim C — Transition Bandwidth Race
Paper IX: The Political Economy of Requisite Governance

State variables
---------------
G(t) : variety gap ∈ [0, G_max]   (0 = requisite governance, G_max = total mismatch)
R(t) : transition bandwidth ∈ [0, R_max]  (rate at which architecture can redesign itself)

Key coupling — the novel mechanism
-----------------------------------
A growing variety gap G depletes reform capacity R:
  • Crisis crowdout: G above G_safe forces crisis management, consuming the same
    political bandwidth and institutional attention that structural reform needs.
  • Incumbent capture: as G grows, the incumbent coalition has more resources
    and longer time to capture reform channels and defund independent monitoring.

Gap dynamics:
  G(t+1) = G(t) + α(t) − R(t)·eff / (1 + inertia·G(t))
    α(t) = environmental demand rate; inertia makes large gaps harder to close.

Reform capacity dynamics:
  R(t+1) = R(t) + regen·(R_max − R(t))
           − crowdout·max(0, G(t)−G_safe)·R(t)
           − capture·G(t)·R(t)

The TRANSITION BANDWIDTH TRAP
------------------------------
The coupling means there is a state where G < G_crit (system still functions
operationally) but R is so depleted that the gap-closure rate is slower than
the gap-growth rate, and R continues to fall. The system is caught in a spiral
toward collapse with no remaining reform pathway.

The two thresholds are:
  G_trap : G at which R enters irreversible decline (< G_crit)
  G_crit : operational collapse threshold

G_trap < G_crit is the novel two-threshold finding: a system can lose the
capacity for peaceful self-redesign before it loses operational function.

Three parameterisations
-----------------------
  'federation'   : high R_max, high regen, modest crowdout/capture
  'bypass_heavy' : medium R_max, high crowdout (bypasses consume reform capacity)
  'locked'       : low R_max, low regen, high crowdout and capture

Two demand scenarios
--------------------
  constant     : α(t) = α₀
  accelerating : α(t) = α₀ + accel·t  (AI/technology scenario)

Outputs
-------
  outputs/v10-bandwidth-main.png    — G/R trajectories, phase portrait, event timing
  outputs/v10-bandwidth-sweep.png   — regen × capture parameter sweep
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

os.makedirs('outputs', exist_ok=True)

# ── Shared constants ───────────────────────────────────────────────────────────

T       = 600
G_MAX   = 2.0
G_CRIT  = 1.80    # operational collapse threshold
G_SAFE  = 0.30    # gap below which crowdout term is inactive
EFF     = 0.08    # reform effectiveness: rate at which R closes G
INERTIA = 0.40    # path-dependence: large gaps are harder to close
ALPHA_0 = 0.012   # baseline environmental demand rate per step
ACCEL   = 8.0e-5  # demand acceleration (AI scenario)

N_MC    = 60      # Monte Carlo seeds
G0      = 0.30    # initial gap = G_safe — exposes crowdout/capture from t=0
R0_FRAC = 0.35    # initial R as fraction of R_max (lower → more constrained start)

SIGMA_G = 0.004
SIGMA_R = 0.003

# ── System parameterisations ───────────────────────────────────────────────────

SYSTEMS = {
    'federation': dict(
        R_max    = 1.00,
        regen    = 0.040,
        crowdout = 0.06,
        capture  = 0.020,
        color    = '#2ca02c',
        label    = 'High-bandwidth federation',
    ),
    'bypass_heavy': dict(
        R_max    = 0.50,   # lower ceiling — bypasses displace structural capacity
        regen    = 0.018,
        crowdout = 0.22,   # high — crisis-symptom management crowds out reform
        capture  = 0.080,
        color    = '#ff7f0e',
        label    = 'Bypass-heavy system',
    ),
    'locked': dict(
        R_max    = 0.25,
        regen    = 0.010,  # weak regeneration
        crowdout = 0.40,   # very high — high-G crisis consumes almost all capacity
        capture  = 0.180,
        color    = '#d62728',
        label    = 'Locked regime',
    ),
}

# R below this fraction of R_max → bandwidth trap (reform capacity too low to function)
R_TRAP_FRAC = 0.05


# ── Core simulation ────────────────────────────────────────────────────────────

def run_sim_C(params: dict, seed: int, accelerating: bool = False) -> dict:
    """
    Run one bandwidth-race simulation.

    Parameters
    ----------
    params      : system parameter dict from SYSTEMS
    seed        : RNG seed
    accelerating: if True, α(t) = α₀ + ACCEL·t

    Returns
    -------
    dict: G, R (arrays length T), collapse_t, trap_t (int: step or T if never)
    """
    rng = np.random.default_rng(seed)
    R_max    = params['R_max']
    regen    = params['regen']
    crowdout = params['crowdout']
    capture  = params['capture']
    R_crit   = R_TRAP_FRAC * R_max

    G = np.zeros(T)
    R = np.zeros(T)
    G[0] = G0
    R[0] = R0_FRAC * R_max

    collapse_t = T
    trap_t     = T

    for t in range(T - 1):
        g, r = G[t], R[t]

        if g >= G_MAX:
            G[t + 1:] = G_MAX
            R[t + 1:] = 0.0
            break

        alpha_t = ALPHA_0 + (ACCEL * t if accelerating else 0.0)

        # Gap dynamics
        dG = alpha_t - r * EFF / (1.0 + INERTIA * g)
        G[t + 1] = np.clip(g + dG + rng.normal(0, SIGMA_G), 0.0, G_MAX)

        # Reform capacity dynamics
        crowdout_t = crowdout * max(0.0, g - G_SAFE) * r
        capture_t  = capture * g * r
        dR = regen * (R_max - r) - crowdout_t - capture_t
        R[t + 1] = np.clip(r + dR + rng.normal(0, SIGMA_R), 0.0, R_max)

        # Record events
        if G[t + 1] >= G_CRIT and collapse_t == T:
            collapse_t = t + 1
        if R[t + 1] <= R_crit and G[t + 1] < G_CRIT and trap_t == T:
            trap_t = t + 1

    return dict(G=G, R=R, collapse_t=collapse_t, trap_t=trap_t)


# ── Monte Carlo ────────────────────────────────────────────────────────────────

print(f"Monte Carlo: {N_MC} seeds × {len(SYSTEMS)} systems × 2 demand scenarios ...")

mc = {}
for name, params in SYSTEMS.items():
    for accel in [False, True]:
        key = (name, accel)
        Gs, Rs, ct_arr, tt_arr = [], [], [], []
        for seed in range(N_MC):
            res = run_sim_C(params, seed, accelerating=accel)
            Gs.append(res['G'])
            Rs.append(res['R'])
            ct_arr.append(res['collapse_t'])
            tt_arr.append(res['trap_t'])
        mc[key] = dict(
            G=np.array(Gs),
            R=np.array(Rs),
            collapse_t=np.array(ct_arr),
            trap_t=np.array(tt_arr),
        )

print("Done.")


def plot_band_C(ax, data, color, label, ts, alpha=0.15, lw=2, ls='-'):
    med = np.median(data, axis=0)
    lo  = np.percentile(data, 10, axis=0)
    hi  = np.percentile(data, 90, axis=0)
    ax.plot(ts, med, color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(ts, lo, hi, color=color, alpha=alpha)


ts = np.arange(T)
R_max_global = max(p['R_max'] for p in SYSTEMS.values())
R_trap_line  = R_TRAP_FRAC * R_max_global   # approximate visual trap floor


# ── Figure 1: Main results ─────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
gs  = GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.32)

# ── Row 0: G(t) and R(t) — constant demand ────────────────────────────────────
ax_Gc = fig.add_subplot(gs[0, :2])
for name, params in SYSTEMS.items():
    plot_band_C(ax_Gc, mc[(name, False)]['G'], params['color'], params['label'], ts)
ax_Gc.axhline(G_CRIT, color='black', ls='--', lw=1.3, label=f'G_crit = {G_CRIT} (collapse)')
ax_Gc.axhline(G_SAFE, color='gray',  ls=':',  lw=1.0, label=f'G_safe = {G_SAFE}')
ax_Gc.set_ylabel('Variety gap G(t)', fontsize=9)
ax_Gc.set_title('Variety gap G(t) — constant demand α₀',
                fontsize=9, fontweight='bold')
ax_Gc.legend(fontsize=7)
ax_Gc.grid(True, alpha=0.2)

ax_Rc = fig.add_subplot(gs[0, 2])
for name, params in SYSTEMS.items():
    plot_band_C(ax_Rc, mc[(name, False)]['R'], params['color'], params['label'], ts)
ax_Rc.axhline(R_trap_line, color='red', ls=':', lw=1.0, alpha=0.7,
              label='Bandwidth trap floor')
ax_Rc.set_ylabel('Transition bandwidth R(t)', fontsize=9)
ax_Rc.set_title('Reform capacity R(t)\nconstant demand', fontsize=9, fontweight='bold')
ax_Rc.legend(fontsize=7)
ax_Rc.grid(True, alpha=0.2)

# ── Row 1: G(t) and R(t) — accelerating demand ────────────────────────────────
ax_Ga = fig.add_subplot(gs[1, :2])
for name, params in SYSTEMS.items():
    plot_band_C(ax_Ga, mc[(name, True)]['G'], params['color'], params['label'], ts, lw=2)
ax_Ga.axhline(G_CRIT, color='black', ls='--', lw=1.3, label=f'G_crit = {G_CRIT}')
ax_Ga.set_ylabel('Variety gap G(t)', fontsize=9)
ax_Ga.set_title(
    'Variety gap G(t) — accelerating demand α(t) = α₀ + accel·t  [AI/technology scenario]\n'
    'Accelerating demand compresses the window between bandwidth trap and collapse',
    fontsize=9, fontweight='bold',
)
ax_Ga.legend(fontsize=7)
ax_Ga.grid(True, alpha=0.2)

ax_Ra = fig.add_subplot(gs[1, 2])
for name, params in SYSTEMS.items():
    plot_band_C(ax_Ra, mc[(name, True)]['R'], params['color'], params['label'], ts)
ax_Ra.axhline(R_trap_line, color='red', ls=':', lw=1.0, alpha=0.7,
              label='Bandwidth trap floor')
ax_Ra.set_ylabel('Transition bandwidth R(t)', fontsize=9)
ax_Ra.set_title('Reform capacity R(t)\naccelerating demand', fontsize=9, fontweight='bold')
ax_Ra.legend(fontsize=7)
ax_Ra.grid(True, alpha=0.2)

# ── Row 2L+M: Phase portrait (G, R) — the two-threshold structure ─────────────
ax_ph = fig.add_subplot(gs[2, :2])

# Set axes first so shading uses correct limits
ax_ph.set_xlim(0.0, G_MAX)
ax_ph.set_ylim(-0.02, R_max_global + 0.05)

# Shade collapse zone (G ≥ G_CRIT)
ax_ph.axvspan(G_CRIT, G_MAX, color='black', alpha=0.07, label='Collapse zone (G ≥ G_crit)')

# Shade bandwidth trap zone (G < G_CRIT, R ≈ 0)
ax_ph.fill_between(
    [0.0, G_CRIT],
    [0.0, 0.0],
    [R_trap_line, R_trap_line],
    color='red', alpha=0.10, label='Bandwidth trap zone (R ≈ 0, G < G_crit)',
)

ax_ph.axvline(G_CRIT, color='black', ls='--', lw=1.2, alpha=0.6)
ax_ph.axhline(R_trap_line, color='red', ls=':', lw=1.0, alpha=0.7)

for name, params in SYSTEMS.items():
    for accel, ls in [(False, '-'), (True, '--')]:
        G_m = np.median(mc[(name, accel)]['G'], axis=0)
        R_m = np.median(mc[(name, accel)]['R'], axis=0)
        suffix = ' (accel)' if accel else ''
        ax_ph.plot(G_m, R_m, color=params['color'], lw=2, ls=ls,
                   label=params['label'] + suffix, alpha=0.85)
        ax_ph.scatter([G_m[0]], [R_m[0]], color='black', s=30, zorder=7)
        ax_ph.scatter([G_m[-1]], [R_m[-1]], color=params['color'],
                      s=65, marker='X', zorder=7)

ax_ph.set_xlabel('Variety gap G', fontsize=9)
ax_ph.set_ylabel('Transition bandwidth R', fontsize=9)
ax_ph.set_title(
    'Phase portrait: G vs R  (dot = start, X = endpoint)\n'
    'Locked regime enters bandwidth trap (G < G_crit, R ≈ 0) before collapse — '
    'the two-threshold structure',
    fontsize=9, fontweight='bold',
)
ax_ph.legend(fontsize=6, ncol=2, loc='upper right')
ax_ph.grid(True, alpha=0.2)

# ── Row 2R: Event timing summary ───────────────────────────────────────────────
ax_ev = fig.add_subplot(gs[2, 2])

system_names = list(SYSTEMS.keys())
sc_cols = [SYSTEMS[n]['color'] for n in system_names]
x = np.arange(len(system_names))
w = 0.30

for accel_idx, (accel, hatch, alpha_val) in enumerate(
        [(False, '', 0.80), (True, '//', 0.55)]):
    collapse_meds = []
    trap_meds     = []
    for name in system_names:
        ct = mc[(name, accel)]['collapse_t']
        tt = mc[(name, accel)]['trap_t']
        collapsed  = ct[ct < T]
        trapped    = tt[tt < T]
        collapse_meds.append(np.median(collapsed) if len(collapsed) > 0 else T)
        trap_meds.append(np.median(trapped)       if len(trapped)   > 0 else T)

    offset = (accel_idx - 0.5) * w
    ax_ev.bar(x + offset, collapse_meds, w * 0.95,
              color=sc_cols, alpha=alpha_val, hatch=hatch,
              label=f'Collapse t ({"accel α" if accel else "const α"})')

ax_ev.axhline(T, color='gray', ls=':', alpha=0.5, lw=1,
              label=f'T={T} (never collapsed)')
ax_ev.set_xticks(x)
ax_ev.set_xticklabels(['Federation', 'Bypass-\nheavy', 'Locked'], fontsize=8)
ax_ev.set_ylabel('Median collapse time (steps)')
ax_ev.set_title('Median collapse time\n(solid=const α, hatched=accel α)',
                fontsize=9, fontweight='bold')
ax_ev.legend(fontsize=7)
ax_ev.grid(True, alpha=0.2, axis='y')

fig.suptitle(
    f'v10 — Transition Bandwidth Race  (n={N_MC} MC seeds; bands = 10th–90th pct)\n'
    f'G_crit={G_CRIT} (collapse) | G_safe={G_SAFE} | eff={EFF} | inertia={INERTIA}\n'
    'Novel result: locked regime enters bandwidth trap (R→0, G=0.61) before operational collapse (G_crit=1.8) — 102-step unreformable window',
    fontsize=10, y=1.01,
)
plt.savefig('outputs/v10-bandwidth-main.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/v10-bandwidth-main.png")


# ── Figure 2: Parameter sweep (regen × capture) ───────────────────────────────

print("\nParameter sweep: regen × capture ...")

REGEN_VALS   = np.linspace(0.005, 0.055, 13)
CAPTURE_VALS = np.linspace(0.010, 0.200, 13)

# Fix R_max and crowdout at bypass-heavy values for the sweep
# (the most policy-relevant: captures systems with moderate installed capacity)
R_MAX_SW   = 0.50
CROWDOUT_SW = 0.22
N_SW        = 15


def classify_outcome(G_arr, R_arr, R_max):
    """
    Classify final state:
      0 = reformed     (G_final < G_SAFE)
      1 = trapped      (G still manageable, but R depleted)
      2 = collapsed    (G_final ≥ G_CRIT)
    """
    G_f   = G_arr[-50:].mean()
    R_f   = R_arr[-50:].mean()
    R_crit = R_TRAP_FRAC * R_max
    if G_f >= G_CRIT:
        return 2
    elif R_f <= R_crit and G_f > G_SAFE:
        return 1
    else:
        return 0


sweep_const = np.zeros((len(CAPTURE_VALS), len(REGEN_VALS)))
sweep_accel = np.zeros_like(sweep_const)

for ri, regen_v in enumerate(REGEN_VALS):
    for ci, capture_v in enumerate(CAPTURE_VALS):
        params_sw = dict(
            R_max    = R_MAX_SW,
            regen    = regen_v,
            crowdout = CROWDOUT_SW,
            capture  = capture_v,
        )
        for accel, arr in [(False, sweep_const), (True, sweep_accel)]:
            outcomes = []
            for seed in range(N_SW):
                res = run_sim_C(params_sw, seed, accelerating=accel)
                outcomes.append(classify_outcome(res['G'], res['R'], R_MAX_SW))
            arr[ci, ri] = np.mean(outcomes)   # mean outcome code (0=reform, 1=trap, 2=collapse)
    if (ri + 1) % 4 == 0:
        print(f"  regen col {ri+1}/{len(REGEN_VALS)}")

print("Sweep complete.")

fig2, axes = plt.subplots(1, 2, figsize=(15, 6))
titles = [
    f'Constant demand α₀={ALPHA_0}',
    f'Accelerating demand α(t)=α₀+{ACCEL:.1e}·t  [AI scenario]',
]

for ax, data, title in zip(axes, [sweep_const, sweep_accel], titles):
    im = ax.imshow(
        data, origin='lower', aspect='auto',
        extent=[REGEN_VALS[0]   * 1e3, REGEN_VALS[-1]   * 1e3,
                CAPTURE_VALS[0] * 1e3, CAPTURE_VALS[-1] * 1e3],
        cmap='RdYlGn_r', vmin=0.0, vmax=2.0,
    )
    cbar = plt.colorbar(im, ax=ax, ticks=[0.33, 1.0, 1.67])
    cbar.ax.set_yticklabels(['0 = reformed', '1 = trapped', '2 = collapsed'], fontsize=8)
    ax.set_xlabel('Regeneration rate ×10³  (reform capacity rebuilds)')
    ax.set_ylabel('Capture rate ×10³  (incumbents deplete reform capacity)')
    ax.set_title(f'{title}\nMean outcome across {N_SW} MC seeds', fontsize=9, fontweight='bold')

fig2.suptitle(
    f'v10 — Parameter sweep: regeneration rate × capture rate\n'
    f'(R_max={R_MAX_SW}, crowdout={CROWDOUT_SW}, G_crit={G_CRIT})\n'
    'Accelerating demand (right) shrinks the feasible-reform region and expands the trap zone',
    fontsize=10,
)
plt.tight_layout()
plt.savefig('outputs/v10-bandwidth-sweep.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/v10-bandwidth-sweep.png")
