#!/usr/bin/env python3
"""
paper_ix_bypass_trap.py
================================
Appendix B — Bypass-Trap Dynamics
Paper IX: The Political Economy of Requisite Governance

State variables
---------------
D(t) : substrate dysfunction ∈ [0, 1]   (0 = functional, 1 = fully broken)
B(t) : bypass load share  ∈ [0, B_ceil(D)]

Key mechanism
-------------
  • Bypass ceiling is capped by substrate D:  B_ceil = 1 − coupling·D
    This is the UPI/land-courts coupling — the bypass can only handle load
    that the substrate is not actively mis-governing.
  • Visible dysfunction:  D_vis = D·(1−B)
  • Reform pressure ∝ D_vis only — the bypass hides dysfunction from
    decision-makers, reducing the signal that drives structural repair.
  • Permanent bypass: D_vis stays low → reform pressure stays low →
    D stays high → actual performance ceiling stays low.
    This is the stable low-performance attractor.
  • Sunset-coupled bypass: when B crosses sunset_threshold, bypass is
    ramped down and full reform pressure is restored. The substrate must
    face the dysfunction it deferred.

Three scenarios
---------------
  'none'      — no bypass installed; all dysfunction is visible
  'permanent' — bypass grows toward ceiling; never dismantled
  'sunset'    — bypass triggers its own dismantling at the threshold

Outputs
-------
  outputs/paper_ix_bypass_trap_main.png   — MC trajectories + phase portrait
  outputs/paper_ix_bypass_trap_sweep.png  — drift_rate × reform_rate heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

os.makedirs('outputs', exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────

T    = 500   # time steps
N_MC = 75    # Monte Carlo seeds for main figure

# Substrate natural dynamics
DRIFT_RATE   = 0.0015   # institutional decay rate (dysfunction grows without reform)
REFORM_RATE  = 0.022    # reform responsiveness per unit visible dysfunction
REFORM_FLOOR = 0.001    # minimum reform effort even when bypass hides everything

# Bypass dynamics
INVEST_RATE      = 0.08    # bypass growth rate per unit visible dysfunction
B_DECAY          = 0.004   # bypass maintenance cost
CEILING_COUPLING = 0.65    # D reduces bypass ceiling: B_ceil = 1 − 0.65·D
P_BYPASS_MAX     = 0.90    # bypass performance quality at D = 0

# Sunset mechanism
SUNSET_THRESHOLD     = 0.45   # bypass share that triggers sunset review
SUNSET_RAMP          = 0.016  # rate at which B is reduced post-trigger
SUNSET_PRESSURE_MULT = 2.5    # reform pressure multiplier after sunset fires

# Noise
SIGMA_D = 0.004
SIGMA_B = 0.002

# Initial conditions
D0 = 0.80   # start with high dysfunction (substrate is broken)
B0 = 0.00   # no bypass initially installed


# ── Core simulation ────────────────────────────────────────────────────────────

def run_sim(scenario: str, seed: int,
            drift: float = DRIFT_RATE,
            reform: float = REFORM_RATE) -> dict:
    """
    Run one simulation. Returns dict with arrays D, B, P_total, P_perc, D_vis.

    Parameters
    ----------
    scenario : 'none' | 'permanent' | 'sunset'
    seed     : RNG seed
    drift    : substrate decay rate (swept in parameter study)
    reform   : reform responsiveness (swept in parameter study)
    """
    rng = np.random.default_rng(seed)
    D = np.zeros(T)
    B = np.zeros(T)
    D[0], B[0] = D0, B0
    sunset_triggered = False

    for t in range(T - 1):
        d, b = D[t], B[t]

        # Bypass performance ceiling (substrate caps bypass)
        b_ceil = max(0.0, 1.0 - CEILING_COUPLING * d)

        # Visible dysfunction: portion not routed through bypass
        d_vis = d * (1.0 - b)

        if scenario == 'none':
            b_next = 0.0
            r_eff  = reform * d          # all dysfunction visible → full pressure

        elif scenario == 'permanent':
            invest = INVEST_RATE * d_vis * max(0.0, b_ceil - b)
            b_next = np.clip(
                b + invest - B_DECAY * b + rng.normal(0, SIGMA_B),
                0.0, b_ceil,
            )
            r_eff = reform * d_vis + REFORM_FLOOR * d

        elif scenario == 'sunset':
            if not sunset_triggered and b >= SUNSET_THRESHOLD:
                sunset_triggered = True

            if not sunset_triggered:
                invest = INVEST_RATE * d_vis * max(0.0, b_ceil - b)
                b_next = np.clip(
                    b + invest - B_DECAY * b + rng.normal(0, SIGMA_B),
                    0.0, b_ceil,
                )
                r_eff = reform * d_vis + REFORM_FLOOR * d
            else:
                # Sunset fired: ramp bypass down, restore + amplify reform pressure
                b_next = max(0.0, b - SUNSET_RAMP)
                r_eff  = SUNSET_PRESSURE_MULT * reform * d   # full D now visible + boost

        else:
            raise ValueError(f"Unknown scenario: {scenario!r}")

        d_next = d + drift - r_eff * d + rng.normal(0, SIGMA_D)
        D[t + 1] = np.clip(d_next, 0.0, 1.0)
        B[t + 1] = b_next

    # Derived series
    D_vis   = D * (1.0 - B)
    P_bypass = np.clip(P_BYPASS_MAX * (1.0 - CEILING_COUPLING * D), 0.0, 1.0)
    P_total  = (1.0 - B) * (1.0 - D) + B * P_bypass   # actual total performance
    P_perc   = 1.0 - D_vis                              # perceived performance

    return dict(D=D, B=B, P_total=P_total, P_perc=P_perc, D_vis=D_vis)


# ── Monte Carlo ────────────────────────────────────────────────────────────────

SCENARIOS = ['none', 'permanent', 'sunset']
COLORS    = {'none': '#1f77b4', 'permanent': '#d62728', 'sunset': '#2ca02c'}
LABELS    = {
    'none':      'No bypass',
    'permanent': 'Permanent bypass',
    'sunset':    'Sunset-coupled bypass',
}

print(f"Running Monte Carlo: {N_MC} seeds × {len(SCENARIOS)} scenarios ...")

mc = {sc: {k: [] for k in ['D', 'B', 'P_total', 'P_perc', 'D_vis']}
      for sc in SCENARIOS}

for sc in SCENARIOS:
    for seed in range(N_MC):
        res = run_sim(sc, seed)
        for k in res:
            mc[sc][k].append(res[k])
    for k in mc[sc]:
        mc[sc][k] = np.array(mc[sc][k])

print("Monte Carlo complete.")


def plot_band(ax, data, color, label, ts, alpha=0.15):
    med = np.median(data, axis=0)
    lo  = np.percentile(data, 10, axis=0)
    hi  = np.percentile(data, 90, axis=0)
    ax.plot(ts, med, color=color, lw=2, label=label)
    ax.fill_between(ts, lo, hi, color=color, alpha=alpha)


ts = np.arange(T)


# ── Figure 1: Main results ─────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 17))
gs  = GridSpec(4, 3, figure=fig, hspace=0.44, wspace=0.32)

# Row 0: actual substrate dysfunction (full width)
ax_D = fig.add_subplot(gs[0, :])
for sc in SCENARIOS:
    plot_band(ax_D, mc[sc]['D'], COLORS[sc], LABELS[sc], ts)
ax_D.set_ylabel('Substrate dysfunction D(t)', fontsize=9)
ax_D.set_title(
    'Actual substrate dysfunction — permanent bypass sustains D; '
    'sunset-coupled bypass forces reform resolution',
    fontsize=9, fontweight='bold',
)
ax_D.legend(fontsize=8)
ax_D.grid(True, alpha=0.2)

# Row 1: visible dysfunction | actual performance | perceived performance
ax_vis = fig.add_subplot(gs[1, 0])
for sc in SCENARIOS:
    plot_band(ax_vis, mc[sc]['D_vis'], COLORS[sc], LABELS[sc], ts)
ax_vis.set_title('Visible dysfunction\n(what drives reform pressure)',
                 fontsize=9, fontweight='bold')
ax_vis.set_ylabel('D(t)·(1−B(t))')
ax_vis.legend(fontsize=7)
ax_vis.grid(True, alpha=0.2)

ax_P = fig.add_subplot(gs[1, 1])
for sc in SCENARIOS:
    plot_band(ax_P, mc[sc]['P_total'], COLORS[sc], LABELS[sc], ts)
ax_P.set_title('Actual total performance\n(ceiling = 1−D; bypass capped by substrate)',
               fontsize=9, fontweight='bold')
ax_P.set_ylabel('P_total(t)')
ax_P.legend(fontsize=7)
ax_P.grid(True, alpha=0.2)

ax_Pp = fig.add_subplot(gs[1, 2])
for sc in SCENARIOS:
    plot_band(ax_Pp, mc[sc]['P_perc'], COLORS[sc], LABELS[sc], ts)
ax_Pp.set_title('Perceived performance (1−D_vis)\n'
                'Permanent bypass: perceived > actual — the gap is the trap',
                fontsize=9, fontweight='bold')
ax_Pp.set_ylabel('1 − D_vis(t)')
ax_Pp.legend(fontsize=7)
ax_Pp.grid(True, alpha=0.2)

# Row 2: bypass share | phase portrait | summary bars
ax_B = fig.add_subplot(gs[2, 0])
for sc in ['permanent', 'sunset']:
    plot_band(ax_B, mc[sc]['B'], COLORS[sc], LABELS[sc], ts)
ax_B.axhline(SUNSET_THRESHOLD, color='gray', ls='--', lw=1.0, alpha=0.7,
             label=f'Sunset threshold = {SUNSET_THRESHOLD}')
ax_B.set_title('Bypass load share B(t)\n(sunset ramps down after threshold)',
               fontsize=9, fontweight='bold')
ax_B.set_ylabel('B(t)')
ax_B.set_xlabel('Time')
ax_B.legend(fontsize=7)
ax_B.grid(True, alpha=0.2)

ax_ph = fig.add_subplot(gs[2, 1])
for sc in SCENARIOS:
    D_m = np.median(mc[sc]['D'],       axis=0)
    P_m = np.median(mc[sc]['P_total'], axis=0)
    ax_ph.plot(D_m, P_m, color=COLORS[sc], lw=2, label=LABELS[sc])
    ax_ph.scatter([D_m[0]], [P_m[0]], color='black',    s=40, zorder=6)
    ax_ph.scatter([D_m[-1]], [P_m[-1]], color=COLORS[sc], s=70, marker='X', zorder=6)
ax_ph.set_xlabel('Substrate dysfunction D')
ax_ph.set_ylabel('Actual performance P')
ax_ph.set_title('Phase portrait (dot=start, X=endpoint)\n'
                'Permanent bypass: low-performance attractor',
                fontsize=9, fontweight='bold')
ax_ph.legend(fontsize=7)
ax_ph.grid(True, alpha=0.2)

ax_sm = fig.add_subplot(gs[2, 2])
x, w = np.arange(len(SCENARIOS)), 0.35
fd = [np.median(mc[sc]['D'][:, -50:])       for sc in SCENARIOS]
fp = [np.median(mc[sc]['P_total'][:, -50:]) for sc in SCENARIOS]
sc_cols = [COLORS[sc] for sc in SCENARIOS]
ax_sm.bar(x - w/2, fd, w, color=sc_cols, alpha=0.85, label='Final D (dysfunction)')
ax_sm.bar(x + w/2, fp, w, color=sc_cols, alpha=0.40, hatch='//', label='Final P (performance)')
ax_sm.set_xticks(x)
ax_sm.set_xticklabels(['No bypass', 'Permanent', 'Sunset'], fontsize=8)
ax_sm.set_title('Final state — median, last 50 steps', fontsize=9, fontweight='bold')
ax_sm.legend(fontsize=7)
ax_sm.grid(True, alpha=0.2, axis='y')
for i, (d, p) in enumerate(zip(fd, fp)):
    ax_sm.text(i - w/2, d + 0.01, f'{d:.2f}', ha='center', va='bottom', fontsize=7)
    ax_sm.text(i + w/2, p + 0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=7)

# Row 3: deception gap (P_perc − P_total) — full width
ax_gap = fig.add_subplot(gs[3, :])
for sc in SCENARIOS:
    gap_data = mc[sc]['P_perc'] - mc[sc]['P_total']
    plot_band(ax_gap, gap_data, COLORS[sc], LABELS[sc], ts)
ax_gap.axhline(0, color='black', lw=0.8, alpha=0.4)
ax_gap.set_ylabel('P_perceived − P_actual\n(deception gap)', fontsize=9)
ax_gap.set_xlabel('Time')
ax_gap.set_title(
    'Deception gap: perceived minus actual performance — the self-concealing trap\n'
    'No bypass: zero gap throughout  |  Sunset: gap appears then closes when B→0  |  '
    'Permanent: persistent positive gap hides substrate dysfunction from decision-makers',
    fontsize=9, fontweight='bold',
)
ax_gap.legend(fontsize=8)
ax_gap.grid(True, alpha=0.2)

fig.suptitle(
    f'v8 — Bypass-Trap Dynamics  (n={N_MC} MC seeds; bands = 10th–90th pct)\n'
    'D = substrate dysfunction | B = bypass load share\n'
    'Permanent bypass traps system in low-performance attractor; '
    'sunset-coupled bypass escapes it',
    fontsize=10, y=1.01,
)
plt.savefig('outputs/paper_ix_bypass_trap_main.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/paper_ix_bypass_trap_main.png")


# ── Figure 2: Parameter sweep (drift_rate × reform_rate) ──────────────────────

print("\nParameter sweep: drift_rate × reform_rate (this may take ~1 min) ...")

DR_VALS = np.linspace(0.0005, 0.004, 14)   # institutional decay rates
RR_VALS = np.linspace(0.008,  0.035, 14)   # reform responsiveness
N_SW    = 20                                # MC seeds per grid cell

sweep = np.zeros((len(RR_VALS), len(DR_VALS), len(SCENARIOS)))

for ri, rv in enumerate(RR_VALS):
    for di, dv in enumerate(DR_VALS):
        for si, sc in enumerate(SCENARIOS):
            final_Ds = [
                run_sim(sc, seed, drift=dv, reform=rv)['D'][-50:].mean()
                for seed in range(N_SW)
            ]
            sweep[ri, di, si] = np.mean(final_Ds)
    if (ri + 1) % 5 == 0:
        print(f"  row {ri+1}/{len(RR_VALS)}")

fig2, axes = plt.subplots(1, 3, figsize=(17, 5))
sc_titles  = ['No bypass', 'Permanent bypass', 'Sunset-coupled bypass']

for si, (ax, title) in enumerate(zip(axes, sc_titles)):
    im = ax.imshow(
        sweep[:, :, si], origin='lower', aspect='auto',
        extent=[DR_VALS[0] * 1e3, DR_VALS[-1] * 1e3,
                RR_VALS[0] * 1e3, RR_VALS[-1] * 1e3],
        cmap='RdYlGn_r', vmin=0.0, vmax=0.85,
    )
    plt.colorbar(im, ax=ax, label='Final dysfunction D (0=reformed, 1=broken)')
    ax.set_xlabel('Drift rate ×10³  (institutional decay)')
    ax.set_ylabel('Reform rate ×10³  (reform responsiveness)')
    ax.set_title(f'{title}\nFinal substrate dysfunction D', fontsize=9, fontweight='bold')

fig2.suptitle(
    f'v8 — Parameter sweep: drift rate × reform rate  (n={N_SW} MC per cell)\n'
    'Green = reform succeeds; Red = dysfunction persists. '
    'Sunset-coupled bypass extends the feasible-reform region.',
    fontsize=10,
)
plt.tight_layout()
plt.savefig('outputs/paper_ix_bypass_trap_sweep.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/paper_ix_bypass_trap_sweep.png")
