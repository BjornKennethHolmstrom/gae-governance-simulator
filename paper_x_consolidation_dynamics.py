#!/usr/bin/env python3
"""
paper_x_consolidation_dynamics.py
=============================================
Sim D2 — Consolidation Dynamics and the Monoculture Attractor
Paper X: Requisite Observer Diversity — Part III §3.3 / Part VI extension

WHAT THIS ADDS BEYOND v11
--------------------------
v11 compares FIXED observer populations (diverse / mixed / monoculture) and
demonstrates the rank-deficiency failure mode. It cannot test Part III's
collapse dynamics: the flow n(t) toward consolidation, the liability ratchet,
or the claim that the monoculture is an absorbing state.

v12 implements the switching dynamics:
  • Organizations periodically re-evaluate their observation strategy.
  • Perceived accuracy is measured as deviation from the ENSEMBLE CONSENSUS
    (organizations cannot observe the true state). This produces an emergent
    positive feedback: shared-system adopters cluster tightly around the
    consensus they collectively constitute, so consolidation makes the shared
    system LOOK progressively more accurate — the epistemic form of the
    liability shield.
  • Liability penalty L(f) = L0 + L1·f raises the effective cost of
    independence as the shared fraction f grows (the one-way ratchet).
  • Switching S→I is near zero (P_BACK): atrophied infrastructure.
  • A protected fraction of organizations never switches (constitutional
    protection, §5.1).

TIMELINE
--------
t ∈ [0, 250):  normal conditions. The shared system genuinely performs better
               on observable dimensions. Consolidation proceeds — driven by
               real short-term advantage, not error.
t = 250:       regime shift. Dimension 5 (observed by NO shared adopter)
               begins persistent drift toward failure.
t ∈ (250, 400]: the question is whether enough independent observers survived
               consolidation to detect the drift and trigger the gate.

This is the paper's narrative made dynamic: diversity is destroyed during
normal times precisely because consolidation is then locally rational, and
the cost appears only when the blind spot becomes load-bearing.

FAILURE: X₅ < −8 at any time within T = 400.

OUTPUTS
-------
  outputs/paper_x_consolidation_dynamics_main.png    n(t) flow, X₅, failure summary
  outputs/paper_x_consolidation_dynamics_sweep.png   2D sweep: protected fraction × L₁
                                        (the Figure D4 the paper describes)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

os.makedirs('outputs', exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────

T          = 400
T_SHIFT    = 250          # regime shift — 25 evaluation cycles of normal-conditions
                          # runway so consolidation dynamics can differentiate
N_OBS      = 20
DIMS       = 5
DIM_BLIND  = 4            # dimension 5 = index 4

# Environment dynamics (as v11)
A_DIAG     = 0.95
Q_SCALE    = 0.1
MU_DRIFT   = 0.10
COUPLING   = 0.08
X5_FAIL    = -8.0

# Observers (as v11)
SIGMA_IND  = np.sqrt(0.5)
SIGMA_SHR  = np.sqrt(0.2)
C_SHR      = np.zeros((4, DIMS))
C_SHR[:4, :4] = np.eye(4)

# Detection & control (as v11)
DETECT_THRESH = -1.5
CONTROL_EFF   = 0.70
CONTROL_RAMP  = 0.25
DIM5_IMPROVE  = 0.8
SIGMA_FLOOR   = np.sqrt(0.1)

# ── Switching dynamics (new in v12) ───────────────────────────────────────────
TAU_SWITCH = 10           # evaluation period (steps)
EVAL_WIN   = 10           # accuracy evaluation window (steps)
C_IND      = 1.0          # cost of independent infrastructure
C_SHARED   = 0.5          # cost of shared infrastructure
L0         = 0.2          # baseline liability cost of independence
# L1 is swept; default below
L1_DEFAULT = 0.8

W_ACC      = 1.0          # weight of perceived accuracy differential in advantage
SWITCH_K   = 2.35         # logistic steepness
SWITCH_MID = 2.36         # midpoint placed BETWEEN the weak-ratchet advantage
                          # ceiling (~1.5) and the strong-ratchet late-stage
                          # advantage (~2.9): weak ratchets saturate at slow
                          # switching; strong ratchets ignite the feedback loop
P_MAX      = 0.45         # max switching probability per evaluation
P_BACK     = 0.002        # probability of S→I per evaluation per org
                          # (atrophied infrastructure: rebuilding an independent
                          # observation system is rare; 20 orgs × 15 evals × 0.002
                          # ≈ 0.6 expected reversions per run — the monoculture is
                          # a near-absorbing state, not an absolute one)

N_MC       = 50


# ── Observer initialisation ────────────────────────────────────────────────────

def make_independent(rng):
    dims = rng.choice(DIMS, size=3, replace=False)
    C = np.zeros((3, DIMS))
    for i, d in enumerate(dims):
        C[i, d] = 0.70
        others = [j for j in range(DIMS) if j != d]
        n_other = rng.integers(1, 3)
        chosen = rng.choice(others, size=n_other, replace=False)
        for j in chosen:
            C[i, j] += 0.30 / n_other
    return {'strategy': 'independent', 'C': C,
            'sigma': np.full(3, SIGMA_IND), 'dims': dims}


def make_shared():
    return {'strategy': 'shared', 'C': C_SHR.copy(),
            'sigma': np.full(4, SIGMA_SHR), 'dims': np.arange(4)}


# ── Core simulation ────────────────────────────────────────────────────────────

def run_sim_D2(protected_frac: float, L1: float, seed: int,
               switching: bool = True) -> dict:
    """
    Run one consolidation-dynamics simulation.

    Parameters
    ----------
    protected_frac : fraction of organizations with constitutional protection
                     (never switch; start independent)
    L1             : liability penalty slope — L(f) = L0 + L1·f
    seed           : RNG seed
    switching      : if False, populations are frozen (v11-style control run)

    Returns
    -------
    dict with X (T×DIMS), n_shared (T,), collapsed (bool),
    dim5_coverage (T,) = number of independent observers seeing dim5,
    gate_active (T,), first_gate (int or T)
    """
    rng = np.random.default_rng(seed)

    n_protected = int(round(protected_frac * N_OBS))
    observers = [make_independent(rng) for _ in range(N_OBS)]
    protected = np.zeros(N_OBS, dtype=bool)
    protected[:n_protected] = True     # first n_protected are constitutionally protected

    shared_noise = rng.normal(0, SIGMA_SHR, (T, 4))

    X = np.zeros((T, DIMS))
    n_shared_hist  = np.zeros(T, dtype=int)
    coverage_hist  = np.zeros(T, dtype=int)
    gate_active    = np.zeros(T, dtype=bool)
    control_eff    = 0.0
    first_gate     = T

    # rolling deviation buffers (per org): squared deviation from consensus
    dev_buffer = [[] for _ in range(N_OBS)]

    for t in range(T):
        # ── 1. Observe ──────────────────────────────────────────────────────
        all_by_dim = {d: [] for d in range(DIMS)}
        ind_dim5_vals = []
        org_obs = []   # (dims, values) per org for deviation computation

        for i, obs in enumerate(observers):
            if obs['strategy'] == 'independent':
                y = obs['C'] @ X[t] + rng.normal(0, obs['sigma'])
            else:
                y = obs['C'] @ X[t] + shared_noise[t]
            org_obs.append((obs['dims'], y))
            for k, d in enumerate(obs['dims']):
                all_by_dim[d].append(y[k])
                if obs['strategy'] == 'independent' and d == DIM_BLIND:
                    ind_dim5_vals.append(y[k])

        consensus = np.full(DIMS, np.nan)
        for d in range(DIMS):
            if all_by_dim[d]:
                consensus[d] = np.mean(all_by_dim[d])

        # record per-org deviation from consensus on dims 0-3 (visible dims)
        for i, (dims_i, y_i) in enumerate(org_obs):
            devs = [(y_i[k] - consensus[d])**2
                    for k, d in enumerate(dims_i)
                    if d < 4 and not np.isnan(consensus[d])]
            if devs:
                dev_buffer[i].append(np.mean(devs))
                if len(dev_buffer[i]) > EVAL_WIN:
                    dev_buffer[i].pop(0)

        coverage_hist[t] = len(ind_dim5_vals)
        n_shared_hist[t] = sum(1 for o in observers if o['strategy'] == 'shared')

        # ── 2. Precautionary gate (v11 mechanism: independent dim5 mean) ───
        x5_est = np.mean(ind_dim5_vals) if ind_dim5_vals else np.nan
        if (not np.isnan(x5_est)) and (x5_est < DETECT_THRESH):
            gate_active[t] = True
            if first_gate == T:
                first_gate = t
            control_eff = min(CONTROL_EFF + CONTROL_RAMP, control_eff + 0.12)
            for obs in observers:
                if obs['strategy'] == 'independent' and DIM_BLIND in obs['dims']:
                    idx = list(obs['dims']).index(DIM_BLIND)
                    obs['sigma'][idx] = max(SIGMA_FLOOR,
                                            obs['sigma'][idx] * DIM5_IMPROVE)
        else:
            control_eff = max(0.0, control_eff - 0.01)

        # ── 3. Strategy re-evaluation every TAU_SWITCH steps ────────────────
        if switching and t > 0 and t % TAU_SWITCH == 0:
            f = n_shared_hist[t] / N_OBS
            L_f = L0 + L1 * f

            # perceived accuracy of each strategy = mean deviation-from-consensus
            shr_devs = [np.mean(dev_buffer[i]) for i, o in enumerate(observers)
                        if o['strategy'] == 'shared' and dev_buffer[i]]
            est_S_dev = np.mean(shr_devs) if shr_devs else SIGMA_SHR**2  # advertised

            for i, obs in enumerate(observers):
                if protected[i]:
                    continue
                if obs['strategy'] == 'independent':
                    own_dev = (np.mean(dev_buffer[i])
                               if dev_buffer[i] else SIGMA_IND**2)
                    advantage = (W_ACC * (own_dev - est_S_dev)
                                 + (C_IND + L_f - C_SHARED))
                    p = P_MAX / (1.0 + np.exp(-SWITCH_K * (advantage - SWITCH_MID)))
                    if rng.random() < p:
                        observers[i] = make_shared()
                        dev_buffer[i] = []
                else:
                    if rng.random() < P_BACK:
                        observers[i] = make_independent(rng)
                        dev_buffer[i] = []

        # ── 4. Environment t → t+1 ─────────────────────────────────────────
        if t < T - 1:
            w = rng.normal(0, Q_SCALE, DIMS)
            X[t+1, :4] = A_DIAG * X[t, :4] + w[:4]
            if t < T_SHIFT:
                X[t+1, DIM_BLIND] = A_DIAG * X[t, DIM_BLIND] + w[DIM_BLIND]
            else:
                X[t+1, DIM_BLIND] = (X[t, DIM_BLIND]
                                      - MU_DRIFT * (1.0 - control_eff)
                                      + w[DIM_BLIND])
                X[t+1, 0] += COUPLING * X[t, DIM_BLIND]

    return {
        'X': X,
        'n_shared': n_shared_hist,
        'coverage': coverage_hist,
        'gate_active': gate_active,
        'first_gate': first_gate,
        'collapsed': bool(np.any(X[:, DIM_BLIND] < X5_FAIL)),
    }


# ── Scenarios for the main figure ──────────────────────────────────────────────

SCENARIOS = [
    dict(prot=0.00, L1=0.2,  color='#1f77b4',
         label='Unprotected, weak ratchet (L₁=0.2)'),
    dict(prot=0.00, L1=1.5,  color='#d62728',
         label='Unprotected, strong ratchet (L₁=1.5)'),
    dict(prot=0.15, L1=1.5,  color='#ff7f0e',
         label='15% protected, strong ratchet'),
    dict(prot=0.30, L1=1.5,  color='#2ca02c',
         label='30% protected, strong ratchet'),
]

print(f"Main figure: {len(SCENARIOS)} scenarios × {N_MC} seeds ...")
mc = []
for sc in SCENARIOS:
    runs = [run_sim_D2(sc['prot'], sc['L1'], seed) for seed in range(N_MC)]
    mc.append({
        'n_shared': np.array([r['n_shared'] for r in runs]),
        'coverage': np.array([r['coverage'] for r in runs]),
        'X5':       np.array([r['X'][:, DIM_BLIND] for r in runs]),
        'collapsed': np.array([r['collapsed'] for r in runs]),
        'first_gate': np.array([r['first_gate'] for r in runs]),
    })
    print(f"  {sc['label']:45s} fail={np.mean(mc[-1]['collapsed']):.2f}")
print("Done.")


def plot_band(ax, data, color, label, ts, alpha=0.15, lw=2):
    med = np.median(data, axis=0)
    lo  = np.percentile(data, 10, axis=0)
    hi  = np.percentile(data, 90, axis=0)
    ax.plot(ts, med, color=color, lw=lw, label=label)
    ax.fill_between(ts, lo, hi, color=color, alpha=alpha)


ts = np.arange(T)

# ── Figure 1: Main results ─────────────────────────────────────────────────────

fig = plt.figure(figsize=(17, 12))
gs  = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.25)

# Panel A: consolidation flow n_shared(t)/N
ax_n = fig.add_subplot(gs[0, :])
for sc, m in zip(SCENARIOS, mc):
    plot_band(ax_n, m['n_shared'] / N_OBS, sc['color'], sc['label'], ts)
ax_n.axvline(T_SHIFT, color='gray', ls=':', lw=1.5, label=f'Regime shift (t={T_SHIFT})')
ax_n.set_ylabel('Shared-system fraction  n(t)/N')
ax_n.set_title(
    'Consolidation dynamics — the flow toward the monoculture attractor\n'
    'Consolidation proceeds during NORMAL conditions: the shared system genuinely '
    'performs better on observable metrics,\nand consensus-relative evaluation makes '
    'it look better still as adoption grows (emergent positive feedback)',
    fontsize=9, fontweight='bold',
)
ax_n.legend(fontsize=7, loc='lower right')
ax_n.grid(True, alpha=0.2)
ax_n.set_ylim(-0.02, 1.05)

# Panel B: dim-5 coverage
ax_c = fig.add_subplot(gs[1, 0])
for sc, m in zip(SCENARIOS, mc):
    plot_band(ax_c, m['coverage'], sc['color'], sc['label'], ts)
ax_c.axvline(T_SHIFT, color='gray', ls=':', lw=1.5)
ax_c.set_ylabel('Independent observers seeing dim 5')
ax_c.set_xlabel('Time step')
ax_c.set_title('Dimension-5 coverage\n(detection capacity surviving consolidation)',
               fontsize=9, fontweight='bold')
ax_c.legend(fontsize=6)
ax_c.grid(True, alpha=0.2)

# Panel C: X5 trajectories
ax_x = fig.add_subplot(gs[1, 1])
for sc, m in zip(SCENARIOS, mc):
    plot_band(ax_x, m['X5'], sc['color'], sc['label'], ts)
ax_x.axhline(X5_FAIL, color='red', ls='--', lw=1.2, label=f'Failure ({X5_FAIL})')
ax_x.axvline(T_SHIFT, color='gray', ls=':', lw=1.5)
ax_x.set_ylabel('X₅ (structural fragility)')
ax_x.set_xlabel('Time step')
ax_x.set_title('Hidden-dimension trajectory\n(does anyone see it drift?)',
               fontsize=9, fontweight='bold')
ax_x.legend(fontsize=6)
ax_x.grid(True, alpha=0.2)

# Panel D: failure probability bars
ax_f = fig.add_subplot(gs[2, 0])
fails = [np.mean(m['collapsed']) for m in mc]
cols  = [sc['color'] for sc in SCENARIOS]
bars = ax_f.bar(range(len(SCENARIOS)), fails, color=cols, alpha=0.85)
ax_f.set_xticks(range(len(SCENARIOS)))
ax_f.set_xticklabels(['0% prot\nL₁=0.2', '0% prot\nL₁=1.5',
                      '15% prot\nL₁=1.5', '30% prot\nL₁=1.5'], fontsize=8)
ax_f.set_ylabel('Failure probability')
ax_f.set_ylim(0, 1.05)
ax_f.set_title(f'Failure probability  (n={N_MC} seeds)', fontsize=9, fontweight='bold')
for b, v in zip(bars, fails):
    ax_f.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.2f}',
              ha='center', fontsize=8)
ax_f.grid(True, alpha=0.2, axis='y')

# Panel E: coverage at shift vs outcome (scatter, all scenarios pooled)
ax_s = fig.add_subplot(gs[2, 1])
for sc, m in zip(SCENARIOS, mc):
    cov_at_shift = m['coverage'][:, T_SHIFT]
    jitter = np.random.default_rng(0).normal(0, 0.08, len(cov_at_shift))
    ax_s.scatter(cov_at_shift + jitter, m['collapsed'] + jitter * 0.5,
                 color=sc['color'], alpha=0.45, s=18)
ax_s.set_xlabel(f'Independent dim-5 coverage at regime shift (t={T_SHIFT})')
ax_s.set_ylabel('Collapsed (1) / survived (0)')
ax_s.set_yticks([0, 1])
ax_s.set_title('Coverage at shift determines outcome\n'
               '(failure occurs iff detection capacity was consolidated away)',
               fontsize=9, fontweight='bold')
ax_s.grid(True, alpha=0.2)

fig.suptitle(
    f'v12 — Consolidation Dynamics (Sim D2)  (n={N_MC} MC seeds; bands = 10th–90th pct)\n'
    f'Switching every {TAU_SWITCH} steps | liability L(f)=L₀+L₁·f, L₀={L0} | '
    f'P(S→I)={P_BACK} (atrophied infrastructure) | regime shift at t={T_SHIFT}',
    fontsize=10, y=1.00,
)
plt.savefig('outputs/paper_x_consolidation_dynamics_main.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/paper_x_consolidation_dynamics_main.png")


# ── Figure 2: 2D sweep — protected fraction × L1 (the paper's Figure D4) ──────

print("\n2D sweep: protected fraction × L1 ...")

PROT_VALS = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
L1_VALS   = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
N_SW      = 20

sweep = np.zeros((len(L1_VALS), len(PROT_VALS)))
for li, l1 in enumerate(L1_VALS):
    for pi, pf in enumerate(PROT_VALS):
        fails = [run_sim_D2(pf, l1, seed=50000 + s)['collapsed']
                 for s in range(N_SW)]
        sweep[li, pi] = np.mean(fails)
    print(f"  L1={l1:.2f} row done")

fig2, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(
    sweep, origin='lower', aspect='auto',
    extent=[PROT_VALS[0] - 0.025, PROT_VALS[-1] + 0.025,
            L1_VALS[0] - 0.125, L1_VALS[-1] + 0.125],
    cmap='RdYlGn_r', vmin=0, vmax=1,
)
plt.colorbar(im, ax=ax, label='Failure probability')
# annotate cells
for li, l1 in enumerate(L1_VALS):
    for pi, pf in enumerate(PROT_VALS):
        ax.text(pf, l1, f'{sweep[li, pi]:.2f}', ha='center', va='center',
                fontsize=7,
                color='white' if sweep[li, pi] > 0.55 else 'black')
ax.set_xlabel('Protected independent observer fraction')
ax.set_ylabel('Liability penalty slope L₁')
ax.set_title(
    f'v12 — Failure probability: protected fraction × liability ratchet strength\n'
    f'(n={N_SW} seeds/cell, T={T}, shift at t={T_SHIFT})\n'
    'High L₁ accelerates consolidation; protection ≥ ~15% preserves detection '
    'regardless of ratchet strength',
    fontsize=10, fontweight='bold',
)
plt.tight_layout()
plt.savefig('outputs/paper_x_consolidation_dynamics_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/paper_x_consolidation_dynamics_sweep.png")
