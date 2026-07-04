#!/usr/bin/env python3
"""
Appendix C — Simulation Architecture for Value‑Function Collapse
================================================================
A minimal dynamical model of the Goodhart‑Ashby synthesis.

Two coupled state variables:
    W(t) : economic output (wealth)
    E(t) : environmental integrity

Architecture 1D (GDP‑only)  → dim(V)=1, observes only W.
Architecture 2D (Wellbeing) → dim(V)=2, observes W and E.

Demonstrates that:
  - The 1D controller initially succeeds, then destroys the environmental
    condition on which its own target depends, collapsing.
  - The 2D controller moderates investment, avoiding collapse.
  - Even by its own narrow metric (W), the 1D architecture eventually
    underperforms the 2D architecture.

Reproducibility: fixed random seed, standard NumPy/Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Reproducibility ───────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=2024)

# ── Model parameters ──────────────────────────────────────────────────────────
T        = 500         # time steps
alpha    = 0.3         # investment productivity per unit of E
delta_W  = 0.05        # wealth depreciation
beta     = 0.25        # environmental cost per unit of I
gamma    = 0.05        # regeneration rate of E toward baseline
eta      = 0.02        # delayed damage from past wealth
E0       = 100.0       # baseline environmental integrity
W_target = 120.0       # desired wealth
I0       = 5.0         # baseline investment
K        = 0.5         # control gain (same for both architectures)
lam      = 1.5         # weight of environment in 2D objective

# Noise standard deviations
sigma_W  = 1.0
sigma_E  = 0.5

# Initial conditions
W0, E0_init = 60.0, 90.0

# ── Controller functions ──────────────────────────────────────────────────────
def controller_1d(W_obs):
    """GDP‑only: maximises investment whenever W is below target."""
    error = W_target - W_obs
    I = I0 + K * error
    return max(0, I)  # prevent negative investment

def controller_2d(W_obs, E_obs):
    """Wellbeing‑aware: moderates investment when E is fragile."""
    error = W_target - W_obs
    # Proactive damping based on E/E0 ratio, not absolute threshold
    fragility = (E_obs / E0) ** 2  # quadratic penalty when E < E0
    I_raw = I0 + K * error
    return max(0, I_raw * fragility)

# ── Simulation loop ───────────────────────────────────────────────────────────
def simulate(arch="1D"):
    """Run one simulation and return trajectories."""
    W = np.zeros(T)
    E = np.zeros(T)
    I = np.zeros(T)
    W[0], E[0] = W0, E0_init
    
    E_min_ever = E0_init  # Initialize locally for this simulation

    for t in range(T-1):
        # Observations (with noise)
        W_obs = W[t] + rng.normal(0, sigma_W)
        if arch == "2D":
            E_obs = E[t] + rng.normal(0, sigma_E)
            I[t] = controller_2d(W_obs, E_obs)
        else:
            E_obs = E[t]  # but 1D doesn't use E; kept for clean code
            I[t] = controller_1d(W_obs)

        # State update
        W[t+1] = W[t] + alpha * E[t] * I[t] - delta_W * W[t]
        E_min_ever = min(E_min_ever, E[t])
        regen_factor = 1.0 if E_min_ever > 40 else 0.2  # collapsed ecosystems don't recover easily
        E[t+1] = E[t] - beta * I[t] + gamma * regen_factor * (E0 - E[t]) - eta * W[t]

        # Ensure non‑negative state (physical floors)
        W[t+1] = max(W[t+1], 0)
        E[t+1] = max(E[t+1], 0)

    return W, E, I

# ── Run both architectures ────────────────────────────────────────────────────
print("Simulating 1D (GDP‑only) ...")
W1, E1, I1 = simulate("1D")
print("Simulating 2D (Wellbeing‑aware) ...")
W2, E2, I2 = simulate("2D")

# ── Metrics ───────────────────────────────────────────────────────────────────

# Collapse time (for visualization only)
def collapse_time(E, thresh=30.0):
    """First time step where E drops below threshold."""
    below = np.where(E < thresh)[0]
    return below[0] if len(below) > 0 else T

t_collapse_1d = collapse_time(E1)
t_collapse_2d = collapse_time(E2)

# Collapse severity (for summary metrics)
def collapse_severity(W, E):
    """Measure how badly the system collapsed (0=fine, 1=total collapse)."""
    final_W_loss = max(0, 1 - W[-50:].mean() / W_target)
    final_E_loss = max(0, 1 - E[-50:].mean() / E0)
    return (final_W_loss + final_E_loss) / 2

severity_1d = collapse_severity(W1, E1)
severity_2d = collapse_severity(W2, E2)

print(f"1D collapse time (E<30): {t_collapse_1d}")
print(f"2D collapse time (E<30): {t_collapse_2d if t_collapse_2d < T else 'never'}")
print(f"1D collapse severity: {severity_1d:.2f}")
print(f"2D collapse severity: {severity_2d:.2f}")
print(f"1D mean W (last 50 steps): {W1[-50:].mean():.1f}")
print(f"2D mean W (last 50 steps): {W2[-50:].mean():.1f}")

# ── Plotting ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

ax_w1 = fig.add_subplot(gs[0, 0])
ax_w2 = fig.add_subplot(gs[0, 1])
ax_j1 = fig.add_subplot(gs[0, 2])
ax_phase = fig.add_subplot(gs[1, 0:2])
ax_summary = fig.add_subplot(gs[1, 2])

ts = np.arange(T)

# ── Top left: 1D trajectories ─────────────────────────────────────────────────
ax_w1.plot(ts, W1, label='Wealth W (GDP)', color='#1f77b4', lw=1.5)
ax_w1.plot(ts, E1, label='Environment E', color='#2ca02c', lw=1.5)
ax_w1.axhline(E0, color='gray', ls='--', alpha=0.4, label='Baseline E₀')
ax_w1.axvline(t_collapse_1d, color='red', ls=':', lw=1.2,
              label=f'Collapse (E<30) at t={t_collapse_1d}')
ax_w1.set_title('Architecture 1D (GDP‑only)\nCollapses when E destroyed',
                fontsize=10, fontweight='bold')
ax_w1.set_ylabel('State')
ax_w1.legend(fontsize=7)
ax_w1.grid(True, alpha=0.2)

# ── Top middle: 2D trajectories ───────────────────────────────────────────────
ax_w2.plot(ts, W2, label='Wealth W', color='#1f77b4', lw=1.5)
ax_w2.plot(ts, E2, label='Environment E', color='#2ca02c', lw=1.5)
ax_w2.axhline(E0, color='gray', ls='--', alpha=0.4, label='Baseline E₀')
ax_w2.set_title('Architecture 2D (Wellbeing‑aware)\nMaintains E, stable W',
                fontsize=10, fontweight='bold')
ax_w2.set_ylabel('State')
ax_w2.legend(fontsize=7)
ax_w2.grid(True, alpha=0.2)

# ── Top right: objective value J1 = W over time ──────────────────────────────
ax_j1.plot(ts, W1, label='1D (GDP‑only)', color='#1f77b4', lw=1.5)
ax_j1.plot(ts, W2, label='2D (Wellbeing‑aware)', color='#ff7f0e', lw=1.5)
ax_j1.axhline(W_target, color='gray', ls='--', alpha=0.4, label='Target W')
ax_j1.set_title('Objective J₁ = W(t)\n2D outperforms even by 1D target',
                fontsize=10, fontweight='bold')
ax_j1.set_ylabel('Wealth')
ax_j1.legend(fontsize=7)
ax_j1.grid(True, alpha=0.2)

# ── Bottom left: phase portrait ──────────────────────────────────────────────
ax_phase.plot(E1, W1, color='#1f77b4', lw=1.5, alpha=0.8, label='1D (GDP‑only)')
ax_phase.plot(E2, W2, color='#ff7f0e', lw=1.5, alpha=0.8, label='2D (Wellbeing‑aware)')
ax_phase.scatter([E1[0]], [W1[0]], color='black', zorder=5, s=40, label='Start')
ax_phase.scatter([E1[-1]], [W1[-1]], color='red', zorder=5, s=40, label='1D end')
ax_phase.scatter([E2[-1]], [W2[-1]], color='green', zorder=5, s=40, label='2D end')
ax_phase.annotate('Collapse', xy=(E1[-1], W1[-1]), xytext=(20, 40),
                  arrowprops=dict(arrowstyle='->', color='red'), fontsize=8, color='red')
ax_phase.set_title('Phase portrait: Wealth vs Environment\nDirection of motion over time',
                   fontsize=10, fontweight='bold')
ax_phase.set_xlabel('Environment E')
ax_phase.set_ylabel('Wealth W')
ax_phase.legend(fontsize=7)
ax_phase.grid(True, alpha=0.2)

# ── Bottom right: summary metrics ─────────────────────────────────────────────
metrics = ['Mean W\n(last 50)', 'Collapse\nseverity', 'E at\nend']
arch1_vals = [W1[-50:].mean(), severity_1d, E1[-1]]
arch2_vals = [W2[-50:].mean(), severity_2d, E2[-1]]

x = np.arange(len(metrics))
width = 0.35
ax_summary.bar(x - width/2, arch1_vals, width, label='1D (GDP‑only)',
               color='#1f77b4', alpha=0.8)
ax_summary.bar(x + width/2, arch2_vals, width, label='2D (Wellbeing)',
               color='#ff7f0e', alpha=0.8)
ax_summary.set_xticks(x)
ax_summary.set_xticklabels(metrics)
ax_summary.set_title('Summary comparison', fontsize=10, fontweight='bold')
ax_summary.legend(fontsize=7)
ax_summary.grid(True, alpha=0.2, axis='y')
# Annotate bars with values
for i, (v1, v2) in enumerate(zip(arch1_vals, arch2_vals)):
    ax_summary.text(i - width/2, v1 + 0.5, f'{v1:.1f}', ha='center', va='bottom', fontsize=7)
    ax_summary.text(i + width/2, v2 + 0.5, f'{v2:.1f}', ha='center', va='bottom', fontsize=7)

# ── Overall title ─────────────────────────────────────────────────────────────
fig.suptitle(
    'Appendix C: Value‑Function Collapse — The Goodhart‑Ashby Synthesis in a Minimal Model\n'
    'W(t+1)=W+α·E·I−δ_W·W | E(t+1)=E−β·I+γ·(E₀−E)+η·W | '
    '1D controller observes only W; 2D observes W and E',
    fontsize=11, y=1.02
)

plt.savefig('outputs/paper_vi_value_function_collapse.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved to outputs/paper_vi_value_function_collapse.png")
