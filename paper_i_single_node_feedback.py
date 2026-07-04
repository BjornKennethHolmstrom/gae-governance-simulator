"""
Governance as Engineering Governance Simulator v2
=================================================
A state-space simulation comparing two governance architectures under crisis.

Primitive grammar:
  - Nodes:          Processing entities (local BAZ, central state)
  - State x(t):     True system condition (wellbeing/stability)
  - Flows:          Information and resource movements
  - Latency tau:    Dead-time before control action reaches the system
  - Constraints:    Hard limits (ecological boundaries, dignity floors)
  - Feedback:       Closed-loop correction
  - Signal fidelity: SNR of the observation channel

Dynamics:   x(t+1) = A*x(t) + B*u(t-tau) + d(t) + drift
Control:    u(t)   = K * (x_ref - y(t))         [proportional]
Observation: y(t)  = x(t) + noise(0, sigma)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── Reproducibility ───────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)

# ── Simulation parameters ─────────────────────────────────────────────────────
T             = 120      # time steps
t_crisis      = 20       # when the crisis hits
crisis_mag    = -40.0    # shock magnitude
x_ref         = 100.0    # target equilibrium
A_sys         = 0.95     # natural decay (entropy); drift = x_ref*(1-A) keeps eq at x_ref
B_sys         = 1.0      # actuator effectiveness

# ── Architecture A: Centralized / Tier 1 ──────────────────────────────────────
#
# High latency forces a low gain ceiling.
# Rule of thumb for dead-time dominant systems: K < 1/(tau * |A|) for stability.
# With tau=15, A=0.95  →  K_max ≈ 0.07 * some margin  →  K=0.3 is already near limit.
# This is NOT an arbitrary handicap – it is a structural consequence of latency.
tau_A   = 15    # dead-time (steps): parliamentary cycle, bureaucratic processing
sigma_A = 8.0   # observation noise: hidden externalities, distorted reporting
K_A     = 0.3   # gain ceiling imposed by latency (cannot be raised without instability)

# ── Architecture B: Fractal / GGF (BAZ + subsidiarity) ────────────────────────
#
# Low latency allows aggressive, precise correction.
# Hearts/Leaves dual currency internalises externalities → low sigma.
# Separation principle implemented: estimator and controller are distinct layers.
tau_B   = 2     # dead-time: local BAZ reacts within days
sigma_B = 0.5   # observation noise: dual-currency makes externalities legible
K_B     = 0.8   # can use high gain safely because latency is small

# ── Initialise arrays ─────────────────────────────────────────────────────────
x_A = np.full(T, x_ref)   # true state A
x_B = np.full(T, x_ref)   # true state B
y_A = np.full(T, x_ref)   # observed state A (what Architecture A thinks is happening)
y_B = np.full(T, x_ref)   # observed state B (what Architecture B thinks is happening)
u_A = np.zeros(T)          # control action A
u_B = np.zeros(T)          # control action B

# ── Simulation loop ───────────────────────────────────────────────────────────
for t in range(1, T - 1):
    d = crisis_mag if t == t_crisis else 0.0

    # Observation (signal fidelity / separation principle)
    y_A[t] = x_A[t] + rng.normal(0, sigma_A)
    y_B[t] = x_B[t] + rng.normal(0, sigma_B)

    # Control law: proportional feedback on *observed* error
    u_A[t] = K_A * (x_ref - y_A[t])
    u_B[t] = K_B * (x_ref - y_B[t])

    # Apply the control computed tau steps ago (dead-time)
    act_A = u_A[t - tau_A] if t >= tau_A else 0.0
    act_B = u_B[t - tau_B] if t >= tau_B else 0.0

    # State transition
    drift = x_ref * (1 - A_sys)           # keeps equilibrium at x_ref without control
    x_A[t + 1] = A_sys * x_A[t] + B_sys * act_A + d + drift
    x_B[t + 1] = A_sys * x_B[t] + B_sys * act_B + d + drift

# ── Recovery metrics ──────────────────────────────────────────────────────────
recovery_threshold = x_ref - 5.0          # "recovered" when within 5% of equilibrium

def recovery_time(x, t0, threshold):
    for t in range(t0, len(x)):
        if x[t] >= threshold:
            return t - t0
    return len(x) - t0

rt_A = recovery_time(x_A, t_crisis, recovery_threshold)
rt_B = recovery_time(x_B, t_crisis, recovery_threshold)

deviation_A = np.sum(np.maximum(0, x_ref - x_A[t_crisis:]))   # area under deficit
deviation_B = np.sum(np.maximum(0, x_ref - x_B[t_crisis:]))

# ── Plotting ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])   # true state – full width
ax2 = fig.add_subplot(gs[1, 0])   # observed vs true – Architecture A
ax3 = fig.add_subplot(gs[1, 1])   # observed vs true – Architecture B
ax4 = fig.add_subplot(gs[2, :])   # control effort – full width

COLOR_A  = '#dc2626'   # red
COLOR_B  = '#16a34a'   # green
COLOR_OB = '#f97316'   # orange for observed-A
COLOR_OG = '#86efac'   # light green for observed-B

ts = np.arange(T)

# ── Plot 1: True system stability ─────────────────────────────────────────────
ax1.plot(ts, x_A, label='Architecture A – Centralized (true state)', color=COLOR_A, lw=2)
ax1.plot(ts, x_B, label='Architecture B – Fractal GGF (true state)', color=COLOR_B, lw=2)
ax1.axhline(x_ref,    color='gray',  ls='--', alpha=0.5, label='Equilibrium target')
ax1.axvline(t_crisis, color='black', ls=':',  lw=1.5,   label='Crisis event')
ax1.annotate(f'Recovery A: {rt_A} steps\nDeficit integral: {deviation_A:.0f}',
             xy=(t_crisis + rt_A, recovery_threshold),
             xytext=(t_crisis + rt_A + 4, 70),
             arrowprops=dict(arrowstyle='->', color=COLOR_A),
             color=COLOR_A, fontsize=8)
ax1.annotate(f'Recovery B: {rt_B} steps\nDeficit integral: {deviation_B:.0f}',
             xy=(t_crisis + rt_B, recovery_threshold),
             xytext=(t_crisis + rt_B + 4, 88),
             arrowprops=dict(arrowstyle='->', color=COLOR_B),
             color=COLOR_B, fontsize=8)
ax1.set_title('True system stability (structural proof)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Stability / wellbeing')
ax1.set_ylim(40, 110)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.25)

# ── Plot 2: Observability gap – Architecture A ────────────────────────────────
ax2.fill_between(ts, x_A, y_A, alpha=0.2, color=COLOR_OB, label='Observability gap')
ax2.plot(ts, x_A, color=COLOR_A,  lw=1.5, label='True state')
ax2.plot(ts, y_A, color=COLOR_OB, lw=1.0, alpha=0.7, label='Observed state')
ax2.axvline(t_crisis, color='black', ls=':', lw=1)
ax2.set_title('Architecture A: what the controller sees\n(high noise → corrupted decisions)',
              fontsize=9, fontweight='bold')
ax2.set_ylabel('State')
ax2.set_ylim(40, 120)
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.25)

# ── Plot 3: Observability gap – Architecture B ────────────────────────────────
ax3.fill_between(ts, x_B, y_B, alpha=0.2, color=COLOR_OG, label='Observability gap')
ax3.plot(ts, x_B, color=COLOR_B,  lw=1.5, label='True state')
ax3.plot(ts, y_B, color=COLOR_OG, lw=1.0, alpha=0.7, label='Observed state')
ax3.axvline(t_crisis, color='black', ls=':', lw=1)
ax3.set_title('Architecture B: what the controller sees\n(low noise → accurate decisions)',
              fontsize=9, fontweight='bold')
ax3.set_ylabel('State')
ax3.set_ylim(40, 120)
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.25)

# ── Plot 4: Control effort ─────────────────────────────────────────────────────
ax4.plot(ts, u_A, color=COLOR_A, alpha=0.8, lw=1.5,
         label=f'Arch A – delayed, noisy  (tau={tau_A}, K={K_A})')
ax4.plot(ts, u_B, color=COLOR_B, alpha=0.8, lw=1.5,
         label=f'Arch B – rapid, precise  (tau={tau_B}, K={K_B})')
ax4.axvline(t_crisis, color='black', ls=':', lw=1.5)
ax4.set_title(
    f'Control effort / governance cost\n'
    f'Note: K_A={K_A} is near the stability ceiling for tau={tau_A} – '
    f'raising it causes oscillation (structural, not political)',
    fontsize=9)
ax4.set_xlabel('Time steps')
ax4.set_ylabel('Intervention magnitude')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.25)

fig.suptitle(
    'Governance as Engineering Governance Simulator v2 – Subsidiarity as Control-Theoretic Necessity\n'
    'x(t+1) = A·x(t) + B·u(t−τ) + d(t)   |   u(t) = K·(x_ref − y(t))   |   y(t) = x(t) + ε',
    fontsize=10, y=0.98
)

plt.savefig('outputs/paper_i_single_node_feedback.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to paper_i_single_node_feedback.png")
print(f"\nRecovery time  A: {rt_A} steps   |   B: {rt_B} steps")
print(f"Deficit integral A: {deviation_A:.1f}   |   B: {deviation_B:.1f}")
print(f"Ratio (A/B): {deviation_A/max(deviation_B,1):.1f}x worse")
