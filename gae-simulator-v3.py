"""
Governance as Engineering Governance Simulator v3 — Multi-Node Subsidiarity Proof
=================================================================================
Extends v2 from a scalar state to a vector state over N coupled nodes,
proving *where* the decision is made matters — not just how fast.

Dynamics per node i:
    x_i(t+1) = A_sys * x_i(t)
               + beta * Σ_j∈neighbours(x_j(t) − x_i(t))   [contagion / coupling]
               + B * u_i(t − tau)
               + d_i(t)
               + drift

Architecture A — Centralized:
    Controller sees only the *national mean* of y(t).
    Applies a uniform correction to every node.
    → Spatial information is destroyed by aggregation.
    → Healthy nodes are disrupted; crisis nodes are under-served.

Architecture B — Fractal / BAZ (subsidiarity):
    Each node i observes y_i(t) with low noise.
    Applies its own local correction u_i = K_B * (x_ref − y_i(t)).
    → Crisis nodes react immediately; healthy nodes are untouched.

Key proof:
    With a localised shock (nodes 2 and 7 only) the centralized controller
    sees a "slight national dip", over-reacts uniformly, and spreads
    disruption. The distributed controller sees exactly where the problem is
    and contains it surgically. Subsidiarity is proven as an engineering
    necessity, not a political preference.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Reproducibility ───────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=7)

# ── Simulation parameters ─────────────────────────────────────────────────────
N          = 10       # number of nodes (municipalities / BAZs)
T          = 120      # time steps
t_crisis   = 20       # when the localised shock hits
crisis_mag = -45.0    # magnitude of the shock
x_ref      = 100.0    # target equilibrium for every node
A_sys      = 0.95     # natural decay per node
B_sys      = 1.0      # actuator effectiveness
drift      = x_ref * (1 - A_sys)   # keeps uncontrolled nodes near x_ref

beta       = 0.03     # coupling strength between nearest neighbours
                      # (crisis bleeds slowly into adjacent nodes)

# Nodes that are hit by the crisis
crisis_nodes = [2, 7]

# ── Architecture A: Centralized ───────────────────────────────────────────────
tau_A    = 12     # latency: policy cycle from observation to all-node action
sigma_A  = 6.0    # per-node measurement noise at the reporting layer
K_A      = 0.30   # gain ceiling imposed by latency (see v2 for derivation)

# ── Architecture B: Fractal / Distributed ────────────────────────────────────
tau_B    = 2      # latency: local BAZ reacts within days
sigma_B  = 0.5    # noise: dual-currency makes local state legible
K_B      = 0.45   # Higher than K_A is justified by lower latency, but the
                  # stability ceiling still applies. Rule of thumb ~1/tau gives
                  # ceiling ≈ 0.5; coupling tightens this further → 0.45 is safe.
                  # Point: B's *latency* advantage dominates over marginal gain.

# ── Helper: nearest-neighbour coupling step ────────────────────────────────────
def couple(x):
    """Return the diffusion correction for a 1-D chain of nodes."""
    correction = np.zeros(N)
    for i in range(N):
        left  = x[i - 1] if i > 0     else x[i]
        right = x[i + 1] if i < N - 1 else x[i]
        correction[i] = beta * ((left - x[i]) + (right - x[i]))
    return correction

# ── Initialise arrays ─────────────────────────────────────────────────────────
x_A  = np.full((T, N), x_ref)   # true states — centralized
x_B  = np.full((T, N), x_ref)   # true states — fractal
y_A  = np.full((T, N), x_ref)   # observations — centralized
y_B  = np.full((T, N), x_ref)   # observations — fractal
u_A  = np.zeros((T, N))          # control actions — centralized
u_B  = np.zeros((T, N))          # control actions — fractal

# ── Simulation loop ───────────────────────────────────────────────────────────
for t in range(1, T - 1):

    # Disturbance vector (only crisis nodes affected)
    d = np.zeros(N)
    if t == t_crisis:
        d[crisis_nodes] = crisis_mag

    # ── Architecture A: observe then aggregate → uniform control ──────────────
    y_A[t] = x_A[t] + rng.normal(0, sigma_A, N)

    # Centralized controller sees only the national mean — spatial info lost
    national_mean_error = x_ref - np.mean(y_A[t])
    u_A[t] = K_A * national_mean_error   # scalar broadcast to all nodes

    act_A  = u_A[t - tau_A] if t >= tau_A else 0.0   # still a scalar

    x_A[t + 1] = (A_sys * x_A[t]
                  + couple(x_A[t])
                  + B_sys * act_A          # same action everywhere
                  + d
                  + drift)

    # ── Architecture B: each node observes and acts locally ───────────────────
    y_B[t] = x_B[t] + rng.normal(0, sigma_B, N)

    u_B[t] = K_B * (x_ref - y_B[t])      # per-node control

    act_B  = u_B[t - tau_B] if t >= tau_B else np.zeros(N)

    x_B[t + 1] = (A_sys * x_B[t]
                  + couple(x_B[t])
                  + B_sys * act_B          # targeted per node
                  + d
                  + drift)

# ── Recovery metrics ──────────────────────────────────────────────────────────
threshold = x_ref - 5.0

def node_recovery_time(x_arr, t0, thr):
    times = []
    for i in range(N):
        rt = T - t0
        for t in range(t0, T):
            if x_arr[t, i] >= thr:
                rt = t - t0
                break
        times.append(rt)
    return np.array(times)

rt_A = node_recovery_time(x_A, t_crisis, threshold)
rt_B = node_recovery_time(x_B, t_crisis, threshold)

deficit_A = np.sum(np.maximum(0, x_ref - x_A[t_crisis:]), axis=0)
deficit_B = np.sum(np.maximum(0, x_ref - x_B[t_crisis:]), axis=0)

ts = np.arange(T)

# ── Plotting ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

ax_hm_A  = fig.add_subplot(gs[0, 0])   # heatmap A
ax_hm_B  = fig.add_subplot(gs[0, 1])   # heatmap B
ax_cr_A  = fig.add_subplot(gs[1, 0])   # crisis nodes A
ax_cr_B  = fig.add_subplot(gs[1, 1])   # crisis nodes B
ax_bar   = fig.add_subplot(gs[2, 0])   # deficit bar chart
ax_ctrl  = fig.add_subplot(gs[2, 1])   # control effort — crisis node 2

# ── Heatmaps: node × time ─────────────────────────────────────────────────────
vmin, vmax = 50, 110

im_A = ax_hm_A.imshow(x_A.T, aspect='auto', cmap='RdYlGn',
                       vmin=vmin, vmax=vmax,
                       extent=[0, T, N - 0.5, -0.5])
ax_hm_A.axvline(t_crisis, color='black', lw=1.5, ls=':', label='Crisis')
ax_hm_A.set_yticks(range(N))
ax_hm_A.set_yticklabels([f'Node {i}' for i in range(N)], fontsize=7)
ax_hm_A.set_title('Architecture A — Centralized\n(uniform response degrades all nodes)',
                   fontsize=9, fontweight='bold')
ax_hm_A.set_xlabel('Time step')
for n in crisis_nodes:
    ax_hm_A.annotate('⚡', xy=(t_crisis, n), fontsize=10, ha='center', va='center',
                     color='black')
plt.colorbar(im_A, ax=ax_hm_A, label='Stability')

im_B = ax_hm_B.imshow(x_B.T, aspect='auto', cmap='RdYlGn',
                       vmin=vmin, vmax=vmax,
                       extent=[0, T, N - 0.5, -0.5])
ax_hm_B.axvline(t_crisis, color='black', lw=1.5, ls=':', label='Crisis')
ax_hm_B.set_yticks(range(N))
ax_hm_B.set_yticklabels([f'Node {i}' for i in range(N)], fontsize=7)
ax_hm_B.set_title('Architecture B — Fractal / BAZ\n(local response contains crisis at source)',
                   fontsize=9, fontweight='bold')
ax_hm_B.set_xlabel('Time step')
for n in crisis_nodes:
    ax_hm_B.annotate('⚡', xy=(t_crisis, n), fontsize=10, ha='center', va='center',
                     color='black')
plt.colorbar(im_B, ax=ax_hm_B, label='Stability')

# ── Crisis node traces ─────────────────────────────────────────────────────────
colors_crisis = ['#ef4444', '#f97316']
colors_healthy = '#94a3b8'

for i, n in enumerate(crisis_nodes):
    ax_cr_A.plot(ts, x_A[:, n], color=colors_crisis[i], lw=2,
                 label=f'Node {n} (crisis)')
for n in range(N):
    if n not in crisis_nodes:
        ax_cr_A.plot(ts, x_A[:, n], color=colors_healthy, lw=0.7, alpha=0.4)
ax_cr_A.axvline(t_crisis, color='black', ls=':', lw=1.5)
ax_cr_A.axhline(x_ref, color='gray', ls='--', alpha=0.4)
ax_cr_A.set_title('Arch A: crisis + healthy nodes\n(healthy nodes disrupted by uniform policy)',
                  fontsize=9, fontweight='bold')
ax_cr_A.set_ylabel('Stability')
ax_cr_A.set_ylim(40, 115)
ax_cr_A.legend(fontsize=7)
ax_cr_A.grid(True, alpha=0.2)

for i, n in enumerate(crisis_nodes):
    ax_cr_B.plot(ts, x_B[:, n], color=colors_crisis[i], lw=2,
                 label=f'Node {n} (crisis)')
for n in range(N):
    if n not in crisis_nodes:
        ax_cr_B.plot(ts, x_B[:, n], color=colors_healthy, lw=0.7, alpha=0.4)
ax_cr_B.axvline(t_crisis, color='black', ls=':', lw=1.5)
ax_cr_B.axhline(x_ref, color='gray', ls='--', alpha=0.4)
ax_cr_B.set_title('Arch B: crisis + healthy nodes\n(healthy nodes unaffected; crisis isolated)',
                  fontsize=9, fontweight='bold')
ax_cr_B.set_ylabel('Stability')
ax_cr_B.set_ylim(40, 115)
ax_cr_B.legend(fontsize=7)
ax_cr_B.grid(True, alpha=0.2)

# ── Deficit bar chart per node ────────────────────────────────────────────────
node_idx = np.arange(N)
width = 0.35
bars_A = ax_bar.bar(node_idx - width / 2, deficit_A, width,
                    color='#dc2626', alpha=0.8, label='Arch A')
bars_B = ax_bar.bar(node_idx + width / 2, deficit_B, width,
                    color='#16a34a', alpha=0.8, label='Arch B')
for n in crisis_nodes:
    ax_bar.annotate('⚡', xy=(n, max(deficit_A[n], deficit_B[n]) + 20),
                    ha='center', fontsize=10)
ax_bar.set_xticks(node_idx)
ax_bar.set_xticklabels([f'N{i}' for i in range(N)])
ax_bar.set_title('Cumulative stability deficit per node\n(non-crisis nodes: Arch A suffers collateral damage)',
                 fontsize=9, fontweight='bold')
ax_bar.set_ylabel('Deficit integral')
ax_bar.legend(fontsize=8)
ax_bar.grid(True, alpha=0.2, axis='y')

total_A = deficit_A.sum()
total_B = deficit_B.sum()
ax_bar.text(0.98, 0.95,
            f'System total\nA: {total_A:.0f}\nB: {total_B:.0f}\nRatio: {total_A/max(total_B,1):.1f}×',
            transform=ax_bar.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ── Control effort on a crisis node ───────────────────────────────────────────
n_show = crisis_nodes[0]
ax_ctrl.plot(ts, u_A[:, n_show] if u_A.ndim == 2 else u_A,
             color='#dc2626', lw=1.5,
             label=f'Arch A — uniform (node {n_show} share)')

# For A the control is a scalar broadcast; scale visually to per-node
u_A_scalar = K_A * (x_ref - np.mean(y_A, axis=1))
ax_ctrl.plot(ts, u_A_scalar, color='#dc2626', lw=1.5,
             label=f'Arch A — national mean signal')
ax_ctrl.plot(ts, u_B[:, n_show], color='#16a34a', lw=1.5,
             label=f'Arch B — local node {n_show}')
ax_ctrl.axvline(t_crisis, color='black', ls=':', lw=1.5)
ax_ctrl.set_title(f'Control signal for node {n_show} (crisis node)\n'
                  f'Arch A responds to diluted national mean; B sees real local state',
                  fontsize=9, fontweight='bold')
ax_ctrl.set_xlabel('Time step')
ax_ctrl.set_ylabel('Intervention magnitude')
ax_ctrl.legend(fontsize=7)
ax_ctrl.grid(True, alpha=0.2)

fig.suptitle(
    'Governance as Engineering Governance Simulator v3 — Subsidiarity as Engineering Necessity\n'
    'Vector state x⃗(t+1) = A·x⃗(t) + coupling(x⃗) + B·u⃗(t−τ) + d⃗(t)\n'
    f'Crisis nodes: {crisis_nodes}  |  N={N} nodes  |  τ_A={tau_A}, τ_B={tau_B}  |  '
    f'σ_A={sigma_A}, σ_B={sigma_B}',
    fontsize=10, y=0.98
)

plt.savefig('outputs/gae-simulator-v3.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'Node':<8} {'Deficit A':>12} {'Deficit B':>12} {'Recovery A':>12} {'Recovery B':>12}")
print('─' * 56)
for i in range(N):
    tag = ' ⚡' if i in crisis_nodes else ''
    print(f"Node {i}{tag:<4} {deficit_A[i]:>12.1f} {deficit_B[i]:>12.1f} "
          f"{rt_A[i]:>12} {rt_B[i]:>12}")
print('─' * 56)
print(f"{'Total':<8} {total_A:>12.1f} {total_B:>12.1f}")
print(f"\nSystem-wide ratio A/B: {total_A/max(total_B,1):.1f}×")
print("Note: non-crisis node deficit in Arch A is collateral damage from uniform policy.")
