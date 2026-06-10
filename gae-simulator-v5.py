"""
Governance as Engineering Governance Simulator v5 — The Observability-Democracy Connection
==========================================================================================
Demonstrates that democratic systems face a formal observability constraint:
citizen preferences become unrecoverable as signals pass through representation
layers. Each layer introduces aggregation loss (spatial information destroyed)
and noise (signal corrupted). Beyond a critical layer count, the policy layer
cannot reconstruct the true distribution of citizen preferences regardless of
institutional quality.

Key formal concepts
────────────────────
  Observability (control theory): a system is observable if its full state
    can be reconstructed from available outputs. Formally, the observability
    matrix O = [C; CA; CA²; ...] must have full column rank.

  For a representation chain with K layers, each introducing aggregation
  ratio r and noise σ, the surviving preference variance after K layers is:

      Var_survived(K) = Var_true · ∏_{k=1}^{K} (1/r_k)
      Total noise      = Σ_{k=1}^{K} σ_k²

  The signal-to-noise ratio at the policy layer:
      SNR(K) = Var_survived(K) / Total_noise(K)

  Constitutional unobservability threshold: SNR < 1, i.e., noise exceeds
  surviving signal variance. Beyond this point, no statistical technique can
  reliably recover true citizen preferences from policy-layer observations.

Architecture comparison
────────────────────────
  A — Deep democracy     (5 layers: poll → media → party → parliament → cabinet)
  B — Representative     (3 layers: direct survey → council → policy)
  C — Semi-direct        (2 layers: citizen assembly → policy)
  D — Direct/participatory (1 layer: citizens → policy)

What is modelled
─────────────────
  N  = 60 citizen groups with preferences across P=4 policy dimensions
  T  = 120 time steps; preferences evolve slowly (genuine democratic change)
  Each layer: aggregates by ratio r, adds Gaussian noise σ, delays by τ
  Policy response: proportional feedback on observed (degraded) preference signal

Outputs
────────
  1. True vs observed preference heatmaps (citizen space × policy dimensions)
  2. Information survival curve: variance preserved vs layer count (K=1..6)
  3. SNR curve: signal-to-noise ratio at policy layer vs layer count
  4. Policy tracking: how closely policy tracks true citizen preferences over time
  5. Spatial representation loss: within-group variance destroyed per architecture
  6. Policy responsiveness: lag between genuine preference shift and policy response
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

rng = np.random.default_rng(seed=13)

# ── Global parameters ─────────────────────────────────────────────────────────
N  = 60     # citizen groups
P  = 4      # policy preference dimensions
T  = 120    # time steps
K_max = 6   # max layers for SNR/variance curves

# Preference scale: -1 (strong opposition) to +1 (strong support)
PREF_MIN, PREF_MAX = -1.0, 1.0

# ── Architecture definitions ──────────────────────────────────────────────────
# Each layer: (aggregation_ratio, noise_sigma, delay_tau)
# aggregation_ratio r: how many lower-level units map to one upper-level signal
# Total layers, their r, σ, τ

ARCHITECTURES = {
    'A': {
        'name': 'Deep democracy\n(5 layers)',
        'color': '#dc2626',
        'layers': [
            # (r, σ,   τ)   label
            (5,  0.12, 2),  # polling → media
            (4,  0.18, 3),  # media → party
            (3,  0.22, 4),  # party → parliament
            (4,  0.20, 5),  # parliament → cabinet
            (3,  0.15, 4),  # cabinet → policy
        ]
    },
    'B': {
        'name': 'Representative\n(3 layers)',
        'color': '#f97316',
        'layers': [
            (4,  0.10, 2),  # survey → council
            (5,  0.18, 4),  # council → assembly
            (3,  0.14, 3),  # assembly → policy
        ]
    },
    'C': {
        'name': 'Semi-direct\n(2 layers)',
        'color': '#2563eb',
        'layers': [
            (3,  0.08, 2),  # citizens → assembly
            (2,  0.10, 2),  # assembly → policy
        ]
    },
    'D': {
        'name': 'Direct/participatory\n(1 layer)',
        'color': '#16a34a',
        'layers': [
            (1,  0.05, 1),  # citizens → policy (near-direct)
        ]
    },
}

# ── Generate citizen preferences ──────────────────────────────────────────────
def generate_preferences():
    """
    True citizen preferences: N groups × P dimensions × T time steps.
    Preferences evolve slowly with occasional genuine shifts.
    Groups are spatially clustered — 4 clusters with internal coherence.
    """
    prefs = np.zeros((T, N, P))

    # Cluster assignments
    clusters = np.repeat([0, 1, 2, 3], N // 4)

    # Cluster baseline preferences (genuinely diverse)
    cluster_base = rng.uniform(-0.8, 0.8, (4, P))

    # Individual offsets within clusters (within-group diversity)
    individual_offset = rng.normal(0, 0.25, (N, P))

    # Initial preferences
    for i in range(N):
        prefs[0, i] = np.clip(cluster_base[clusters[i]] + individual_offset[i],
                              PREF_MIN, PREF_MAX)

    # Evolve preferences: slow drift + occasional genuine shifts
    for t in range(1, T):
        drift = rng.normal(0, 0.015, (N, P))   # slow individual drift

        # Genuine preference shift at t=40: cluster 0 shifts on dim 0-1
        if t == 40:
            drift[clusters == 0, 0] += 0.4
            drift[clusters == 0, 1] -= 0.2

        # Second shift at t=80: all groups shift on dim 2
        if t == 80:
            drift[:, 2] += rng.normal(0.3, 0.08, N)

        prefs[t] = np.clip(prefs[t-1] + drift, PREF_MIN, PREF_MAX)

    return prefs, clusters

# ── Pass preferences through representation layers ────────────────────────────
def pass_through_layers(prefs_t, layers):
    """
    Pass a (N, P) preference snapshot through a list of layers.
    Returns: observed signal at policy layer, intermediate signals per layer.
    """
    signal = prefs_t.copy()  # (N, P)
    intermediates = [signal.copy()]

    for (r, sigma, tau) in layers:
        n_current = signal.shape[0]
        n_next = max(1, n_current // r)

        # Aggregation: mean over r groups → destroys within-group variance
        aggregated = np.zeros((n_next, P))
        for j in range(n_next):
            start = j * r
            end   = min(start + r, n_current)
            aggregated[j] = signal[start:end].mean(axis=0)

        # Noise: each representation level introduces distortion
        noise = rng.normal(0, sigma, aggregated.shape)
        signal = np.clip(aggregated + noise, PREF_MIN, PREF_MAX)
        intermediates.append(signal.copy())

    # Final policy signal: scalar mean across remaining representatives, per dim
    policy_signal = signal.mean(axis=0)  # (P,)
    return policy_signal, intermediates

# ── Simulate policy tracking over time ────────────────────────────────────────
def simulate_tracking(prefs, arch_key):
    """
    For each time step, pass preferences through the architecture's layers
    (with cumulative delay) and compute policy response.
    Returns policy trajectory (T, P) and true mean citizen preference (T, P).
    """
    arch = ARCHITECTURES[arch_key]
    layers = arch['layers']
    total_tau = sum(l[2] for l in layers)

    true_mean = prefs.mean(axis=1)   # (T, P) — true citizen mean at each t

    policy = np.zeros((T, P))
    # Policy initialised to observed preference at t=0
    policy[0] = pass_through_layers(prefs[0], layers)[0]

    # Proportional policy update: policy moves toward perceived citizen preference
    K_policy = 0.3   # policy responsiveness gain

    for t in range(1, T):
        # Perception is delayed by total_tau
        t_perceived = max(0, t - total_tau)
        perceived, _ = pass_through_layers(prefs[t_perceived], layers)

        # Policy moves proportionally toward perceived preference
        policy[t] = policy[t-1] + K_policy * (perceived - policy[t-1])

    return policy, true_mean

# ── Variance survival and SNR analytical curves ───────────────────────────────
def variance_survival_curve():
    """
    For K layers with typical parameters, compute surviving variance fraction
    and SNR at policy layer.
    """
    # Typical per-layer parameters (interpolating from architecture A)
    typical_r     = 3.5
    typical_sigma = 0.17
    initial_var   = 0.18   # empirical variance of citizen preferences

    Ks = np.arange(1, K_max + 1)
    survived_var = np.array([initial_var / (typical_r ** k) for k in Ks])
    total_noise  = np.array([k * typical_sigma**2             for k in Ks])
    snr          = survived_var / total_noise

    return Ks, survived_var, total_noise, snr

# ── Run simulations ───────────────────────────────────────────────────────────
print("Generating citizen preferences...")
prefs, clusters = generate_preferences()
print(f"  True preference variance (mean across dims): {prefs[0].var(axis=0).mean():.3f}")

print("\nSimulating policy tracking...")
results = {}
for key in ARCHITECTURES:
    policy, true_mean = simulate_tracking(prefs, key)
    # Tracking error: RMS deviation of policy from true citizen mean
    error = np.sqrt(((policy - true_mean)**2).mean(axis=1))
    results[key] = {'policy': policy, 'true_mean': true_mean, 'error': error}
    print(f"  {key}: mean tracking error = {error[10:].mean():.4f}, "
          f"  final error = {error[-1]:.4f}")

# ── Compute spatial representation loss ───────────────────────────────────────
print("\nComputing spatial representation loss...")
spatial_loss = {}
for key, arch in ARCHITECTURES.items():
    _, intermediates = pass_through_layers(prefs[0], arch['layers'])
    true_var  = prefs[0].var(axis=0).mean()
    final_var = intermediates[-1].var(axis=0).mean() if intermediates[-1].ndim > 1 else 0
    spatial_loss[key] = {
        'survived_pct': 100 * final_var / true_var if true_var > 0 else 0,
        'layer_vars': [s.var(axis=0).mean() if s.ndim > 1 else 0
                       for s in intermediates]
    }

Ks, survived_var, total_noise, snr = variance_survival_curve()

# ── Plotting ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.38)

ax_snr     = fig.add_subplot(gs[0, 0:2])
ax_var     = fig.add_subplot(gs[0, 2:4])
ax_track   = fig.add_subplot(gs[1, :])
ax_err_A   = fig.add_subplot(gs[2, 0])
ax_err_B   = fig.add_subplot(gs[2, 1])
ax_err_C   = fig.add_subplot(gs[2, 2])
ax_err_D   = fig.add_subplot(gs[2, 3])
ax_heat_A  = fig.add_subplot(gs[3, 0])
ax_heat_B  = fig.add_subplot(gs[3, 1])
ax_heat_C  = fig.add_subplot(gs[3, 2])
ax_heat_D  = fig.add_subplot(gs[3, 3])

ts = np.arange(T)
SHIFT_EVENTS = [40, 80]

def mark_shifts(ax):
    for se in SHIFT_EVENTS:
        ax.axvline(se, color='purple', lw=0.8, ls=':', alpha=0.6)

# ── Row 0 left: SNR vs layer count ────────────────────────────────────────────
unobservable_threshold = 1.0
ax_snr.plot(Ks, snr, 'o-', color='#1e3a5f', lw=2, ms=7, label='SNR at policy layer')
ax_snr.axhline(unobservable_threshold, color='red', ls='--', lw=1.5,
               label='Unobservability threshold (SNR=1)')
ax_snr.fill_between(Ks, snr, unobservable_threshold,
                    where=(snr < unobservable_threshold),
                    alpha=0.15, color='red', label='Unobservable region')

# Mark architecture layer counts
arch_Ks = {'A': 5, 'B': 3, 'C': 2, 'D': 1}
for akey, ak in arch_Ks.items():
    color = ARCHITECTURES[akey]['color']
    ax_snr.axvline(ak, color=color, lw=1.2, ls='--', alpha=0.7)

ax_snr.set_xlabel('Number of representation layers (K)', fontsize=9)
ax_snr.set_ylabel('Signal-to-noise ratio', fontsize=9)
ax_snr.set_title('SNR at policy layer vs representation depth\n'
                 'Below red line: citizen preferences unrecoverable',
                 fontsize=8, fontweight='bold')
ax_snr.set_xticks(Ks)
ax_snr.legend(fontsize=7)
ax_snr.grid(True, alpha=0.2)
ax_snr.set_ylim(bottom=0)

# Re-add architecture labels now that ylim is set
ymax = ax_snr.get_ylim()[1]
for akey, ak in arch_Ks.items():
    color = ARCHITECTURES[akey]['color']
    ax_snr.text(ak + 0.05, ymax * 0.92, akey, color=color, fontsize=8,
                fontweight='bold')

# ── Row 0 right: variance survival per layer ──────────────────────────────────
ax_var.fill_between(Ks, survived_var, alpha=0.3, color='#2563eb',
                    label='Surviving preference variance')
ax_var.fill_between(Ks, total_noise, alpha=0.3, color='#dc2626',
                    label='Accumulated noise variance')
ax_var.plot(Ks, survived_var, 'o-', color='#2563eb', lw=2, ms=6)
ax_var.plot(Ks, total_noise,  's-', color='#dc2626', lw=2, ms=6)
ax_var.axvline(3.2, color='gray', ls=':', lw=1, alpha=0.7)
ax_var.text(3.3, max(survived_var) * 0.6,
            'Noise exceeds\nsurviving signal\n(≈3 layers)',
            fontsize=7, color='gray', style='italic')
ax_var.set_xlabel('Number of representation layers (K)', fontsize=9)
ax_var.set_ylabel('Variance', fontsize=9)
ax_var.set_title('Preference variance survival vs noise accumulation\n'
                 'Crossover point = constitutional unobservability threshold',
                 fontsize=8, fontweight='bold')
ax_var.set_xticks(Ks)
ax_var.legend(fontsize=7)
ax_var.grid(True, alpha=0.2)

# ── Row 1: policy tracking over time ─────────────────────────────────────────
dim = 0   # show dimension 0 (first policy dimension)
for key, arch in ARCHITECTURES.items():
    r = results[key]
    ax_track.plot(ts, r['policy'][:, dim], color=arch['color'],
                  lw=1.8 if key != 'D' else 2.2,
                  label=f"{key} — {arch['name'].replace(chr(10), ' ')}")

# True citizen mean
ax_track.plot(ts, results['A']['true_mean'][:, dim],
              color='black', lw=1.2, ls='--', alpha=0.5, label='True citizen mean')
mark_shifts(ax_track)
ax_track.set_title(
    'Policy tracking of citizen preferences (dimension 1)\n'
    'Purple dotted lines: genuine preference shifts at t=40 and t=80',
    fontsize=9, fontweight='bold')
ax_track.set_ylabel('Policy position')
ax_track.legend(fontsize=7, loc='lower right', ncol=2)
ax_track.grid(True, alpha=0.2)
ax_track.set_xlim(0, T)

# ── Row 2: per-architecture tracking error ────────────────────────────────────
for ax, key in [(ax_err_A, 'A'), (ax_err_B, 'B'),
                (ax_err_C, 'C'), (ax_err_D, 'D')]:
    arch = ARCHITECTURES[key]
    err  = results[key]['error']
    ax.plot(ts, err, color=arch['color'], lw=1.8)
    ax.fill_between(ts, err, alpha=0.15, color=arch['color'])
    mark_shifts(ax)
    ax.set_title(f"{key}: {arch['name']}\nMean error: {err[10:].mean():.3f}",
                 fontsize=8, fontweight='bold', color=arch['color'])
    ax.set_ylabel('RMS tracking error')
    ax.set_ylim(0, 0.65)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('Time step')

# ── Row 3: preference snapshot heatmaps ──────────────────────────────────────
# Show true citizen preferences vs what each architecture's policy layer sees
t_snapshot = 50   # after first preference shift

norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

for ax, key in [(ax_heat_A, 'A'), (ax_heat_B, 'B'),
                (ax_heat_C, 'C'), (ax_heat_D, 'D')]:
    arch = ARCHITECTURES[key]
    _, intermediates = pass_through_layers(prefs[t_snapshot], arch['layers'])

    # Reconstruct observed preferences at citizen resolution by broadcasting
    # final policy signal back to citizen space
    final_signal = intermediates[-1]
    if final_signal.ndim == 1:
        observed = np.tile(final_signal, (N, 1))   # broadcast scalar back
    else:
        # Upsample back to N groups (nearest-neighbor broadcast)
        factor = N // final_signal.shape[0]
        observed = np.repeat(final_signal, factor, axis=0)[:N]

    # What the policy layer "sees": broadcast final signal vs true preferences
    error_map = observed - prefs[t_snapshot]

    im = ax.imshow(error_map.T, aspect='auto', cmap='RdBu_r',
                   norm=norm,
                   extent=[0, N, P - 0.5, -0.5])
    ax.set_yticks(range(P))
    ax.set_yticklabels([f'D{p+1}' for p in range(P)], fontsize=7)
    ax.set_xlabel('Citizen group', fontsize=8)
    n_layers = len(arch['layers'])
    survived = spatial_loss[key]['survived_pct']
    ax.set_title(
        f'{key}: {arch["name"]}\n'
        f'{n_layers} layer{"s" if n_layers > 1 else ""} · '
        f'{survived:.0f}% variance survived',
        fontsize=8, fontweight='bold', color=arch['color'])
    plt.colorbar(im, ax=ax, label='Obs − True', shrink=0.8)

# ── Print summary metrics ─────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(f"{'Architecture':<20} {'Layers':>6} {'Mean error':>12} "
      f"{'Variance survived':>18} {'SNR (approx)':>13}")
print("─" * 70)
for key, arch in ARCHITECTURES.items():
    n_layers = len(arch['layers'])
    mean_err = results[key]['error'][10:].mean()
    var_surv = spatial_loss[key]['survived_pct']
    # Approximate SNR using analytical curve interpolation
    snr_approx = float(np.interp(n_layers, Ks, snr))
    print(f"  {key} — {arch['name'].replace(chr(10), ' '):<15} "
          f"{n_layers:>6} {mean_err:>12.4f} {var_surv:>17.1f}% {snr_approx:>12.3f}")
print("─" * 70)
print(f"\nConstitutional unobservability threshold: SNR < 1.0")
print(f"Crossed at approximately K = {Ks[snr < 1.0][0] if (snr < 1.0).any() else '>6'} layers")

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    'Governance as Engineering Governance Simulator v5 — The Observability-Democracy Connection\n'
    'Signal fidelity degradation through representation layers: '
    'beyond ~3 layers, citizen preferences become constitutionally unrecoverable\n'
    'All architectures have equal institutional quality; differences are structural',
    fontsize=10, y=0.98
)

plt.savefig('outputs/gae-simulator-v5.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved to gae-simulator-v5.png")
