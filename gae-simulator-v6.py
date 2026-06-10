"""
Governance as Engineering Governance Simulator v6 — Commons Governance and Requisite Variety
============================================================================================
Demonstrates that commons management is fundamentally a feedback loop
integrity problem. The tragedy of the commons is not a moral failure
but an architectural one: individual extraction decisions are made
without adequate feedback from the collective resource state.

Ashby's Law of Requisite Variety: a regulator must have at least as
much variety as the system it governs. Proximity — physical, seasonal,
relational, ecological — determines the effective variety available to
a governance system. Remote managers observing aggregate statistics
have structurally lower variety than local communities observing
the resource directly across multiple signal dimensions.

Key formal concepts
────────────────────
  Feedback loop integrity: the degree to which an agent's decisions
    are coupled to the consequences of those decisions. Zero integrity
    = open-loop (tragedy of the commons). Full integrity = closed-loop
    with immediate, multi-dimensional feedback.

  Requisite variety (Ashby, 1956): V_regulator ≥ V_disturbance.
    A governance system with variety less than the resource system's
    disturbance variety cannot maintain the resource within desired
    bounds — not because of institutional failure, but because the
    observation channel is too narrow to distinguish states that
    require different responses.

  Observation dimensionality: the number of independent signal
    dimensions available to the governance system. State management
    observes one aggregate dimension (stock level). Community commons
    observe multiple dimensions simultaneously (stock, distribution,
    seasonal state, ecological indicators, social pressure signals).
    Bioregional/indigenous governance additionally observes slow
    ecological variables invisible to external managers on short
    observation windows.

Resource and disturbance model
────────────────────────────────
  A shared renewable resource (e.g. fishery, forest, aquifer) with:
  - Logistic growth dynamics: dR/dt = r·R·(1 - R/K) - E(t)
  - Multi-scale disturbances: fast stochastic shocks, medium seasonal
    cycles, slow decadal trend (climate-driven carrying capacity shift)
  - Spatial heterogeneity: 12 resource patches with different
    productivity and connectivity
  - 20 user groups extracting from the resource

Governance architectures
──────────────────────────
  A — Open access       No governance. Pure individual optimisation.
  B — State management  Central regulator. Annual stock survey (high
                        latency, single dimension). Uniform quota.
  C — Market mechanism  Price signal proxies for scarcity. Medium
                        latency. Responds to aggregate, not distribution.
  D — Community commons Ostrom-style local rules. Low latency,
                        multi-dimensional observation. Graduated
                        sanctions. Boundary rules.
  E — Bioregional       Adds slow ecological signal dimensions:
                        seasonal indicators, species co-occurrence,
                        soil/water quality. Highest requisite variety.
                        Closest formal model of indigenous governance
                        systems with long-run ecological knowledge.

Output panels
──────────────
  1. Resource trajectory: stock level over time for all architectures
  2. Requisite variety diagram: observation dimensions vs disturbance
     frequency bands — which architectures cover which bands
  3. Extraction rate per user group over time (equity dimension)
  4. Resource collapse risk: fraction of time below 20% of K
  5. Feedback loop integrity: signal lag and dimensionality per arch
  6. Slow variable tracking: how well each architecture detects and
     responds to the decadal carrying-capacity trend
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

rng = np.random.default_rng(seed=42)

# ── Global resource parameters ────────────────────────────────────────────────
N_PATCHES  = 12      # spatial resource patches
N_USERS    = 20      # user groups
T          = 360     # time steps (months — 30 years)
K_BASE     = 100.0   # baseline carrying capacity per patch
r_GROWTH   = 0.08    # intrinsic growth rate per step
COLLAPSE   = 0.20    # collapse threshold (fraction of K)

# Disturbance parameters
FAST_SIGMA    = 3.0   # stochastic monthly shock std dev
MED_PERIOD    = 12    # seasonal cycle (12 months)
MED_AMP       = 8.0   # seasonal amplitude (carrying capacity variation)
SLOW_PERIOD   = 240   # decadal trend period
SLOW_AMP      = 20.0  # decadal carrying capacity decline amplitude

# ── Architecture definitions ──────────────────────────────────────────────────
ARCHITECTURES = {
    'A': {
        'name': 'Open access',
        'color': '#dc2626',
        'obs_lag': 0,          # no governance lag — immediate but uncoordinated
        'obs_dims': 1,         # only own patch stock visible to each user
        'quota_rigidity': 0.0, # no quota — unconstrained extraction
        'sanctioning': 0.0,    # no sanctions
        'slow_signal': False,  # no slow ecological signal
        'description': 'No governance;\npure individual optimisation'
    },
    'B': {
        'name': 'State management',
        'color': '#f97316',
        'obs_lag': 12,         # annual stock survey
        'obs_dims': 1,         # aggregate stock only
        'quota_rigidity': 0.7, # quotas but imperfect compliance
        'sanctioning': 0.3,    # weak enforcement
        'slow_signal': False,
        'description': 'Central regulator;\nannual survey, uniform quota'
    },
    'C': {
        'name': 'Market mechanism',
        'color': '#eab308',
        'obs_lag': 3,          # quarterly price signal
        'obs_dims': 1,         # price as single aggregate proxy
        'quota_rigidity': 0.0, # no quota — price signal only
        'sanctioning': 0.0,    # market self-regulates (theoretically)
        'slow_signal': False,
        'description': 'Price signal as\nscarcity proxy'
    },
    'D': {
        'name': 'Community commons',
        'color': '#2563eb',
        'obs_lag': 1,          # monthly community monitoring
        'obs_dims': 3,         # stock level + distribution + social pressure
        'quota_rigidity': 0.9, # strong community-enforced rules
        'sanctioning': 0.8,    # graduated sanctions
        'slow_signal': False,
        'description': 'Ostrom-style;\nlocal rules, monitoring, sanctions'
    },
    'E': {
        'name': 'Bioregional / indigenous',
        'color': '#16a34a',
        'obs_lag': 1,          # continuous relational monitoring
        'obs_dims': 6,         # + seasonal indicators, species co-occurrence,
                               #   soil/water quality, elder knowledge signals
        'quota_rigidity': 0.95,
        'sanctioning': 0.9,
        'slow_signal': True,   # observes slow ecological variables
        'description': 'Highest variety;\nseasonal, ecological, relational signals'
    },
}

# ── Resource dynamics ─────────────────────────────────────────────────────────
def carrying_capacity(t, slow_signal_available=False):
    """
    True carrying capacity: logistic baseline + seasonal cycle + slow decline.
    slow_signal_available: whether the governance system can observe this.
    """
    seasonal = MED_AMP * np.sin(2 * np.pi * t / MED_PERIOD)
    slow     = -SLOW_AMP * np.sin(2 * np.pi * t / SLOW_PERIOD)  # net decline
    return K_BASE + seasonal + slow

def logistic_growth(R, K, r):
    return r * R * (1 - R / max(K, 1.0))

def generate_patch_connectivity():
    """Spatial connectivity matrix — patches influence neighbours."""
    C = np.zeros((N_PATCHES, N_PATCHES))
    for i in range(N_PATCHES):
        for j in range(N_PATCHES):
            if i != j and abs(i - j) <= 2:
                C[i, j] = 0.02
    return C

# ── Extraction decisions per architecture ────────────────────────────────────
def compute_extraction(arch_key, t, R_obs, K_obs, price, arch):
    """
    Each user group decides how much to extract based on what the
    architecture allows them to observe and what constraints apply.
    Returns extraction per user (N_USERS,).
    """
    key = arch_key
    base_need = 2.0   # baseline need per user per step

    if key == 'A':
        # Open access: each user maximises extraction, no coordination
        # Extraction rises when stock is high, falls only at near-collapse
        greed = rng.uniform(0.8, 1.5, N_USERS)
        extraction = base_need * greed * (R_obs / max(R_obs, 1.0))
        extraction = np.clip(extraction, 0, R_obs / N_USERS * 1.8)

    elif key == 'B':
        # State quota: uniform allocation, imperfect compliance
        total_allowed = R_obs * 0.3   # 30% of observed stock
        quota_per_user = total_allowed / N_USERS
        compliance = rng.uniform(0.5, 1.0, N_USERS)
        # Some users over-extract — quota rigidity is partial
        extraction = quota_per_user * (arch['quota_rigidity'] * compliance
                                       + (1 - arch['quota_rigidity']) * 1.4)

    elif key == 'C':
        # Market: extraction responds to price signal
        # High price → reduce extraction (profitable to preserve)
        # Low price → increase extraction (cheap resource)
        price_effect = 1.0 / (1.0 + 0.1 * price)
        extraction = base_need * price_effect * rng.uniform(0.7, 1.3, N_USERS)

    elif key == 'D':
        # Community commons: rule-based with graduated sanctions
        # Extraction scaled to sustainable yield; sanctions for over-extraction
        sustainable = R_obs * r_GROWTH * 0.8 / N_USERS
        deviation   = rng.normal(0, 0.15, N_USERS)
        extraction  = sustainable * (1 + deviation)
        # Sanctions: users who deviate positively are penalised
        violators   = deviation > 0.1
        extraction[violators] *= (1 - arch['sanctioning'] * 0.5)
        extraction  = np.clip(extraction, 0.1, sustainable * 1.5)

    elif key == 'E':
        # Bioregional/indigenous: multi-dimensional signal drives
        # adaptive extraction respecting seasonal and slow signals
        # Extraction follows seasonal availability naturally
        seasonal_factor = 0.7 + 0.3 * np.sin(2 * np.pi * t / MED_PERIOD)
        sustainable = R_obs * r_GROWTH * seasonal_factor / N_USERS
        # Strong social accountability — very low variance
        deviation   = rng.normal(0, 0.06, N_USERS)
        extraction  = sustainable * (1 + deviation)
        extraction  = np.clip(extraction, 0.05, sustainable * 1.2)

    return np.maximum(extraction, 0)

# ── Observation channel ───────────────────────────────────────────────────────
def observe_resource(R_true, t, arch):
    """
    What the governance system actually sees: delayed, possibly
    aggregated, possibly noisy version of the true resource state.
    """
    lag   = arch['obs_lag']
    dims  = arch['obs_dims']
    t_obs = max(0, t - lag)

    # Base signal: aggregate stock (all architectures have this)
    R_obs = np.sum(R_true) + rng.normal(0, 2.0 * (5 - min(dims, 4)))

    # Multi-dimensional signal bonus: community and bioregional see
    # distribution across patches, not just aggregate
    if dims >= 3:
        # Distribution signal: can detect localised depletion
        R_obs = np.sum(R_true)   # more accurate — no aggregation loss
        R_obs += rng.normal(0, 1.0)

    # Slow signal: bioregional sees long-run trend
    K_obs = K_BASE   # default: assume stable K
    if arch['slow_signal']:
        K_obs = carrying_capacity(t, slow_signal_available=True)
        K_obs += rng.normal(0, 2.0)   # some noise even with local knowledge

    return max(R_obs, 0.1), K_obs

# ── Simulate one architecture ─────────────────────────────────────────────────
def simulate(arch_key):
    arch       = ARCHITECTURES[arch_key]
    C          = generate_patch_connectivity()
    R          = np.full(N_PATCHES, K_BASE * 0.7)   # initial stock
    price      = 5.0

    history_R  = np.zeros((T, N_PATCHES))
    history_E  = np.zeros((T, N_USERS))
    history_K  = np.zeros(T)

    for t in range(T):
        K_true = carrying_capacity(t)
        history_K[t] = K_true

        # Observe resource (lagged, possibly aggregated)
        R_obs, K_obs = observe_resource(R, t, arch)

        # Price: inverse of observed stock (normalised)
        price = 10.0 * (K_BASE / max(R_obs, 1.0))

        # Extraction decisions
        E_users = compute_extraction(arch_key, t, R_obs, K_obs, price, arch)
        E_total = E_users.sum()

        # Apportion extraction across patches proportional to stock
        patch_share = R / max(R.sum(), 0.01)
        E_patch = E_total * patch_share

        # Resource dynamics: growth + connectivity + disturbance - extraction
        fast_shock = rng.normal(0, FAST_SIGMA, N_PATCHES)
        growth = np.array([logistic_growth(R[p], K_true, r_GROWTH)
                           for p in range(N_PATCHES)])
        diffusion = C @ R - R * C.sum(axis=1)

        R = R + growth + diffusion + fast_shock - E_patch
        R = np.clip(R, 0, K_true * 1.5)

        history_R[t] = R
        history_E[t] = E_users

    return history_R, history_E, history_K

# ── Run all simulations ───────────────────────────────────────────────────────
print("Running commons governance simulations...\n")
results = {}
for key in ARCHITECTURES:
    print(f"  Simulating {key}: {ARCHITECTURES[key]['name']}...")
    R, E, K = simulate(key)
    total_stock = R.sum(axis=1)
    K_total = K * N_PATCHES
    collapsed = (total_stock < COLLAPSE * K_total).mean() * 100
    mean_stock_pct = (total_stock / K_total).mean() * 100
    gini = lambda x: (np.abs(np.subtract.outer(x, x)).mean() / (2 * x.mean() + 1e-9))
    mean_gini = np.mean([gini(E[t]) for t in range(10, T)])
    results[key] = {
        'R': R, 'E': E, 'K': K,
        'total_stock': total_stock,
        'K_total': K_total,
        'collapsed_pct': collapsed,
        'mean_stock_pct': mean_stock_pct,
        'mean_gini': mean_gini,
    }
    print(f"    Mean stock: {mean_stock_pct:.1f}% of K  |  "
          f"Collapse risk: {collapsed:.1f}%  |  "
          f"Extraction inequality (Gini): {mean_gini:.3f}")

# ── Plotting ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

ax_stock   = fig.add_subplot(gs[0, :])
ax_variety = fig.add_subplot(gs[1, 0:2])
ax_equity  = fig.add_subplot(gs[1, 2:4])
ax_slow    = fig.add_subplot(gs[2, 0:2])
ax_summary = fig.add_subplot(gs[2, 2:4])

ts = np.arange(T)

# ── Row 0: resource stock trajectories ───────────────────────────────────────
for key, arch in ARCHITECTURES.items():
    r = results[key]
    stock_pct = r['total_stock'] / r['K_total'] * 100
    ax_stock.plot(ts, stock_pct, color=arch['color'],
                  lw=2.0 if key in ('D', 'E') else 1.5,
                  alpha=0.9, label=f"{key} — {arch['name']}")

# True carrying capacity (normalised — always 100% by definition, show seasonal)
K_pct = results['E']['K'] / K_BASE * 100
ax_stock.plot(ts, K_pct, color='black', lw=0.8, ls=':', alpha=0.4,
              label='Carrying capacity (seasonal + trend)')
ax_stock.axhline(COLLAPSE * 100, color='red', lw=1.2, ls='--', alpha=0.7,
                 label=f'Collapse threshold ({int(COLLAPSE*100)}% of K)')
ax_stock.fill_between(ts, 0, COLLAPSE * 100, alpha=0.06, color='red')

# Mark decadal shift
ax_stock.axvline(SLOW_PERIOD // 2, color='purple', lw=0.8, ls=':', alpha=0.5)
ax_stock.text(SLOW_PERIOD // 2 + 3, 15, 'Slow ecological\nshift peak', 
              fontsize=7, color='purple', style='italic')

ax_stock.set_title(
    'Commons resource stock over 30 years (% of carrying capacity)\n'
    'All architectures face identical disturbances; differences are structural',
    fontsize=9, fontweight='bold')
ax_stock.set_ylabel('Stock (% of K)')
ax_stock.set_xlabel('Time (months)')
ax_stock.legend(fontsize=7, loc='lower left', ncol=3)
ax_stock.grid(True, alpha=0.2)
ax_stock.set_ylim(0, 140)
ax_stock.set_xlim(0, T)

# ── Row 1 left: requisite variety diagram ─────────────────────────────────────
# Show observation dimensions vs disturbance frequency bands
freq_bands = {
    'Fast\n(monthly shocks)': (0.08, 0.50),
    'Medium\n(seasonal cycle)': (0.06, 0.09),
    'Slow\n(decadal trend)': (0.0, 0.06),
}
obs_dims_map = {
    'A': 1, 'B': 1, 'C': 1, 'D': 3, 'E': 6
}
max_dims = 6

ax_variety.set_xlim(-0.5, 4.5)
ax_variety.set_ylim(0, max_dims + 1)

band_colors = ['#fca5a5', '#fde68a', '#bbf7d0']
band_labels = list(freq_bands.keys())

for bi, (band, (fmin, fmax)) in enumerate(freq_bands.items()):
    # Draw observation dimension coverage per architecture
    for ai, key in enumerate(ARCHITECTURES):
        arch   = ARCHITECTURES[key]
        dims   = obs_dims_map[key]
        covers = (
            (band == 'Fast\n(monthly shocks)'   and dims >= 1) or
            (band == 'Medium\n(seasonal cycle)'  and dims >= 3) or
            (band == 'Slow\n(decadal trend)'     and dims >= 6)
        )
        color = arch['color'] if covers else '#e5e7eb'
        ax_variety.add_patch(mpatches.FancyBboxPatch(
            (ai - 0.35, bi * 2 + 0.1), 0.7, 1.6,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='white', lw=1.5, alpha=0.85
        ))
        label = '✓' if covers else '✗'
        ax_variety.text(ai, bi * 2 + 0.95, label,
                        ha='center', va='center', fontsize=11,
                        color='white' if covers else '#9ca3af', fontweight='bold')

for bi, band in enumerate(band_labels):
    ax_variety.text(-0.5, bi * 2 + 0.95, band,
                    ha='right', va='center', fontsize=7.5)

ax_variety.set_xticks(range(5))
ax_variety.set_xticklabels([f"{k}\n({obs_dims_map[k]} dim{'s' if obs_dims_map[k]>1 else ''})"
                             for k in ARCHITECTURES], fontsize=8)
ax_variety.set_yticks([])
ax_variety.set_title(
    'Requisite variety coverage\nWhich architectures observe which disturbance bands',
    fontsize=8, fontweight='bold')
ax_variety.spines[['left', 'right', 'top', 'bottom']].set_visible(False)

# ── Row 1 right: extraction equity (Gini over time) ──────────────────────────
window = 12
for key, arch in ARCHITECTURES.items():
    E = results[key]['E']
    gini_series = []
    for t in range(T):
        e = E[t]
        if e.mean() > 0.01:
            g = np.abs(np.subtract.outer(e, e)).mean() / (2 * e.mean())
        else:
            g = 0
        gini_series.append(g)
    gini_smooth = np.convolve(gini_series, np.ones(window)/window, mode='same')
    ax_equity.plot(ts, gini_smooth, color=arch['color'],
                   lw=1.8, label=f"{key}")

ax_equity.set_title(
    'Extraction inequality over time (Gini coefficient)\n'
    'Lower = more equitable distribution among users',
    fontsize=8, fontweight='bold')
ax_equity.set_ylabel('Gini coefficient')
ax_equity.set_xlabel('Time (months)')
ax_equity.legend(fontsize=8)
ax_equity.grid(True, alpha=0.2)
ax_equity.set_xlim(0, T)
ax_equity.set_ylim(0, 0.6)

# ── Row 2 left: slow variable tracking ───────────────────────────────────────
# Show true K trend vs what each arch can detect (12-month rolling stock mean)
K_true_total = np.array([carrying_capacity(t) * N_PATCHES for t in range(T)])

ax_slow.plot(ts, K_true_total / (K_BASE * N_PATCHES) * 100,
             color='black', lw=1.5, ls='--', label='True carrying capacity')

for key, arch in ARCHITECTURES.items():
    r = results[key]
    # Proxy: 24-month rolling mean of stock as arch's "awareness" of K
    roll = np.convolve(r['total_stock'] / (K_BASE * N_PATCHES) * 100,
                       np.ones(24)/24, mode='same')
    ax_slow.plot(ts, roll, color=arch['color'], lw=1.5, alpha=0.75,
                 label=f"{key}")

ax_slow.set_title(
    'Slow variable tracking: carrying capacity trend detection\n'
    'Does the governance system detect the decadal ecological shift?',
    fontsize=8, fontweight='bold')
ax_slow.set_ylabel('Stock / K (%, 24-month rolling mean)')
ax_slow.set_xlabel('Time (months)')
ax_slow.legend(fontsize=7)
ax_slow.grid(True, alpha=0.2)
ax_slow.set_xlim(0, T)

# ── Row 2 right: summary bar chart ───────────────────────────────────────────
keys   = list(ARCHITECTURES.keys())
colors = [ARCHITECTURES[k]['color'] for k in keys]
x      = np.arange(len(keys))
w      = 0.28

mean_stocks   = [results[k]['mean_stock_pct']  for k in keys]
collapse_risk = [results[k]['collapsed_pct']   for k in keys]
gini_mean     = [results[k]['mean_gini'] * 100 for k in keys]

bars1 = ax_summary.bar(x - w, mean_stocks,   w, label='Mean stock (% of K)',
                        color=colors, alpha=0.85)
bars2 = ax_summary.bar(x,     collapse_risk, w, label='Collapse risk (%)',
                        color=colors, alpha=0.50, hatch='///')
bars3 = ax_summary.bar(x + w, gini_mean,    w, label='Gini × 100',
                        color=colors, alpha=0.35, hatch='...')

ax_summary.set_xticks(x)
ax_summary.set_xticklabels(keys, fontsize=10)
ax_summary.set_title(
    'Summary: mean stock, collapse risk, extraction inequality',
    fontsize=8, fontweight='bold')
ax_summary.legend(fontsize=7)
ax_summary.grid(True, alpha=0.2, axis='y')
ax_summary.set_ylabel('Value')

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print(f"{'Architecture':<28} {'Mean stock':>10} {'Collapse risk':>14} "
      f"{'Gini':>8} {'Obs dims':>9}")
print("─" * 72)
for key, arch in ARCHITECTURES.items():
    r = results[key]
    print(f"  {key} — {arch['name']:<22} "
          f"{r['mean_stock_pct']:>9.1f}%"
          f"{r['collapsed_pct']:>13.1f}%"
          f"{r['mean_gini']:>9.3f}"
          f"{arch['obs_dims']:>9}")
print("─" * 72)
print(f"\nSlow ecological shift: carrying capacity declines by ~{SLOW_AMP:.0f} units "
      f"over {SLOW_PERIOD} months")
print(f"Only Architecture E (obs_dims=6) observes slow signal directly")
print(f"Collapse threshold: {int(COLLAPSE*100)}% of carrying capacity")

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    'Governance as Engineering Governance Simulator v6 — Commons Governance and Requisite Variety\n'
    'Feedback loop integrity and observation dimensionality determine commons outcomes\n'
    'All architectures face identical resources and disturbances; differences are structural',
    fontsize=10, y=1.01
)

plt.savefig('outputs/gae-simulator-v6.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved to gae-simulator-v6.png")
