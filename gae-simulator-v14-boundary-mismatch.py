#!/usr/bin/env python3
"""
gae-simulator-v14-boundary-mismatch.py
=====================================
Paper XII — Boundary Selection Deficits  (corrected rewrite)

Simulation of multi-jurisdiction system with perfect internal controllers,
varying only the relationship between jurisdictional boundaries and the
underlying coupling structure (stochastic block model).

Four scenarios:
  (a) Perfectly matched  — jurisdictions coincide with SBM clusters.
  (b) Westphalian        — random balanced partition.
  (c) Sykes-Picot        — each SBM block's strongest within-block edge is cut
                           (Appendix B.5 exact per-block procedure).
  (d) Adaptive           — starts Westphalian; merges high-B_est pairs after
                           τ_adj latency when B_est exceeds threshold.

Corrections vs. previous version
---------------------------------
1. Sykes-Picot construction (partition_sykes_picot):
   Previous code ran 5 000 random balanced partitions and picked the one with
   the highest total cross-boundary coupling weight — a global max-cut heuristic
   dominated by between-block edges that left the within-block structure intact.
   Fixed: follows Appendix B.5 exactly — within each SBM block, rank the three
   within-block pairs by ||K_ij||+||K_ji||, force the top pair's endpoints into
   different jurisdictions, then exhaustively optimise over remaining assignments.
   Result: 12/12 within-block pairs cut (vs ≤9/12 with the old heuristic).

2. B_est estimator:
   Previous code compared a single-timestep squared norm against a fixed prior
   noise floor (0.09), so B_est never reached the 0.3 threshold and adaptive
   renegotiation never triggered (the adaptive curve was effectively a noisier
   re-run of Westphalian).
   Fixed: accumulates w_in vectors and actual noise vectors over a rolling window,
   computes B = Var(w_in) / (Var(w_in) + Var(noise_sample)) using empirical
   noise variance that self-calibrates to the run. B_THRESH restored to
   appendix value 0.30 (meaningful because fix 6 ensures B_est reflects
   only cross-boundary coupling).

3. Loop gain estimator:
   Previous code averaged instantaneous sqrt(||w_in||²/||y_out||²), which is not
   the variance-ratio estimator and produced a paradox where "Perfect match" showed
   the highest loop gains. Fixed: accumulates w_in and y_out *vectors* over the
   full post-burn window, then computes sqrt(mean_var(w_in)/mean_var(y_out)) once.

4. Dead code removed:
   run_simulation() (never called, contained a global-state bug where
   pending_merges bled across MC seeds via function attribute storage) is gone.
   run_sim_metric() is the sole simulation entry point.

5. Pending-merge index stability:
   Merges now store frozensets of member nodes rather than integer indices,
   so earlier merges resizing `jur` cannot corrupt later merge lookups.

6. Within-jurisdiction coupling feedforward (critical correctness fix):
   Previous code applied the LQR gain per-subsystem independently, so the
   partition had no effect on closed-loop dynamics — all scenarios (a/b/c)
   produced identical stability curves. Fixed: after the per-subsystem LQR
   term, the controller subtracts K_full[i,j] @ x[j] for every co-jur j≠i,
   cancelling all within-jurisdiction coupling the controller can observe.
   The residual uncompensated disturbance is then cross-boundary coupling
   only — the quantity the M-Δ framework analyses.

Implementation matches Appendix B.7:
  N=12, k=3, T=500, burn-in=50, 100 MC seeds (20 for scenario d sweep), LQR.
"""

import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
import os, time
from itertools import permutations

os.makedirs('outputs', exist_ok=True)

# ── Global parameters (Appendix B.7) ─────────────────────────────────────────
N          = 12
k          = 3
M_BLOCKS   = 4
BLOCK_SIZE = 3
A_ii = 0.95 * np.eye(k)
B_i  = np.eye(k)
W    = 0.01 * np.eye(k)

Q_lqr = np.eye(k)
R_lqr = 0.1 * np.eye(k)

T_sim   = 500
T_burn  = 50
T_total = T_sim + T_burn

C_WITHIN  = 1.0
C_BETWEEN = 0.1

GAMMA_SWEEP = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])

D_THRESH = 1e4
N_MC     = 100
SEED_BASE = 20240601

# Renegotiation parameters (scenario d)
# B_THRESH restored to appendix B.7 value (0.30).  With the within-jurisdiction
# feedforward fix, B_est now reflects only cross-boundary coupling variance,
# so the original calibration is meaningful again.
T_RENEG  = 50
B_THRESH = 0.30
TAU_ADJ  = 10
B_WINDOW = 40   # rolling window length for B_est

# ── LQR gain ──────────────────────────────────────────────────────────────────
P_lqr = solve_discrete_are(A_ii, B_i, Q_lqr, R_lqr)
K_c   = np.linalg.inv(R_lqr + B_i.T @ P_lqr @ B_i) @ B_i.T @ P_lqr @ A_ii


# ── Coupling matrix generation ────────────────────────────────────────────────
def generate_coupling_matrices(rng, gamma):
    """K[i,j] ∈ ℝ^{k×k}: coupling from subsystem j → i, ||K[i,j]||_spec = gamma*c."""
    blocks = np.repeat(np.arange(M_BLOCKS), BLOCK_SIZE)
    R = np.zeros((N, N, k, k))
    for i in range(N):
        for j in range(N):
            if i != j:
                M = rng.standard_normal((k, k))
                U, _, Vt = np.linalg.svd(M, full_matrices=False)
                R[i, j] = U @ Vt
    C = np.where(blocks[:, None] == blocks[None, :],
                 C_WITHIN, C_BETWEEN).astype(float)
    np.fill_diagonal(C, 0.0)
    K = gamma * C[:, :, None, None] * R
    return K, blocks


# ── Partition constructors ────────────────────────────────────────────────────
def partition_perfect(blocks):
    return [np.where(blocks == b)[0].tolist() for b in range(M_BLOCKS)]


def partition_random(rng):
    perm = rng.permutation(N)
    return [perm[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].tolist()
            for i in range(M_BLOCKS)]


def partition_sykes_picot(K_full, blocks):
    """
    Appendix B.5: for each SBM block find the highest-weight within-block pair
    and force its two endpoints into different jurisdictions.  Then exhaustively
    enumerate all balanced (4×3) completions and pick the one that maximises
    total within-block cut weight.
    """
    # Step 1: per-block top pair
    top_pairs = []
    for b in range(M_BLOCKS):
        members = np.where(blocks == b)[0].tolist()
        best_w, best_pair = -1.0, None
        for ii in range(len(members)):
            for jj in range(ii + 1, len(members)):
                ni, nj = members[ii], members[jj]
                w = (np.linalg.norm(K_full[ni, nj]) +
                     np.linalg.norm(K_full[nj, ni]))
                if w > best_w:
                    best_w, best_pair = w, (ni, nj)
        pi, pj = best_pair
        third = [m for m in members if m not in (pi, pj)][0]
        top_pairs.append((pi, pj, third))

    def cut_weight(partition):
        assign = {}
        for jidx, members in enumerate(partition):
            for m in members:
                assign[m] = jidx
        total = 0.0
        for b in range(M_BLOCKS):
            bl = np.where(blocks == b)[0].tolist()
            for ii in range(len(bl)):
                for jj in range(ii + 1, len(bl)):
                    ni, nj = bl[ii], bl[jj]
                    if assign[ni] != assign[nj]:
                        total += (np.linalg.norm(K_full[ni, nj]) +
                                  np.linalg.norm(K_full[nj, ni]))
        return total

    # Step 2: enumerate assignments for (pi, pj) pairs, then fill thirds
    best_part, best_w = None, -1.0
    jur_choices = [(a, b) for a in range(M_BLOCKS)
                   for b in range(M_BLOCKS) if a != b]

    for c0 in jur_choices:
        for c1 in jur_choices:
            for c2 in jur_choices:
                for c3 in jur_choices:
                    choices = [c0, c1, c2, c3]
                    assign = {}
                    for b, (ja, jb) in enumerate(choices):
                        pi, pj, _ = top_pairs[b]
                        assign[pi] = ja
                        assign[pj] = jb
                    jur_count = [0] * M_BLOCKS
                    for node, jidx in assign.items():
                        jur_count[jidx] += 1
                    thirds = [top_pairs[b][2] for b in range(M_BLOCKS)]
                    slots  = [BLOCK_SIZE - jur_count[j] for j in range(M_BLOCKS)]
                    if any(s < 0 for s in slots):
                        continue
                    if sum(slots) != M_BLOCKS:
                        continue
                    available = []
                    for j in range(M_BLOCKS):
                        available.extend([j] * slots[j])
                    for perm in set(permutations(available)):
                        trial = dict(assign)
                        for b, jidx in enumerate(perm):
                            trial[thirds[b]] = jidx
                        part = [[] for _ in range(M_BLOCKS)]
                        for node, jidx in trial.items():
                            part[jidx].append(node)
                        if any(len(p) != BLOCK_SIZE for p in part):
                            continue
                        w = cut_weight(part)
                        if w > best_w:
                            best_w, best_part = w, [list(p) for p in part]

    return best_part


# ── B_est: empirical variance-ratio estimator ─────────────────────────────────
def estimate_B_windowed(w_in_vecs_window, noise_vecs_window, current_n_jur):
    """
    B_est_j = Var(w_in_j) / (Var(w_in_j) + Var(noise_j))
    using empirical per-dimension variances over the rolling window.

    w_in_vecs_window  : list of (n_jur, k) arrays — inflow vector per jur
    noise_vecs_window : list of (n_jur, k) arrays — summed noise per jur
    current_n_jur     : int — current jurisdiction count; frames from before
                        a merge (with a different shape) are discarded so that
                        np.array() receives a homogeneous stack.

    Returns array of shape (current_n_jur,).
    """
    # Filter out frames recorded under a different jurisdiction geometry
    w_filtered = [f for f in w_in_vecs_window  if f.shape[0] == current_n_jur]
    n_filtered = [f for f in noise_vecs_window if f.shape[0] == current_n_jur]

    if len(w_filtered) < 4:
        return np.zeros(current_n_jur)

    W_arr = np.array(w_filtered)   # (T, current_n_jur, k)
    N_arr = np.array(n_filtered)   # (T, current_n_jur, k)

    # Mean over k dimensions of per-dim variance over time
    var_w = np.mean(np.var(W_arr, axis=0), axis=-1)  # (current_n_jur,)
    var_n = np.mean(np.var(N_arr, axis=0), axis=-1)  # (current_n_jur,)

    B_est = var_w / (var_w + var_n + 1e-12)
    return B_est


# ── Single simulation run ─────────────────────────────────────────────────────
def run_sim_metric(jurisdictions_init, gamma, rng, renegotiate=False):
    """
    Run one simulation trajectory.

    Returns
    -------
    S              : float  — stability metric (negated avg squared norm)
    stable         : bool
    loop_gain_est  : float  — variance-ratio M-Δ loop gain for jurisdiction 0
    extra          : (B_est_series, jur_series) or None
    """
    K_full, blocks = generate_coupling_matrices(rng, gamma)
    jur = [list(j) for j in jurisdictions_init]
    x   = np.zeros((N, k))

    sum_sq = 0.0
    count  = 0

    # Loop gain accumulators for jurisdiction 0 (fixed; post burn-in)
    w_in_0_acc  = []
    y_out_0_acc = []

    # Rolling windows for B_est: store (n_jur, k) inflow and noise per step
    w_in_win  = []   # list of (n_jur, k)
    noise_win = []   # list of (n_jur, k)

    # Pending merges: (t_execute, frozenset_a, frozenset_b)
    # Member-set storage makes indices immune to earlier-merge resizing.
    pending_merges = []

    B_est_series = []
    jur_series   = []

    for t in range(T_total):

        # ── Control ──────────────────────────────────────────────────────────
        # Per-subsystem LQR stabilises nominal internal dynamics.
        # Within-jurisdiction coupling feedforward cancels all coupling the
        # controller can observe (co-jurisdictional states).  The only residual
        # uncompensated disturbance is then the cross-boundary coupling — which
        # is the quantity the paper's M-Δ framework analyses.
        u = np.zeros((N, k))
        for jm in jur:
            idx = np.array(jm)
            u[idx] = -(x[idx] @ K_c.T)
            for li, i in enumerate(idx):
                for lj, j in enumerate(idx):
                    if li != lj:
                        u[i] -= K_full[i, j] @ x[j]

        # ── Coupling ─────────────────────────────────────────────────────────
        coupling = np.zeros((N, k))
        for i in range(N):
            for j in range(N):
                if i != j:
                    coupling[i] += K_full[i, j] @ x[j]

        # ── Noise ────────────────────────────────────────────────────────────
        noise = rng.multivariate_normal(np.zeros(k), W, size=N)

        # ── State update ──────────────────────────────────────────────────────
        x_next = (A_ii @ x.T).T + u + coupling + noise

        # ── Divergence guard ─────────────────────────────────────────────────
        if np.sum(x_next ** 2) > D_THRESH:
            extra_div = (None, None) if renegotiate else None
            return -np.inf, False, 0.0, extra_div

        # ── Post-burn accumulation ────────────────────────────────────────────
        if t >= T_burn:
            sum_sq += np.sum(x_next ** 2)
            count  += 1

            # Loop gain (jurisdiction 0, fixed membership for comparability)
            j0   = jur[0]
            idx0 = np.array(j0)
            out0 = set(range(N)) - set(j0)

            w_in_0  = sum((K_full[i, j] @ x[j] for i in idx0 for j in out0),
                          np.zeros(k))
            y_out_0 = sum((K_full[j_out, i_in] @ x[i_in]
                           for j_out in out0 for i_in in idx0),
                          np.zeros(k))

            w_in_0_acc.append(np.array(w_in_0))
            y_out_0_acc.append(np.array(y_out_0))

        # ── B_est rolling window (scenario d) ────────────────────────────────
        if renegotiate:
            n_jur      = len(jur)
            w_in_frame = np.zeros((n_jur, k))
            ns_frame   = np.zeros((n_jur, k))
            for ji, jm in enumerate(jur):
                idx_j = np.array(jm)
                out_j = set(range(N)) - set(jm)
                for i in idx_j:
                    for jo in out_j:
                        w_in_frame[ji] += K_full[i, jo] @ x[jo]
                ns_frame[ji] = noise[idx_j].sum(axis=0)

            w_in_win.append(w_in_frame)
            noise_win.append(ns_frame)
            if len(w_in_win) > B_WINDOW:
                w_in_win.pop(0)
                noise_win.pop(0)

        x = x_next

        # ── Renegotiation trigger ─────────────────────────────────────────────
        if renegotiate and t > 0 and t % T_RENEG == 0:
            B_ests = estimate_B_windowed(w_in_win, noise_win, len(jur))
            if len(B_ests) != len(jur):
                B_ests = np.zeros(len(jur))

            B_est_series.append((t, float(B_ests.mean())))
            jur_series.append((t, [list(j) for j in jur]))

            high_B = np.where(B_ests > B_THRESH)[0]
            if len(high_B) >= 2:
                best_score = -1.0
                best_a_set = best_b_set = None
                for ii in range(len(high_B)):
                    for jj in range(ii + 1, len(high_B)):
                        a, b = int(high_B[ii]), int(high_B[jj])
                        cross = sum(
                            np.linalg.norm(K_full[ni, nj]) +
                            np.linalg.norm(K_full[nj, ni])
                            for ni in jur[a] for nj in jur[b]
                        )
                        if cross > best_score:
                            best_score = cross
                            best_a_set = frozenset(jur[a])
                            best_b_set = frozenset(jur[b])
                if best_a_set is not None:
                    pending_merges.append((t + TAU_ADJ, best_a_set, best_b_set))

        # ── Execute pending merges ────────────────────────────────────────────
        if pending_merges and renegotiate:
            still_pending = []
            for (t_exec, set_a, set_b) in pending_merges:
                if t >= t_exec:
                    idx_a = next((i for i, j in enumerate(jur)
                                  if frozenset(j) == set_a), None)
                    idx_b = next((i for i, j in enumerate(jur)
                                  if frozenset(j) == set_b), None)
                    if idx_a is not None and idx_b is not None and idx_a != idx_b:
                        merged = jur[idx_a] + jur[idx_b]
                        jur    = [j for i, j in enumerate(jur)
                                  if i not in (idx_a, idx_b)]
                        jur.append(merged)
                        w_in_win.clear()
                        noise_win.clear()
                    # else: already merged — skip silently
                else:
                    still_pending.append((t_exec, set_a, set_b))
            pending_merges[:] = still_pending

    # ── Final metrics ─────────────────────────────────────────────────────────
    S = -(sum_sq / count) if count > 0 else 0.0

    if len(w_in_0_acc) >= 2:
        arr_in  = np.array(w_in_0_acc)   # (T_sim, k)
        arr_out = np.array(y_out_0_acc)
        var_in  = np.mean(np.var(arr_in,  axis=0)) + 1e-12
        var_out = np.mean(np.var(arr_out, axis=0)) + 1e-12
        loop_gain_est = float(np.sqrt(var_in / var_out))
    else:
        loop_gain_est = 0.0

    extra = (B_est_series, jur_series) if renegotiate else None
    return S, True, loop_gain_est, extra


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Paper XII — Boundary Mismatch Simulation (v14, corrected)")
    print(f"N={N}, k={k}, T={T_sim}, burn={T_burn}, N_MC={N_MC}")
    print(f"B_THRESH={B_THRESH}, TAU_ADJ={TAU_ADJ}, B_WINDOW={B_WINDOW}")
    print(f"γ sweep: {GAMMA_SWEEP}\n")

    blocks = np.repeat(np.arange(M_BLOCKS), BLOCK_SIZE)

    # ── Fixed partitions ──────────────────────────────────────────────────────
    part_a = partition_perfect(blocks)

    rng_b  = np.random.default_rng(SEED_BASE + 2)
    part_b = partition_random(rng_b)

    K_ref, _ = generate_coupling_matrices(
        np.random.default_rng(SEED_BASE + 3), 0.3)
    print("Building Sykes-Picot partition (exhaustive search)...")
    t_sp = time.time()
    part_c = partition_sykes_picot(K_ref, blocks)
    print(f"  Done in {time.time()-t_sp:.1f}s\n")

    parts  = {'a': part_a, 'b': part_b, 'c': part_c}
    colors = {'a': '#1b9e77', 'b': '#d95f02', 'c': '#7570b3', 'd': '#e7298a'}
    labels = {'a': 'Perfect match (a)',  'b': 'Westphalian (b)',
              'c': 'Sykes-Picot (c)',    'd': 'Adaptive (d)'}

    # ── Partition diagnostics ─────────────────────────────────────────────────
    print("Partition diagnostics (within-block pairs cut / 12 total):")
    for sc, part in parts.items():
        assign = {}
        for ji, members in enumerate(part):
            for m in members:
                assign[m] = ji
        cuts = sum(
            1
            for b in range(M_BLOCKS)
            for bl in [np.where(blocks == b)[0].tolist()]
            for ii in range(len(bl))
            for jj in range(ii + 1, len(bl))
            if assign[bl[ii]] != assign[bl[jj]]
        )
        print(f"  ({sc}): {cuts}/12 within-block pairs cut")
    print()

    # ── Run scenarios a, b, c ─────────────────────────────────────────────────
    results    = {sc: {g: [] for g in GAMMA_SWEEP} for sc in ['a', 'b', 'c']}
    loop_gains = {sc: {g: [] for g in GAMMA_SWEEP} for sc in ['a', 'b', 'c']}

    total_runs = len(GAMMA_SWEEP) * N_MC * 3
    cnt        = 0
    t_start    = time.time()

    for gamma in GAMMA_SWEEP:
        for sc in ['a', 'b', 'c']:
            for seed in range(N_MC):
                rng = np.random.default_rng(SEED_BASE + 1000 * cnt)
                S, stable, lg, _ = run_sim_metric(
                    parts[sc], gamma, rng, renegotiate=False)
                if stable:
                    results[sc][gamma].append(S)
                    loop_gains[sc][gamma].append(lg)
                cnt += 1
        elapsed = time.time() - t_start
        eta     = elapsed / cnt * (total_runs - cnt) if cnt else 0
        print(f"  γ={gamma:.2f}  [{cnt}/{total_runs}]  "
              f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    print(f"\nCompleted {cnt} runs in {time.time()-t_start:.1f}s")

    # ── Figure 1: Stability surface + loop gain ───────────────────────────────
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for sc in ['a', 'b', 'c']:
        medians, q1s, q3s, lg_meds = [], [], [], []
        for gamma in GAMMA_SWEEP:
            arr = np.array(results[sc][gamma])
            medians.append(np.median(arr))
            q1s.append(np.percentile(arr, 25))
            q3s.append(np.percentile(arr, 75))
            lg_meds.append(np.median(loop_gains[sc][gamma]))
        ax1.fill_between(GAMMA_SWEEP, q1s, q3s, alpha=0.18, color=colors[sc])
        ax1.plot(GAMMA_SWEEP, medians, 'o-',
                 color=colors[sc], lw=2, label=labels[sc])
        ax2.plot(GAMMA_SWEEP, lg_meds, 'o-',
                 color=colors[sc], lw=2, label=labels[sc])

    ax1.set_xlabel('Coupling strength γ')
    ax1.set_ylabel('Stability S  (higher = more stable)')
    ax1.set_title('Stability vs coupling strength\n(median ± IQR, 100 MC seeds)')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.axhline(0, color='gray', lw=0.8)

    ax2.set_xlabel('Coupling strength γ')
    ax2.set_ylabel('Estimated M-Δ loop gain')
    ax2.set_title('Loop gain vs coupling strength\n(variance-ratio estimator)')
    ax2.axhline(1.0, color='red', ls='--', lw=1.2, label='Unity gain threshold')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig1.suptitle(
        'Boundary Mismatch Simulation (v14, corrected)\n'
        'Perfect match most stable; Sykes-Picot degrades earliest',
        fontsize=11)
    plt.tight_layout()
    fig1.savefig('outputs/v14-stability-loopgain.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/v14-stability-loopgain.png")

    # ── Scenario (d) sweep ────────────────────────────────────────────────────
    print("\nRunning scenario (d) sweep (20 seeds per γ)...")
    res_d  = {}
    t_d    = time.time()
    for gamma in GAMMA_SWEEP:
        slist = []
        for seed in range(20):
            rng = np.random.default_rng(SEED_BASE + 3000 + int(gamma * 1000) + seed)
            S, stable, _, _ = run_sim_metric(
                part_b, gamma, rng, renegotiate=True)
            if stable:
                slist.append(S)
        res_d[gamma] = np.median(slist) if slist else -np.inf
    print(f"Scenario d sweep done in {time.time()-t_d:.1f}s")

    fig2, ax = plt.subplots(figsize=(8, 5))
    med_b = [np.median(results['b'][g]) for g in GAMMA_SWEEP]
    ax.plot(GAMMA_SWEEP, med_b, 's-',
            color=colors['b'], lw=2, label='Westphalian (b) — 100 seeds')
    ax.plot(GAMMA_SWEEP, [res_d[g] for g in GAMMA_SWEEP], 'o-',
            color=colors['d'], lw=2, label='Adaptive renegotiation (d) — 20 seeds')
    ax.set_xlabel('Coupling strength γ')
    ax.set_ylabel('Median stability S')
    ax.set_title('Adaptive renegotiation vs Westphalian baseline')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig('outputs/v14-adaptive-sweep.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/v14-adaptive-sweep.png")

    # ── Scenario (d) single-run trajectory ───────────────────────────────────
    # Use γ=0.30 for trajectory: enough coupling to trigger renegotiation
    gamma_traj = 0.30
    print(f"\nRunning single trajectory for scenario (d) at γ={gamma_traj}...")
    rng_traj = np.random.default_rng(SEED_BASE + 9999)
    S_d, stable_d, lg_d, (B_hist, jur_hist) = run_sim_metric(
        part_b, gamma_traj, rng_traj, renegotiate=True)
    print(f"  S={S_d:.3f}, stable={stable_d}, loop_gain={lg_d:.3f}")
    if B_hist:
        vals = [v for _, v in B_hist]
        print(f"  B_est values: {[f'{v:.3f}' for v in vals]}")
        print(f"  Triggered above {B_THRESH}: "
              f"{sum(1 for v in vals if v > B_THRESH)} time(s)")
    if jur_hist:
        print(f"  Jurisdiction counts: {[len(j) for _, j in jur_hist]}")

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    if B_hist:
        times_b, vals_b = zip(*B_hist)
        ax3a.plot(times_b, vals_b, 'o-', color=colors['d'], ms=5)
        ax3a.axhline(B_THRESH, color='red', ls='--', lw=1.2,
                     label=f'Renegotiation threshold (B={B_THRESH})')
        ax3a.set_ylabel('Estimated B  (mean across jurisdictions)')
        ax3a.set_title(
            f'Boundary mismatch estimate over time  (scenario d, γ={gamma_traj})')
        ax3a.set_ylim(bottom=0)
        ax3a.legend(); ax3a.grid(True, alpha=0.3)

    if jur_hist:
        times_j = [t for t, _ in jur_hist]
        n_jur   = [len(j) for _, j in jur_hist]
        ax3b.step(times_j, n_jur, where='post', color=colors['d'], lw=2)
        ax3b.set_ylabel('Number of jurisdictions')
        ax3b.set_xlabel('Time step')
        ax3b.set_title('Jurisdictional consolidation over time')
        ax3b.set_ylim(0, M_BLOCKS + 1)
        ax3b.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig('outputs/v14-adaptive-trajectory.png', dpi=150,
                 bbox_inches='tight')
    print("Saved: outputs/v14-adaptive-trajectory.png")

    plt.show()
    print("\nAll done.")


if __name__ == '__main__':
    main()
