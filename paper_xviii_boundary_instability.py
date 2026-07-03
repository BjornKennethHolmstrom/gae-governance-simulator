"""
Paper XVIII — The Boundary Instability Principle: reflexive boundary drift
under local learning.

Self-contained simulation for the Governance as Engineering series.
One file per paper; figures written to outputs/.

WHAT THIS DEMONSTRATES
----------------------
The corrected minimal model of §3.1: two subsystems, each governed by a local
controller that believes its jurisdiction is closed and calibrated, coupled
through a channel eps(t) that depends on boundary clarity b(t) and on an
ACCUMULATED coupling stock c(t). Local learning adjusts gains k_i by gradient
descent on the local residual; the residual is exactly the cross-boundary
influence (plus model staleness) that the local model omits, so learning
absorbs coupling into internal gain — restoring perceived calm without
removing real coupling. Tested: the reflexive boundary cycle of §3.2, the
Boundary Dissolution Index of §5, and the Critical Learning Bandwidth of §4.

REGISTERED PREDICTIONS (falsification consequences bind)
---------------------------------------------------------
  P1. A sustained boundary oscillation (recurrent factorizable-calm ->
      hidden-accumulation -> collapse -> recovery) exists in a non-degenerate
      region R of (beta, eta) space.
      If P1 fails: §3.2's four-phase narrative is demoted to a transient.
  P2. The Boundary Dissolution Index rho_boundary — §5 defines it on
      cross-boundary PREDICTION-ERROR correlations; here, with one monitored
      variable per side, the largest singular value of the cross-correlation
      block reduces to |rolling corr| — rises during hidden accumulation with
      a POSITIVE lead time before b collapses.
      If P2 fails: §5 and §6.1 lose their support as defined and the index
      is revised or dropped. Both the error-based index (as registered) and
      a state-based variant |rolling corr(x_1, x_2)| are instrumented, so a
      failure of the registered form can be localised.
  P3. The viability window [eta_min(r_env), eta_max(dDelta/dtheta)] narrows
      as r_env and the reflexivity strength grow, and CLOSES for some
      parameter combinations (zero-viability, §4).
      If P3 fails: the Decomposability Frontier is demoted from [R within
      the model] to [H].

MODEL, WITH DECLARED CHOICES
----------------------------
State:      x_i(t+1) = (a_t - k_i) x_i + eps(t) x_j + w_i,   soft-saturated
            at X_SAT by tanh (institutional variables have finite range; the
            NF excursion is bounded, not divergent).
Coupling:   eps(t) = eps0 + alpha (1 - b(t)) + beta c(t)
Stock:      c(t+1) = (1 - mu_c) c + mu_c x_1 x_2 + nu (|dk_1| + |dk_2|)
Belief:     each subsystem predicts x_hat_i = (a0 - k_i) x_i — a CLOSED and
            CALIBRATED local model. Residual r_i = x_i(t+1) - x_hat_i is
            therefore the literal unmodeled influence: coupling + drift
            staleness + noise.
Learning:   dk_i = eta x_i r_i (gradient step on r_i^2 under the local
            model), gain leak (1 - lambda_k) per step, clip to [0, K_MAX].
Boundary:   b(t+1) = sigmoid( gamma_b (1 - mean|r|/R_SCALE)
                              - delta_b |eps| + h_b (b - 0.5) ).

Choices declared rather than hidden (each is load-bearing and each is the
minimal implementation of something the outline's narrative already asserts):

  (1) Gain leak lambda_k. Control effort is costly and un-renewed authority
      relaxes. Without it the gain update is a pure ratchet (the flaw flagged
      in review) and no relaxation oscillation is possible.
  (2) Coupling STOCK c(t), an EMA of the interaction product, rather than
      the instantaneous product beta x_1 x_2 of the outline. §3.2's phase 4
      requires coupling to PERSIST after collapse ("real coupling remains
      elevated"); an instantaneous product cannot persist. The stock is the
      minimal persistence mechanism, and institutionally the more faithful
      one: coupling channels are built by repeated interaction.
  (3) Policy-velocity channel nu. The outline's premise is dDelta/dtheta
      != 0, but its §3.1 model made coupling depend on theta only through
      states. Exploratory runs (recorded in the changelog) showed that with
      state-mediated reflexivity alone, faster learning is monotonically
      stabilising and NO eta_max exists. The nu term — each side's rule
      changes create interfaces that entangle the jurisdictions — is the
      minimal DIRECT dDelta/dtheta channel, and it is the knob P3 sweeps.
  (4) The h_b (b - 0.5) term is AR(1) self-excitation (memory), not
      hysteresis proper; the body text is corrected to match.
  (5) Residual convention: closed-model residual (regulator reading), not
      the outline's open-loop baseline x - a x. Registered as a change;
      it makes r_i the literal boundary-mismatch signal.

Numbers are illustrative of a structural claim, not a calibration of any
real system. [R within the model] applies to what the runs show; the
institutional reading is [IP], argued in the body.

EXPERIMENTS
-----------
  A. Phase-cycle exhibit + P1 regime map over (beta, eta): quiescent /
     cycling / locked-NF classification.                       -> P1
  B. Early-warning event study: superposed-epoch rho trajectories aligned
     at collapse; warned fraction and median lead for the error-based
     index (as registered) and the state-based variant.        -> P2
  C. Critical Learning Bandwidth: drift a_t = a0 + r_env t; eta sweep
     shows the two failure modes (slow -> state divergence; fast ->
     boundary dissolution); window width mapped over (r_env, nu) with the
     closure frontier.                                         -> P3

CHANGELOG
---------
2026-07-03  Initial construction, iterated in-session. Exploratory findings
            that shaped the final model, recorded for the paper:
            - beta (instantaneous product), X_SAT=6: NF excursion
              unrecoverable (eps ~ 20 >> K_MAX); permanent saturation.
            - Instantaneous-product coupling at recoverable scales: burst
              accumulation (~40 steps), no room for early warning; replaced
              by the stock c(t) per declared choice (2).
            - Locked-NF third regime found at high beta: after collapse,
              elevated eps holds b at 0 permanently (boundary never
              recovers). Kept and reported in the P1 map.
            - With nu=0: eta_max DOES NOT EXIST (fast learning monotonically
              stabilising). dDelta/dtheta requires a direct channel; nu
              added per declared choice (3).
            - P2 methodology upgraded mid-session: a naive threshold-crossing
              count reported 86-94% "warned" with ~390-step leads, but the
              leads approached the full cycle length and the state-based
              index was above threshold 45% of DEEP-CALM time — an
              always-on detector, not an early warning. Replaced by the ROC
              event study (detection vs deep-calm false alarms). Under the
              honest test, P2 as registered FAILS; see printed verdicts.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

SEED = 20260703

# ---------------------------------------------------------------------------
# Baseline parameters (§3.1)
# ---------------------------------------------------------------------------
A0       = 0.96     # local open-loop pole (shared, symmetric case)
EPS0     = 0.02     # baseline structural coupling
ALPHA    = 0.12     # boundary-confusion coupling: eps gains alpha*(1-b)
BETA     = 1.00     # state-interaction reflexivity (via the stock c)
NU       = 0.05     # policy-velocity reflexivity: the model's dDelta/dtheta
MU_C     = 0.02     # accumulation/decay rate of the coupling stock
SIGMA_W  = 0.06     # process noise std (excitation keeps learning alive)
ETA      = 0.05     # learning rate (baseline)
LAMBDA_K = 0.010    # gain leak per step (control effort cost)
K_MAX    = 6.0      # hard cap on gains
X_SAT    = 2.0      # soft saturation scale (tanh)

GAMMA_B  = 4.0      # boundary sensitivity to calm
DELTA_B  = 8.0      # boundary sensitivity to |coupling|
H_B      = 2.0      # boundary memory (self-excitation, not hysteresis)
S_B      = 2.0      # sigmoid steepness
R_SCALE  = 0.15     # residual normalisation in the boundary update
B0       = 0.95     # initial boundary clarity

RHO_WIN  = 150      # rolling window for rho_boundary (corr noise ~ 0.08)
RHO_THR  = 0.40     # warning threshold (~5x the window noise floor)


def sigmoid(z, s=S_B):
    return 1.0 / (1.0 + np.exp(-np.clip(s * z, -60.0, 60.0)))


def run(T=6000, eta=ETA, beta=BETA, nu=NU, alpha=ALPHA, eps0=EPS0, a0=A0,
        r_env=0.0, lambda_k=LAMBDA_K, sigma_w=SIGMA_W, seed=0,
        gamma_b=GAMMA_B, delta_b=DELTA_B, h_b=H_B, mu_c=MU_C):
    """One trajectory. Returns dict of time series."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=2) * sigma_w
    k1 = k2 = 0.0
    b = B0
    c = 0.0
    W = rng.normal(size=(T, 2)) * sigma_w

    xs = np.empty((T, 2)); ks = np.empty((T, 2))
    bs = np.empty(T); eps_h = np.empty(T); cs = np.empty(T)
    rs = np.empty((T, 2))
    for t in range(T):
        a_t = a0 + r_env * t
        eps = eps0 + alpha * (1.0 - b) + beta * c
        # closed, calibrated local beliefs vs true update
        xh1 = (a0 - k1) * x1
        xh2 = (a0 - k2) * x2
        xn1 = (a_t - k1) * x1 + eps * x2 + W[t, 0]
        xn2 = (a_t - k2) * x2 + eps * x1 + W[t, 1]
        xn1 = X_SAT * np.tanh(xn1 / X_SAT)
        xn2 = X_SAT * np.tanh(xn2 / X_SAT)
        r1 = xn1 - xh1
        r2 = xn2 - xh2
        # local learning with leak
        dk1 = eta * x1 * r1
        dk2 = eta * x2 * r2
        k1 = min(max((1.0 - lambda_k) * k1 + dk1, 0.0), K_MAX)
        k2 = min(max((1.0 - lambda_k) * k2 + dk2, 0.0), K_MAX)
        # coupling stock: state interaction + policy velocity
        c = (1.0 - mu_c) * c + mu_c * x1 * x2 + nu * (abs(dk1) + abs(dk2))
        # boundary clarity
        calm = 1.0 - 0.5 * (abs(r1) + abs(r2)) / R_SCALE
        b = sigmoid(gamma_b * calm - delta_b * abs(eps) + h_b * (b - 0.5))
        x1, x2 = xn1, xn2
        xs[t, 0], xs[t, 1] = x1, x2
        ks[t, 0], ks[t, 1] = k1, k2
        bs[t] = b; eps_h[t] = eps; cs[t] = c
        rs[t, 0], rs[t, 1] = r1, r2
    return dict(x=xs, k=ks, b=bs, eps=eps_h, c=cs, r=rs)


def rolling_abs_corr(u, v, win=RHO_WIN):
    """|rolling corr(u, v)| via cumulative sums. With one monitored variable
    per side, the largest singular value of the cross-boundary correlation
    block reduces to this."""
    T = len(u)
    cu, cv = np.cumsum(u), np.cumsum(v)
    cuu, cvv, cuv = np.cumsum(u * u), np.cumsum(v * v), np.cumsum(u * v)
    out = np.full(T, np.nan)
    n = float(win)
    for t in range(win, T):
        su = cu[t] - cu[t - win]; sv = cv[t] - cv[t - win]
        suu = cuu[t] - cuu[t - win]; svv = cvv[t] - cvv[t - win]
        suv = cuv[t] - cuv[t - win]
        vu = suu - su * su / n
        vv = svv - sv * sv / n
        if vu > 1e-14 and vv > 1e-14:
            out[t] = abs((suv - su * sv / n) / np.sqrt(vu * vv))
    return out


def collapse_events(b, min_sep=200):
    dc = np.where((b[:-1] >= 0.5) & (b[1:] < 0.5))[0]
    ev = []
    for t in dc:
        if not ev or t - ev[-1] > min_sep:
            ev.append(int(t))
    return ev


# ---------------------------------------------------------------------------
# A. Phase-cycle exhibit + P1 regime map
# ---------------------------------------------------------------------------
def sim_A():
    tr = run(T=6000, seed=3)
    b, eps, c, k, r, x = tr["b"], tr["eps"], tr["c"], tr["k"], tr["r"], tr["x"]
    rho_e = rolling_abs_corr(r[:, 0], r[:, 1])
    rho_s = rolling_abs_corr(x[:, 0], x[:, 1])
    ev = collapse_events(b)

    # exhibit window: around one mid-run event
    t0 = ev[2] if len(ev) > 2 else ev[-1]
    lo, hi = max(0, t0 - 500), min(len(b), t0 + 300)
    tt = np.arange(lo, hi)
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 9.0), sharex=True)
    axes[0].plot(tt, b[lo:hi], color="#1f3b57", lw=1.8)
    axes[0].set_ylabel("boundary clarity $b$")
    axes[0].axvline(t0, color="#b04632", ls="--", lw=1)
    axes[1].plot(tt, eps[lo:hi], color="#b04632", lw=1.8, label="$\\varepsilon$")
    axes[1].plot(tt, c[lo:hi], color="#8a6d3b", lw=1.4, label="stock $c$")
    axes[1].axhline(1 - A0, color="0.6", ls=":", lw=1,
                    label="$1-a_0$ (calm-phase instability level)")
    axes[1].set_ylabel("coupling"); axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(tt, k[lo:hi, 0], color="#3f7d3f", lw=1.6, label="$k_1$")
    axes[2].plot(tt, k[lo:hi, 1], color="#3f7d3f", lw=1.0, ls="--", label="$k_2$")
    axes[2].set_ylabel("learned gains"); axes[2].legend(frameon=False, fontsize=8)
    axes[3].plot(tt, rho_s[lo:hi], color="#6a3d9a", lw=1.8,
                 label="$\\rho$ (state-based)")
    axes[3].plot(tt, rho_e[lo:hi], color="#999", lw=1.4,
                 label="$\\rho$ (error-based, §5 as registered)")
    axes[3].axhline(RHO_THR, color="0.6", ls=":", lw=1)
    axes[3].set_ylabel("dissolution index"); axes[3].set_xlabel("step")
    axes[3].legend(frameon=False, fontsize=8)
    axes[0].set_title("A. One reflexive boundary cycle: calm, hidden accumulation, "
                      "collapse, recovery")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xviii_A_phase_cycle.png"), dpi=130)
    plt.close(fig)

    # P1 regime map over (beta, eta)
    betas = np.linspace(0.2, 2.4, 12)
    etas = np.geomspace(0.005, 0.5, 12)
    regime = np.zeros((len(betas), len(etas)))
    n_col = np.zeros_like(regime)
    for i, be in enumerate(betas):
        for j, et in enumerate(etas):
            frac_nf, ncol = [], []
            for s in range(2):
                tr = run(T=4000, beta=be, eta=et, seed=10 + s)
                bh = tr["b"][2000:]
                frac_nf.append(np.mean(bh < 0.5))
                ncol.append(len(collapse_events(tr["b"])))
            fnf, nc = np.mean(frac_nf), np.mean(ncol)
            n_col[i, j] = nc
            if fnf > 0.8:
                regime[i, j] = 2          # locked NF
            elif nc >= 2 and fnf > 0.005:
                regime[i, j] = 1          # cycling
            else:
                regime[i, j] = 0          # quiescent
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    pm = ax.pcolormesh(etas, betas, regime, cmap="RdYlBu_r", shading="nearest",
                       vmin=0, vmax=2)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate $\\eta$"); ax.set_ylabel("reflexivity $\\beta$")
    ax.set_title("A2. Regimes: quiescent (blue), cycling (yellow), "
                 "locked non-factorizable (red)")
    cb = fig.colorbar(pm, ticks=[0, 1, 2]); cb.ax.set_yticklabels(
        ["quiescent", "cycling", "locked NF"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xviii_A2_regime_map.png"), dpi=130)
    plt.close(fig)

    frac_cycling = float(np.mean(regime == 1))
    return dict(n_events_exhibit=len(ev), frac_cycling=frac_cycling,
                regimes=regime, betas=betas, etas=etas)


# ---------------------------------------------------------------------------
# B. Early-warning event study (P2) — ROC methodology
# ---------------------------------------------------------------------------
# A threshold-crossing count alone overstates an index that is simply "often
# on": with a ~600-step cycle, a detector that fires early in every calm
# phase produces long "leads" and looks supported. The honest test is a
# detection / false-alarm tradeoff:
#   detection:   index exceeds threshold somewhere in [t0-300, t0-10] before
#                a collapse at t0, while the dashboard is still green (b>0.9);
#   false alarm: index exceeds threshold during deep calm (b>0.9 and more
#                than 400 steps before the next collapse).
# Three detectors: the error-based index (§5 as registered), the state-based
# level variant, and the state-based variant detrended by its own trailing
# median (a rise detector).

DET_LO, DET_HI = 300, 10
CALM_GAP = 400


def trailing_median(v, win=400):
    out = np.full(len(v), np.nan)
    for t in range(win, len(v)):
        seg = v[t - win:t]
        seg = seg[~np.isnan(seg)]
        if len(seg) > win // 2:
            out[t] = np.median(seg)
    return out


def sim_B(n_seeds=10, T=6000):
    ev_wins = {"e": [], "s": [], "sd": []}
    calm_all = {"e": [], "s": [], "sd": []}
    span = 400
    stack = {"e": [], "s": [], "b": []}
    for s in range(n_seeds):
        tr = run(T=T, seed=100 + s)
        b, r, x = tr["b"], tr["r"], tr["x"]
        rho = {"e": rolling_abs_corr(r[:, 0], r[:, 1]),
               "s": rolling_abs_corr(x[:, 0], x[:, 1])}
        rho["sd"] = rho["s"] - trailing_median(rho["s"])
        ev = collapse_events(b)
        next_col = np.full(len(b), 10 ** 9)
        for t0 in reversed(ev):
            next_col[:t0] = np.minimum(next_col[:t0], t0 - np.arange(t0))
        calm = (b > 0.9) & (next_col > CALM_GAP)
        for key, v in rho.items():
            calm_all[key].append(v[calm & ~np.isnan(v)])
            for t0 in ev:
                lo, hi = t0 - DET_LO, t0 - DET_HI
                if lo < RHO_WIN + 400:
                    continue
                w = v[lo:hi]; bb = b[lo:hi]
                m = (bb > 0.9) & ~np.isnan(w)
                if m.sum() >= 20:
                    ev_wins[key].append((w[m], np.arange(lo, hi)[m], t0))
        for t0 in ev:
            if t0 - span >= 0:
                stack["e"].append(rho["e"][t0 - span:t0 + 50])
                stack["s"].append(rho["s"][t0 - span:t0 + 50])
                stack["b"].append(b[t0 - span:t0 + 50])

    def roc(key, n_th=60):
        calm = np.concatenate(calm_all[key])
        wins = ev_wins[key]
        ths = np.linspace(np.nanmin(calm), np.nanmax(calm) + 0.3, n_th)
        dets, fas, leads = [], [], []
        for th in ths:
            det = 0; ld = []
            for w, tt, t0 in wins:
                hit = np.where(w >= th)[0]
                if len(hit):
                    det += 1
                    ld.append(t0 - tt[hit[0]])
            dets.append(det / len(wins))
            fas.append(float(np.mean(calm >= th)))
            leads.append(np.median(ld) if ld else np.nan)
        dets, fas, leads = map(np.array, (dets, fas, leads))
        ok = fas <= 0.10
        if ok.any():
            i = np.argmax(dets * ok)
            best = dict(thr=float(ths[i]), detect=float(dets[i]),
                        fa=float(fas[i]), lead=float(leads[i]))
        else:
            best = None
        return dets, fas, best, len(wins)

    curves, bests = {}, {}
    for key in ("e", "s", "sd"):
        d, f, best, n_ev = roc(key)
        curves[key] = (f, d)
        bests[key] = best

    # figure: superposed epoch + ROC
    me = np.nanmean(np.array(stack["e"]), axis=0)
    ms = np.nanmean(np.array(stack["s"]), axis=0)
    mb = np.nanmean(np.array(stack["b"]), axis=0)
    tt = np.arange(-span, 50)
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax1.plot(tt, ms, color="#6a3d9a", lw=2.2, label="$\\rho$ state-based")
    ax1.plot(tt, me, color="#999", lw=2.0, label="$\\rho$ error-based (§5)")
    ax1.set_xlabel("steps relative to boundary collapse")
    ax1.set_ylabel("dissolution index (event mean)")
    ax2 = ax1.twinx()
    ax2.plot(tt, mb, color="#1f3b57", lw=1.6, alpha=0.7)
    ax2.set_ylabel("boundary clarity $b$", color="#1f3b57")
    ax1.axvline(0, color="#b04632", ls="--", lw=1)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.set_title("B. Superposed epoch at collapse")
    labels = {"e": "error-based (§5 as registered)",
              "s": "state-based (level)",
              "sd": "state-based (detrended)"}
    cols = {"e": "#999", "s": "#6a3d9a", "sd": "#b04632"}
    for key in ("e", "s", "sd"):
        f, d = curves[key]
        o = np.argsort(f)
        ax3.plot(f[o], d[o], lw=2, color=cols[key], label=labels[key])
    ax3.plot([0, 1], [0, 1], color="0.8", ls=":", lw=1)
    ax3.axvline(0.10, color="0.6", ls="--", lw=1)
    ax3.set_xlabel("false-alarm rate (deep calm)")
    ax3.set_ylabel("detection rate (pre-collapse, $b>0.9$)")
    ax3.set_title("B2. Early-warning ROC")
    ax3.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xviii_B_early_warning.png"), dpi=130)
    plt.close(fig)

    return dict(n_events=len(ev_wins["e"]), bests=bests)


# ---------------------------------------------------------------------------
# C. Critical Learning Bandwidth (P3)
# ---------------------------------------------------------------------------
ETA_GRID = np.geomspace(0.005, 5.0, 13)

def viable_metrics(eta, r_env, nu, seeds=3, T=5000):
    msx, sat, nf = [], [], []
    for s in range(seeds):
        tr = run(T=T, eta=eta, r_env=r_env, nu=nu, seed=200 + s)
        tail = slice(T // 2, None)
        x, b = tr["x"][tail], tr["b"][tail]
        msx.append(np.mean(x ** 2))
        sat.append(np.mean(np.abs(x) > 0.9 * X_SAT))
        nf.append(np.mean(b < 0.5))
    return float(np.mean(msx)), float(np.mean(sat)), float(np.mean(nf))


def sim_C():
    # (i) slice: the two failure modes vs eta at moderate reflexivity
    r_env0, nu0 = 5e-5, 0.15
    msxs, nfs = [], []
    for et in ETA_GRID:
        m, s_, n = viable_metrics(et, r_env0, nu0)
        msxs.append(m); nfs.append(n)
    fig, ax1 = plt.subplots(figsize=(7.8, 4.6))
    ax1.plot(ETA_GRID, msxs, "o-", color="#3f7d3f", lw=2, ms=4,
             label="mean square state (tracking failure)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("learning rate $\\eta$")
    ax1.set_ylabel("mean square state", color="#3f7d3f")
    ax2 = ax1.twinx()
    ax2.plot(ETA_GRID, nfs, "s-", color="#b04632", lw=2, ms=4,
             label="NF residence (boundary failure)")
    ax2.set_ylabel("fraction of time $b<0.5$", color="#b04632")
    ax2.axhline(0.2, color="#b04632", ls=":", lw=1)
    ax1.set_title("C. Two failure modes bound the learning rate\n"
                  f"($r_{{env}}$={r_env0}, $\\nu$={nu0}): slow learning loses the plant, "
                  "fast learning dissolves the boundary")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xviii_C_bandwidth_slice.png"), dpi=130)
    plt.close(fig)

    # (ii) window width over (r_env, nu)
    r_envs = np.linspace(0.0, 1.2e-4, 7)
    nus = np.linspace(0.0, 0.35, 8)
    width = np.zeros((len(nus), len(r_envs)))
    for i, nu in enumerate(nus):
        for j, re_ in enumerate(r_envs):
            ok = 0
            for et in ETA_GRID:
                m, s_, n = viable_metrics(et, re_, nu, seeds=2, T=4000)
                if s_ < 0.05 and m < 0.5 and n < 0.2:
                    ok += 1
            width[i, j] = ok / len(ETA_GRID)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    pm = ax.pcolormesh(r_envs, nus, width, cmap="viridis", shading="nearest",
                       vmin=0, vmax=width.max())
    cs = ax.contour(r_envs, nus, width, levels=[1e-9], colors="#b04632",
                    linewidths=2)
    ax.set_xlabel("environmental drift rate $r_{env}$")
    ax.set_ylabel("reflexivity $\\nu$  ($\\partial\\Delta/\\partial\\theta$)")
    ax.set_title("C2. Viability window width; red contour = closure "
                 "(zero-viability region above/right)")
    fig.colorbar(pm, label="fraction of $\\eta$ grid viable")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xviii_C2_window_map.png"), dpi=130)
    plt.close(fig)

    closed = width == 0
    return dict(width=width, r_envs=r_envs, nus=nus,
                any_closed=bool(closed.any()),
                closed_frac=float(closed.mean()),
                slice_nf=nfs, slice_msx=msxs)


if __name__ == "__main__":
    t_start = time.time()
    a = sim_A()
    b = sim_B()
    c = sim_C()

    print("=" * 70)
    print("PAPER XVIII — BOUNDARY INSTABILITY : VERIFIED RESULTS")
    print(f"(seed {SEED}; baseline a0={A0}, beta={BETA}, nu={NU}, eta={ETA})")
    print("=" * 70)

    print("\n[A] Phase cycle and P1 regime map")
    print(f"  collapse-recovery events in exhibit run (T=6000): "
          f"{a['n_events_exhibit']}")
    print(f"  fraction of (beta, eta) grid in the CYCLING regime: "
          f"{a['frac_cycling']:.2f}")
    print("  three regimes present: quiescent, cycling, locked-NF")
    print("  P1 VERDICT: " + ("SUPPORTED — sustained oscillation in a "
          "non-degenerate region" if a['frac_cycling'] > 0.05 else
          "NOT SUPPORTED"))

    print("\n[B] Early warning (P2): ROC over pre-collapse window "
          f"[t0-{DET_LO}, t0-{DET_HI}] vs deep calm")
    print(f"  events analysed: {b['n_events']}")
    names = {"e": "error-based (§5 as registered)",
             "s": "state-based (level)         ",
             "sd": "state-based (detrended)     "}
    for key in ("e", "s", "sd"):
        bb = b["bests"][key]
        if bb:
            print(f"  {names[key]}: best point with FA<=0.10 -> "
                  f"detect {bb['detect']:.2f}, FA {bb['fa']:.2f}, "
                  f"median lead {bb['lead']:.0f} steps (thr {bb['thr']:.2f})")
        else:
            print(f"  {names[key]}: no operating point with FA<=0.10")
    de = b["bests"]["e"]["detect"] if b["bests"]["e"] else 0.0
    ds = max((b["bests"][k]["detect"] if b["bests"][k] else 0.0)
             for k in ("s", "sd"))
    print("  P2 VERDICT: as registered (error-based), "
          + ("SUPPORTED" if de >= 0.8 else "NOT SUPPORTED")
          + f" (detect {de:.2f} at FA<=0.10);")
    print("              state-based variants "
          + ("SUPPORTED" if ds >= 0.8 else
             ("PARTIALLY SUPPORTED" if ds >= 0.5 else "NOT SUPPORTED"))
          + f" (best detect {ds:.2f} at FA<=0.10)")
    print("  (interpretation for §5: local adaptation absorbs coupling into "
          "gain, so local\n   prediction errors are exactly where the "
          "coupling is hardest to see — learning\n   launders the evidence. "
          "The index must be defined on cross-boundary STATE\n   covariances; "
          "even then, most of the discriminative signal is compressed into\n"
          "   the fast final approach to collapse.)")

    print("\n[C] Critical Learning Bandwidth (P3)")
    print(f"  slice at r_env=5e-5, nu=0.15: NF residence at eta ends = "
          f"{c['slice_nf'][0]:.2f} (slow) ... {c['slice_nf'][-1]:.2f} (fast); "
          f"minimum {min(c['slice_nf']):.2f} in between")
    print(f"  window closes somewhere on the (r_env, nu) grid: "
          f"{c['any_closed']}  (closed fraction of grid: "
          f"{c['closed_frac']:.2f})")
    print("  P3 VERDICT: " + ("SUPPORTED — zero-viability region exists"
          if c['any_closed'] else "NOT SUPPORTED"))

    print(f"\nRuntime: {time.time()-t_start:.0f}s")
    print("Figures written to outputs/: xviii_A_phase_cycle, "
          "xviii_A2_regime_map,\n  xviii_B_early_warning, "
          "xviii_C_bandwidth_slice, xviii_C2_window_map")
