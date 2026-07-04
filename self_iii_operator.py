"""
Self III — The Operator: simulation for Part IV (the operator-seeded legitimacy spiral).

WHAT THIS DEMONSTRATES
----------------------
Inherited Unobservability (Part I) + the operator-seeded spiral (Part IV.3), made
quantitatively visible. A primitive-complete, legitimacy-safe governance loop is held
fixed, and a single scalar -- the interior fidelity phi of one operator node -- is swept
downward. The result to exhibit is a threshold in phi below which the otherwise well-formed
system crosses into the low-legitimacy attractor. It is phi, not any architectural
parameter, that moves the system across the threshold.

DISCIPLINE (Part V: Against Borrowed Formal Authority)
------------------------------------------------------
This file invents NO new dynamics. The plant, the Kalman filter, the legitimacy update
(B_eff = L*B, V = V0/L, hysteresis, delivery/transparency/betrayal terms) are reused
verbatim from Paper XIII, Appendix B, with its built-legitimacy parameter set. The
operator enters only as:
  (1) a persistent interior disturbance d_int driving the interior dimension x2 -- the
      standing generative pressure (grievance, injured standing) that does not decay on
      its own and must be perceived to be countered; and
  (2) an operator node that attenuates the interior measurement, C_true = diag(1, phi),
      while the institution's filter takes the muted reading at face value (C_filter = I).
The gap between what the node passes up (phi*x2) and what the institution believes it sees
(x2) IS the uncorrectable distortion of Part III; phi -> 0 is the null-space / Inherited
Unobservability case for the interior dimension.

phi is a parameter ON the existing observation channel, not a new dynamic. The numbers are
illustrative of a structural claim, not a calibration of any real system.

CHANGELOG
---------
2026-06-19  Initial run. Extension of Paper XIII Appendix B; built-legitimacy params
            unchanged. Added: interior disturbance d_int on x2; operator node C_true =
            diag(1, phi) with face-value filter C_filter = I. Transparency held at T=1,
            lambda=1, no deception (D=0) throughout, so phi is the only thing that varies.
            d_int = 0.10 chosen so the basin separatrix falls mid-range (phi* ~ 0.33):
            for phi >= 0.5 the high-L equilibrium is solidly stable; for phi <= 0.2 the
            low-L attractor is solidly reached; in between lies the separatrix, where
            seed-to-seed bistability is expected and reported, not suppressed (this is
            Paper XIII's numerically identified L_crit, here driven by phi alone).
"""

import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------

# --- Paper XIII plant (Appendix B.6, built-legitimacy baseline) -- UNCHANGED -------------
A   = 0.95 * np.eye(2)          # dynamics: slow self-decay toward target (origin)
B   = np.eye(2)                 # actuation
W   = 0.01 * np.eye(2)          # process-noise covariance
V0  = 0.05 * np.eye(2)          # baseline measurement-noise covariance
Q   = np.eye(2)                 # LQR state cost
R   = 0.1 * np.eye(2)           # LQR control cost

# --- Paper XIII legitimacy dynamics (built-legitimacy regime) -- UNCHANGED ---------------
ALPHA_DROP     = 0.12           # delivery sensitivity when performance worsening
ALPHA_RECOVERY = 0.03           # delivery sensitivity when performance improving
BETA           = 0.08           # transparency sensitivity
GAMMA          = 0.5            # betrayal sensitivity (inactive: no deception here)
DELTA          = 0.005          # exogenous trust drift
T_TRANSPARENCY = 1.0            # full transparency: x_rep = x, no suppression
L0             = 0.90           # initial legitimacy: start in the high-L, safe state

# --- Operator extension (the only new elements) -----------------------------------------
D_INT = 0.10                    # persistent interior disturbance on x2 (the operator dim)
                                # standing generative pressure; does not self-decay
E2 = np.array([0.0, 1.0])       # disturbance enters the interior dimension only

# --- Run controls -----------------------------------------------------------------------
T_STEPS  = 300                  # horizon (Paper XIII)
T_BURN   = 20                   # burn-in excluded from metrics (Paper XIII)
N_SEEDS  = 100                  # Monte Carlo seeds (Paper XIII)
PHI_GRID = np.round(np.linspace(0.0, 1.0, 31), 4)   # interior-fidelity sweep
STEADY_WINDOW = 100             # final steps used for steady-state legitimacy metric

OUTDIR = "outputs"

# ----------------------------------------------------------------------------------------
# Controller gain (LQR on nominal design system) -- Paper XIII -> K ~ 0.75 I
# ----------------------------------------------------------------------------------------
P_are = solve_discrete_are(A, B, Q, R)
K = np.linalg.solve(R + B.T @ P_are @ B, B.T @ P_are @ A)

# ----------------------------------------------------------------------------------------
# One run: fixed phi, fixed seed. Returns full L, x2, delivery-gap trajectories.
# ----------------------------------------------------------------------------------------
_SW  = np.sqrt(W[0, 0])    # process-noise std (W = 0.01 I -> 0.1)
_SV0 = np.sqrt(V0[0, 0])   # baseline measurement-noise std (V0 = 0.05 I)
_I2  = np.eye(2)

def run(phi, seed):
    """One trajectory. C_filter = I (the institution takes the muted reading at face
    value); the true measurement of the interior dim is attenuated by phi. Noise draws
    are scalar (covariances are scaled identities) and the 2x2 innovation inverse is
    explicit, for speed -- numerically identical to the matrix form."""
    rng = np.random.default_rng(seed)
    x    = np.zeros(2)        # true state, starts at target
    xhat = np.zeros(2)        # filter estimate (believes C = I)
    P    = 0.1 * _I2          # filter covariance
    L    = L0
    prev_gap2 = 0.0

    Wn = rng.normal(size=(T_STEPS, 2)) * _SW
    Vn = rng.normal(size=(T_STEPS, 2))   # scaled below by sv0 / sqrt(L_C)

    L_hist, x2_hist, gap_hist = np.empty(T_STEPS), np.empty(T_STEPS), np.empty(T_STEPS)

    for t in range(T_STEPS):
        u = -K @ xhat                                  # Paper XIII control law

        Lc   = max(L, 1e-3)                            # Paper XIII: V = V0 / L_C
        vstd = _SV0 / np.sqrt(Lc)
        # operator node: interior measurement attenuated by phi; institution unaware
        y = np.array([x[0] + Vn[t, 0] * vstd,
                      phi * x[1] + Vn[t, 1] * vstd])
        V = V0 / Lc

        # Kalman update, C_filter = I (face value) -> S = P + V
        S = P + V
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        Sinv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
        Kf = P @ Sinv
        xhat = xhat + Kf @ (y - xhat)
        P = (_I2 - Kf) @ P

        # legitimacy update (Paper XIII A.1); x_rep = x at full transparency, D = 0
        gap2  = float(x @ x)
        alpha = ALPHA_DROP if gap2 > prev_gap2 else ALPHA_RECOVERY
        L = min(max(L - alpha * gap2 + BETA * T_TRANSPARENCY + DELTA, 0.0), 1.0)
        prev_gap2 = gap2

        L_hist[t], x2_hist[t], gap_hist[t] = L, x[1], gap2

        # propagate true plant (B_eff = L*B) + standing interior disturbance
        x = A @ x + L * (B @ u) + Wn[t] + E2 * D_INT
        xhat = A @ xhat + L * (B @ u)                  # filter predict (knows its own L)
        P = A @ P @ A.T + W

    return L_hist, x2_hist, gap_hist

# ----------------------------------------------------------------------------------------
# Sweep phi over the grid, Monte Carlo over seeds, collect steady-state legitimacy.
# ----------------------------------------------------------------------------------------
def sweep():
    med = np.empty(len(PHI_GRID))
    lo  = np.empty(len(PHI_GRID))
    hi  = np.empty(len(PHI_GRID))
    for i, phi in enumerate(PHI_GRID):
        finals = np.empty(N_SEEDS)
        for s in range(N_SEEDS):
            L_hist, _, _ = run(phi, seed=1000 * i + s)
            finals[s] = L_hist[-STEADY_WINDOW:].mean()
        med[i] = np.median(finals)
        lo[i]  = np.percentile(finals, 5)
        hi[i]  = np.percentile(finals, 95)
    return med, lo, hi

# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"LQR gain K (diag) = {np.diag(K).round(4)}  (exact for Q=I, R=0.1 I)")
    print(f"Sweeping phi over {len(PHI_GRID)} points, {N_SEEDS} seeds each ...")

    med, lo, hi = sweep()

    # Threshold: largest phi at which median steady-state L has collapsed below 0.5
    # (midpoint between the safe high-L state and total collapse), read as the separatrix.
    collapsed = med < 0.5
    if collapsed.any():
        phi_star = PHI_GRID[collapsed][-1]
    else:
        phi_star = float("nan")

    L_at_1 = med[-1]
    L_at_0 = med[0]
    print(f"\nmedian steady-state L at phi=1.0 : {L_at_1:.3f}")
    print(f"median steady-state L at phi=0.0 : {L_at_0:.3f}")
    print(f"interior-fidelity threshold phi* : {phi_star:.3f}  (median L crosses 0.5)")

    # --- Representative trajectories: one safe phi, one collapsed phi ---
    phi_hi = 1.0
    phi_lo = 0.15   # solidly inside the collapsed regime (below phi* ~ 0.33)
    seed_demo = 7
    Lh_hi, x2_hi, g_hi = run(phi_hi, seed_demo)
    Lh_lo, x2_lo, g_lo = run(phi_lo, seed_demo)

    # ------------------------------------------------------------------ Figure 1: sweep
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.fill_between(PHI_GRID, lo, hi, alpha=0.20, color="#3b6ea5", label="5–95th percentile")
    ax.plot(PHI_GRID, med, color="#3b6ea5", lw=2.2, label="median steady-state $L$")
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    if phi_star == phi_star:
        ax.axvline(phi_star, color="#a53b3b", ls="--", lw=1.5,
                   label=fr"threshold $\phi^*\approx{phi_star:.2f}$")
    ax.set_xlabel(r"operator interior fidelity  $\phi$")
    ax.set_ylabel(r"steady-state legitimacy  $L$")
    ax.set_title("Inherited unobservability seeds the legitimacy spiral\n"
                 "(all architectural parameters fixed; only $\\phi$ varies)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/self_iii_operator_phi_sweep.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------- Figure 2: L(t) trajectories
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(Lh_hi, color="#2f8a4c", lw=2, label=fr"$\phi={phi_hi:.2f}$ (node perceives interior)")
    ax.plot(Lh_lo, color="#a53b3b", lw=2, label=fr"$\phi={phi_lo:.2f}$ (node blind to interior)")
    ax.set_xlabel("time step"); ax.set_ylabel(r"legitimacy  $L(t)$")
    ax.set_title("The operator-seeded performance–legitimacy spiral")
    ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/self_iii_operator_legitimacy_trajectories.png", dpi=150); plt.close(fig)

    # ------------------------------------------ Figure 3: interior state + delivery gap
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    a1.plot(x2_hi, color="#2f8a4c", lw=1.8, label=fr"$\phi={phi_hi:.2f}$")
    a1.plot(x2_lo, color="#a53b3b", lw=1.8, label=fr"$\phi={phi_lo:.2f}$")
    a1.set_xlabel("time step"); a1.set_ylabel(r"interior state  $x_2(t)$")
    a1.set_title("Uncontrolled interior dimension"); a1.legend(frameon=False, fontsize=9)
    a2.plot(g_hi, color="#2f8a4c", lw=1.8, label=fr"$\phi={phi_hi:.2f}$")
    a2.plot(g_lo, color="#a53b3b", lw=1.8, label=fr"$\phi={phi_lo:.2f}$")
    a2.set_xlabel("time step"); a2.set_ylabel(r"delivery gap  $\|x_{rep}\|^2$")
    a2.set_title("Standing delivery gap feeds the spiral"); a2.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/self_iii_operator_interior_and_gap.png", dpi=150); plt.close(fig)

    print(f"\nFigures written to {OUTDIR}/:")
    print("  self_iii_operator_phi_sweep.png")
    print("  self_iii_operator_legitimacy_trajectories.png")
    print("  self_iii_operator_interior_and_gap.png")

if __name__ == "__main__":
    main()
