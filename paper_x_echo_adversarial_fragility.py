"""
paper_x_echo_adversarial_fragility.py
--------------------------------------------------------------------
Does the variance-optimal observer allocation build its own attack surface?

The refined, correct claim: it does -- but only against dependence the
optimizer could NOT measure. This separates two covariances:

  Sigma_est  : the defender's estimate, built from OBSERVABLE structure
               (shared fabs / funders / models it can audit -> Gamma).
  Sigma_true : reality, which also contains a HIDDEN common-input factor
               (a shared reanalysis dataset / timing source) that Gamma
               does not flag, so Sigma_est omits it.

Allocations are optimized on Sigma_est; all damage is evaluated on Sigma_true.

Attack = inject a shared error component on a target set S (rank-1, PSD):
    realized_var(w,S,s) = w^T Sigma_true w + s*( sum_{i in S} w_i sigma_true_i )^2

STRUCTURAL adversary (headline) has two moves:
  (a) observable-factor move  -> S = loaders of a factor visible in Sigma_est
  (b) hidden common-input move (SPOOF) -> S = loaders of the hidden factor,
      which Sigma_est treats as independent.

OMNISCIENT adversary (bound): worst k channels, chosen against the weights.

Allocations:
  flat   : 1/N
  gmv    : argmin w^T Sigma_est w                    (variance-optimal)
  robust : argmin_w max over OBSERVABLE-factor attacks (Gamma-aware defence)

Graduation test (fixed in advance): if gmv only pays a few % over flat -> a
smooth cost -> section. If gmv crosses ABOVE flat past a threshold in attack
strength or in the size of the unmeasured dependence -> sharp regime -> result.
"""

import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


def build_ensemble(hidden_load=0.35):
    """24 'crowd' channels with visible mutual correlation; 12 'clean' channels
    that LOOK independent and precise in Sigma_est but secretly share a hidden
    common-input factor. GMV piles onto the clean set; that is the trap."""
    n_crowd, n_clean = 24, 12
    N = n_crowd + n_clean
    idx = {"crowd": np.arange(0, n_crowd), "clean": np.arange(n_crowd, N)}

    def col(members, load):
        v = np.zeros(N); v[members] = load; return v

    # observable factors (in Sigma_est): two overlapping crowd factors
    A_obs = np.stack([
        col(idx["crowd"][:16], 0.8),
        col(idx["crowd"][8:], 0.8),
    ], axis=1)

    d = np.empty(N)
    d[idx["crowd"]] = 0.40                 # crowd: moderate idiosyncratic noise
    d[idx["clean"]] = 0.12                 # clean: look precise -> GMV loves them

    # hidden common-input factor: loads on the clean set only; NOT in Sigma_est
    a_hidden = np.zeros(N); a_hidden[idx["clean"]] = hidden_load

    Sigma_est  = A_obs @ A_obs.T + np.diag(d)
    Sigma_true = Sigma_est + np.outer(a_hidden, a_hidden)
    return dict(N=N, idx=idx, A_obs=A_obs, a_hidden=a_hidden,
                Sigma_est=Sigma_est, Sigma_true=Sigma_true,
                sig_true=np.sqrt(np.diag(Sigma_true)))


# ---- attacks ---------------------------------------------------------------
def realized_var(w, E, vS, s):
    return float(w @ E["Sigma_true"] @ w + s * (w @ vS) ** 2)

def factor_target(A_col, sig):
    v = np.zeros_like(sig); m = A_col > 1e-9; v[m] = sig[m]; return v

def hidden_target(a_hidden, sig):
    v = np.zeros_like(sig); m = a_hidden > 1e-9; v[m] = sig[m]; return v


# ---- allocations (optimised on Sigma_est) ----------------------------------
def simplex_min(fun, N):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    r = minimize(fun, np.full(N, 1/N), method="SLSQP", bounds=[(0, 1)]*N,
                 constraints=cons, options={"maxiter": 800, "ftol": 1e-12})
    return r.x

def alloc_flat(E): return np.full(E["N"], 1/E["N"])
def alloc_gmv(E):
    S = E["Sigma_est"]; return simplex_min(lambda w: w @ S @ w, E["N"])
def alloc_robust(E, s_design=3.0):
    S = E["Sigma_est"]; sig = np.sqrt(np.diag(E["Sigma_true"]))
    tg = [factor_target(E["A_obs"][:, f], sig) for f in range(E["A_obs"].shape[1])]
    def worst(w):
        base = w @ S @ w
        return max(base + s_design*(w @ v)**2 for v in tg)
    return simplex_min(worst, E["N"])


def worst_observable(w, E, s):
    sig = E["sig_true"]
    return max(realized_var(w, E, factor_target(E["A_obs"][:, f], sig), s)
               for f in range(E["A_obs"].shape[1]))

def spoof(w, E, s):
    return realized_var(w, E, hidden_target(E["a_hidden"], E["sig_true"]), s)

def worst_omniscient(w, E, k, s):
    sig = E["sig_true"]; order = np.argsort(-(w*sig)); S = order[:k]
    v = np.zeros_like(sig); v[S] = sig[S]
    return realized_var(w, E, v, s)


def main():
    E = build_ensemble()
    A = {"flat": alloc_flat(E), "gmv": alloc_gmv(E), "robust": alloc_robust(E)}
    print("pre-attack TRUE variance:",
          {k: round(float(w @ E["Sigma_true"] @ w), 3) for k, w in A.items()})
    print("weight on clean/hidden set:",
          {k: round(float(w[E["idx"]["clean"]].sum()), 3) for k, w in A.items()})

    ss = np.linspace(0, 8, 33)
    obs = {k: np.array([worst_observable(w, E, s) for s in ss]) for k, w in A.items()}
    spf = {k: np.array([spoof(w, E, s) for s in ss]) for k, w in A.items()}
    ks = np.arange(0, 15); s_fix = 4.0
    omni = {k: np.array([worst_omniscient(w, E, kk, s_fix) for kk in ks]) for k, w in A.items()}

    def crossing(y_gmv, y_flat):
        m = y_gmv > y_flat
        return ss[np.argmax(m)] if np.any(m) else np.nan
    cross_spoof = crossing(spf["gmv"], spf["flat"])

    # threshold sweep: crossover strength vs size of the UNMEASURED dependence
    hl = np.linspace(0.0, 0.55, 23); cross_vs_hidden = []
    for h in hl:
        Eh = build_ensemble(hidden_load=h)
        Ah = {"flat": alloc_flat(Eh), "gmv": alloc_gmv(Eh)}
        g = np.array([spoof(Ah["gmv"], Eh, s) for s in ss])
        f = np.array([spoof(Ah["flat"], Eh, s) for s in ss])
        m = g > f
        cross_vs_hidden.append(ss[np.argmax(m)] if np.any(m) else np.nan)
    cross_vs_hidden = np.array(cross_vs_hidden)

    col = {"flat": "#264653", "gmv": "#e76f51", "robust": "#2a9d8f"}
    plt.rcParams.update({"font.family": "monospace", "font.size": 8.5})
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))

    for k, y in obs.items(): ax[0].plot(ss, y, color=col[k], lw=2, label=k)
    ax[0].set_title("Structural: OBSERVABLE-factor move (defender sees it)")
    ax[0].set_xlabel("attack strength s"); ax[0].set_ylabel("realized error variance")
    ax[0].legend(fontsize=8)

    for k, y in spf.items(): ax[1].plot(ss, y, color=col[k], lw=2, label=k)
    if not np.isnan(cross_spoof):
        ax[1].axvline(cross_spoof, ls="--", c="0.5")
        ax[1].text(cross_spoof, ax[1].get_ylim()[1]*0.85,
                   f" gmv & robust\n exceed flat\n at s={cross_spoof:.1f}", fontsize=7, color="0.3")
    ax[1].set_title("HEADLINE  Structural: HIDDEN common-input (spoof)")
    ax[1].set_xlabel("spoof strength s"); ax[1].set_ylabel("realized error variance")
    ax[1].legend(fontsize=8)

    for k, y in omni.items(): ax[2].plot(ks, y, "-o", ms=3, color=col[k], lw=2, label=k)
    ax[2].set_title(f"Omniscient bound: worst k channels (s={s_fix})")
    ax[2].set_xlabel("channels compromised k"); ax[2].set_ylabel("realized error variance")
    ax[2].legend(fontsize=8)

    fig.suptitle("The variance-optimal allocation is fragile to the dependence it could not measure", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = os.path.join(OUT, "paper_x_echo_adversarial_fragility.png")
    fig.savefig(png, dpi=140)

    print("\n--- headline: spoof (hidden common input) ---")
    for k in A:
        print(f"  {k:6s}: error  s=0 -> {spf[k][0]:.2f} ,  s=8 -> {spf[k][-1]:.2f}")
    print(f"  gmv AND robust exceed flat at spoof strength s = {cross_spoof:.2f}"
          if not np.isnan(cross_spoof) else "  gmv never exceeds flat")
    print("\n--- control: observable-factor attack (defender sees it) ---")
    for k in A:
        print(f"  {k:6s}: worst-case  s=8 -> {obs[k][-1]:.2f}")
    print("\n--- omniscient bound at k=14 ---")
    for k in A:
        print(f"  {k:6s}: {omni[k][-1]:.2f}")
    finite = cross_vs_hidden[np.isfinite(cross_vs_hidden)]
    print(f"\nthreshold sweep: gmv becomes worse-than-flat once hidden load exceeds "
          f"~{hl[np.argmax(np.isfinite(cross_vs_hidden))]:.2f}"
          if finite.size else "no crossover in sweep")
    print("saved:", png)


if __name__ == "__main__":
    main()
