"""
study_i_observer_correlation.py  --  v0.3 (frozen with protocol S1-0.3)
Study 1: observer correlation in an AI ensemble (tests Paper X).
Analysis space: log errors e_mi = ln(estimate/truth) (logit for rates).
Ensemble = geometric mean (arithmetic mean in log space).
All second moments UNCENTERED: shared bias counts as correlated error.
Primary estimand: rho_eff = 2*sum_{i<j} S_ij / ((N-1)*trace(S)),  S = E'E/n.
Identity (exact): MSE_ens/mean(MSE_m) = (1-rho_eff)/N + rho_eff.
Data CSV: item_id,item_class,truth,model,estimate  (orders/clamps upstream rules in protocol §4-5)
"""
import numpy as np
from scipy.stats import norm

B_BOOT = 10_000
LOG_CAP = np.log(100.0)        # clamp: two orders of magnitude
RATE_LO, RATE_HI = 0.005, 0.995

def clamp_log_error(est, truth):
    if est <= 0:
        return -LOG_CAP, True
    e = np.log(est / truth)
    return (np.clip(e, -LOG_CAP, LOG_CAP), abs(e) > LOG_CAP)

def clamp_logit_error(est, truth):
    p = np.clip(est, RATE_LO, RATE_HI)
    logit = lambda x: np.log(x / (1 - x))
    return logit(p) - logit(truth), (est < RATE_LO or est > RATE_HI)

def rho_eff(E):
    S = (E.T @ E) / E.shape[0]
    N = S.shape[0]
    return (S.sum() - np.trace(S)) / ((N - 1) * np.trace(S))

def rhat(E):
    return np.mean(E.mean(1) ** 2) / np.mean(np.mean(E ** 2, 0))

def std_pairwise_mean(E):
    Z = E / E.std(0, ddof=1)
    S = (Z.T @ Z) / E.shape[0]
    N = S.shape[0]
    return (S.sum() - np.trace(S)) / ((N - 1) * np.trace(S))

def bca_ci(E, fn, B=B_BOOT, alpha=0.05, seed=0):
    r = np.random.default_rng(seed); n = E.shape[0]
    th = fn(E)
    boots = np.array([fn(E[r.integers(0, n, n)]) for _ in range(B)])
    z0 = norm.ppf(np.clip((boots < th).mean(), 1e-4, 1 - 1e-4))
    jack = np.array([fn(np.delete(E, i, axis=0)) for i in range(n)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum(); den = 6 * (((jm - jack) ** 2).sum()) ** 1.5
    a = num / den if den > 0 else 0.0
    q = lambda z: norm.cdf(z0 + (z0 + norm.ppf(z)) / (1 - a * (z0 + norm.ppf(z))))
    return np.quantile(boots, [q(alpha / 2), q(1 - alpha / 2)])

def calibrated_ci(E, fn, target=0.95, B=2000, reps=300, seed=0):
    """Parametric calibration: MVN with the SAMPLE uncentered covariance;
    widen nominal level until empirical coverage >= target at those parameters."""
    r = np.random.default_rng(seed); n, N = E.shape
    S = (E.T @ E) / n
    true_val = (S.sum() - np.trace(S)) / ((N - 1) * np.trace(S))
    L = np.linalg.cholesky(S + 1e-10 * np.eye(N))
    for nominal in (0.95, 0.97, 0.99, 0.995):
        cover = 0
        for k in range(reps):
            Esim = r.standard_normal((n, N)) @ L.T
            lo, hi = bca_ci(Esim, fn, B=400, alpha=1 - nominal, seed=k)
            cover += (lo <= true_val <= hi)
        if cover / reps >= target:
            break
    return bca_ci(E, fn, B=B_BOOT, alpha=1 - nominal, seed=seed + 1), nominal

def p2_test(E_tail, E_central, B=B_BOOT, seed=0):
    r = np.random.default_rng(seed)
    obs = std_pairwise_mean(E_tail) - std_pairwise_mean(E_central)
    pooled = np.vstack([E_tail, E_central]); nt = E_tail.shape[0]
    cnt = sum(
        std_pairwise_mean(pooled[p[:nt]]) - std_pairwise_mean(pooled[p[nt:]]) >= obs
        for p in (r.permutation(pooled.shape[0]) for _ in range(B))
    )
    sec = rho_eff(E_tail) - rho_eff(E_central)
    return obs, (cnt + 1) / (B + 1), sec

def model_orders(n_items=50, n_models=6, master_seed=2026):
    """Per-model seeded item orders, published at battery freeze (protocol §4)."""
    return {m: list(np.random.default_rng(master_seed + m).permutation(n_items))
            for m in range(n_models)}

if __name__ == "__main__":
    # smoke test: identity + recovery
    r = np.random.default_rng(1)
    sig = np.linspace(0.5, 2.0, 6)
    E = (np.sqrt(0.4) * r.standard_normal((50, 1))
         + np.sqrt(0.6) * r.standard_normal((50, 6))) * sig[None, :]
    lhs, re = rhat(E), rho_eff(E)
    assert abs(lhs - ((1 - re) / 6 + re)) < 1e-12, "identity broken"
    print(f"identity OK; rho_eff={re:.3f}; std-pairwise={std_pairwise_mean(E):.3f}")
    print("BCa 95%:", np.round(bca_ci(E, rho_eff, B=1000), 3))
    print("orders sample:", model_orders()[0][:8])
