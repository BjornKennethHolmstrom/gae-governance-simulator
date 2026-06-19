"""
Paper XI chain-model prototype v0.1
Minimal numerical check of the three formal claims of Part II:
  E1: minimum control energy grows ~exponentially with delegation depth
  E2: per-layer rank deficiencies accumulate; captured target fraction falls
  E3: noise injected late in the chain dominates delivered noise
Illustrative parameters. NOT the series-convention full simulation.
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

rng = np.random.default_rng(11)
m = 6            # directive/state dimension
SEEDS = 100

def rand_orth(n, r):
    M = r.standard_normal((n, n))
    Q, _ = np.linalg.qr(M)
    return Q

def layer(r, smin=0.7, smax=1.0):
    """Random contractive translation: U diag(s) V', s ~ U[smin,smax]."""
    s = r.uniform(smin, smax, m)
    return rand_orth(m, r) @ np.diag(s) @ rand_orth(m, r)

def gramian(A, Beff):
    return solve_discrete_lyapunov(A, Beff @ Beff.T)

# ---------------- E1: energy vs depth ----------------
depths = range(0, 8)
ratios = {n: [] for n in depths}
for seed in range(SEEDS):
    r = np.random.default_rng(seed)
    A = np.diag(r.uniform(0.85, 0.98, m))
    layers = [layer(r) for _ in range(7)]
    xf = r.standard_normal(m); xf /= np.linalg.norm(xf)
    E0 = None
    for n in depths:
        Pi = np.eye(m)
        for i in range(n):
            Pi = layers[i] @ Pi
        W = gramian(A, Pi)
        E = float(xf @ np.linalg.pinv(W, rcond=1e-12) @ xf)
        if n == 0:
            E0 = E
        ratios[n].append(E / E0)

print("E1: minimum-energy ratio E_min(n)/E_min(0), median [IQR] over", SEEDS, "seeds")
prev = None
for n in depths:
    a = np.array(ratios[n])
    med = np.median(a); q1, q3 = np.percentile(a, [25, 75])
    growth = "" if prev is None else f"  x{med/prev:.2f} vs depth {n-1}"
    print(f"  depth {n}: median {med:10.2f}  [{q1:8.2f}, {q3:10.2f}]{growth}")
    prev = med

# geometric-mean singular value of U[0.7,1.0] -> predicted per-layer factor
gln = -( (1*np.log(1)-1) - (0.7*np.log(0.7)-0.7) ) / 0.3
print(f"  predicted per-layer median factor ~ exp(2*{gln:.4f}) = {np.exp(2*gln):.3f}")

# ---------------- E2: rank decay under three blind-spot geometries ----------------
print("\nE2: composite rank, clean-transmission dimensions (s>=0.99), min nonzero s")
print("    three kernel geometries, d_i = 1 per layer, m = 6")
for regime in ("orthogonal", "random", "identical"):
    rk_mean = np.zeros(m+1); clean_mean = np.zeros(m+1); mins_mean = np.zeros(m+1)
    for seed in range(SEEDS):
        r = np.random.default_rng(1000+seed)
        if regime == "orthogonal":
            V = rand_orth(m, r)          # columns: mutually orthogonal kernels
            kerns = [V[:, i] for i in range(m)]
        elif regime == "identical":
            v = r.standard_normal(m); v /= np.linalg.norm(v)
            kerns = [v]*m
        else:
            kerns = []
            for _ in range(m):
                v = r.standard_normal(m); v /= np.linalg.norm(v)
                kerns.append(v)
        Pi = np.eye(m)
        for n in range(0, m+1):
            if n > 0:
                v = kerns[n-1]
                Pi = (np.eye(m) - np.outer(v, v)) @ Pi
            s = np.linalg.svd(Pi, compute_uv=False)
            rk = int((s > 1e-10).sum())
            rk_mean[n] += rk
            clean_mean[n] += int((s >= 0.99).sum())
            nz = s[s > 1e-10]
            mins_mean[n] += (nz.min() if len(nz) else 0.0)
    rk_mean /= SEEDS; clean_mean /= SEEDS; mins_mean /= SEEDS
    print(f"  {regime:>10} kernels:")
    for n in range(0, m+1):
        print(f"    depth {n}: rank {rk_mean[n]:.2f}  clean dims {clean_mean[n]:.2f}  min nonzero s {mins_mean[n]:.3f}")

# ---------------- E3: delivered-noise share by injection layer ----------------
print("\nE3: share of delivered noise variance by injection layer (n = 6)")
shares = np.zeros((SEEDS, 6))
for seed in range(SEEDS):
    r = np.random.default_rng(2000+seed)
    layers6 = [layer(r) for _ in range(6)]
    var = []
    for k in range(1, 7):           # noise injected at layer k
        T = np.eye(m)
        for j in range(k, 6):        # passes through layers k+1..6 (index k..5)
            T = layers6[j] @ T
        var.append(np.trace(T @ T.T))
    var = np.array(var)
    shares[seed] = var / var.sum()
mean_shares = shares.mean(axis=0)
for k in range(6):
    print(f"  layer {k+1}: {mean_shares[k]*100:5.1f} %")
print(f"  last two layers carry {100*(mean_shares[-2:].sum()):.1f} % of delivered noise variance")

# ---------------- E4: depth comparison table ----------------
print("\nE4: median energy ratio and delivered signal gain by architecture depth")
for n in (2, 4, 7):
    a = np.array(ratios[n]); med = np.median(a)
    # delivered signal power for unit directive ~ E ||Pi v||^2, estimate over seeds
    gains = []
    for seed in range(SEEDS):
        r = np.random.default_rng(seed)
        _ = np.diag(r.uniform(0.85, 0.98, m))
        Ls = [layer(r) for _ in range(7)]
        Pi = np.eye(m)
        for i in range(n):
            Pi = Ls[i] @ Pi
        v = r.standard_normal(m); v /= np.linalg.norm(v)
        gains.append(np.linalg.norm(Pi @ v)**2)
    print(f"  depth {n}: median E_min ratio {med:9.2f}   mean delivered signal power {np.mean(gains):.3f}")
