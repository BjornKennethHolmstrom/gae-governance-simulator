"""
self_ii_appendix_a_correlation_tax.py

Reproduces every numerical claim in Self II, Appendix A
(The Self-Observation Ensemble and the Correlation Tax).

Flat repo: gae-governance-simulator/
Run: python self_ii_appendix_a_correlation_tax.py
"""

import numpy as np

rng = np.random.default_rng(20260616)


def N_eff(N, rho):
    "Kish design effect (A.2): effective number of independent observers."
    return N / (1.0 + (N - 1.0) * rho)


def mc_pooled_var(N, rho, sigma=1.0, trials=400_000):
    "Monte-Carlo variance of the equal-weight pooled estimator (A.2)."
    zc = rng.standard_normal(trials)
    zi = rng.standard_normal((trials, N))
    eps = (np.sqrt(rho) * zc[:, None] + np.sqrt(1 - rho) * zi) * sigma
    return eps.mean(axis=1).var()


def two_pathway_corr(rho_m, rho_d):
    "Closed-form compounded correlation across model and data pathways (A.4)."
    return 1 - (1 - rho_m) * (1 - rho_d)


def mc_two_pathway_corr(rho_m, rho_d, N=8, trials=400_000):
    "Monte-Carlo confirmation of the A.4 compounding identity."
    m = rng.standard_normal(trials)
    d = rng.standard_normal(trials)
    u = rng.standard_normal((trials, N))
    a, b, c = np.sqrt(rho_m), np.sqrt((1 - rho_m) * rho_d), np.sqrt((1 - rho_m) * (1 - rho_d))
    eps = a * m[:, None] + b * d[:, None] + c * u
    return np.corrcoef(eps[:, 0], eps[:, 1])[0, 1]


def neff_equal_blocks(block_sizes):
    "Equal-weight N_eff = inverse Herfindahl index of block sizes (A.5)."
    N = sum(block_sizes)
    return N ** 2 / sum(k * k for k in block_sizes)


def neff_opt_blocks(block_sizes, rho_w=1 - 1e-6):
    "Optimal-weight N_eff = block count B (A.5), via GLS: 1^T Sigma^{-1} 1."
    N = sum(block_sizes)
    Sig = np.zeros((N, N))
    i = 0
    for k in block_sizes:
        Sig[i:i + k, i:i + k] = rho_w
        i += k
    np.fill_diagonal(Sig, 1.0)
    one = np.ones(N)
    return one @ np.linalg.solve(Sig, one)


if __name__ == "__main__":
    print("A.2  Study-1 anchor   N_eff(6, 0.97) =", round(N_eff(6, 0.97), 4))
    print("A.2  saturation 1/rho  N_eff(1e6,0.5) =", round(N_eff(10 ** 6, 0.5), 4))
    print("A.2  MC vs closed form (N=6, rho=.97) =",
          round(mc_pooled_var(6, 0.97), 4), "vs", round((1 - 0.97) / 6 + 0.97, 4))
    print("A.4  compounded corr (.2,.8) closed   =", two_pathway_corr(0.2, 0.8),
          " MC =", round(mc_two_pathway_corr(0.2, 0.8), 4))
    print("A.5  block N_eff (equal, optimal):")
    for bs in ([5, 1], [4, 1, 1], [6], [1] * 6):
        print(f"        blocks {str(bs):<12} equal={neff_equal_blocks(bs):.3f}  opt(B)={neff_opt_blocks(bs):.3f}")
