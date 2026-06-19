"""
self_ii_appendix_b_actuation_chain.py

Reproduces the numerical claims in Self II, Appendix B
(The Actuation Chain and the Energy Law).

Flat repo: gae-governance-simulator/
Run: python self_ii_appendix_b_actuation_chain.py
"""

import numpy as np


def cascade(D, a=0.5, c=0.7, b=1.0):
    "Depth-D actuation cascade: lower-bidiagonal A (diag a, subdiag c), input at layer 1 (B.1)."
    A = np.zeros((D, D))
    np.fill_diagonal(A, a)
    for i in range(1, D):
        A[i, i - 1] = c
    B = np.zeros((D, 1))
    B[0, 0] = b
    return A, B


def min_energy(A, B, x_t, T):
    "Minimum control energy to reach x_t in T steps; inf if x_t outside reachable set (B.2, B.4)."
    cols, M = [], B.copy()
    for _ in range(T):
        cols.append(M.copy())
        M = A @ M
    R = np.hstack(cols)                       # columns A^k B, k = 0..T-1
    z, *_ = np.linalg.lstsq(R, x_t, rcond=None)
    if np.linalg.norm(R @ z - x_t) > 1e-8:    # unreachable in horizon T
        return np.inf
    W = R @ R.T                               # finite-horizon controllability Gramian
    return float(x_t @ np.linalg.pinv(W, rcond=1e-15) @ x_t)


if __name__ == "__main__":
    print("B.4  Unreachability threshold (horizon T < depth D => J = inf):")
    for D in (3, 5, 7):
        A, B = cascade(D)
        eD = np.eye(D)[-1]
        for T in (D - 1, D, D + 2):
            J = min_energy(A, B, eD, T)
            print(f"      D={D} T={T:<2}: {'UNREACHABLE' if np.isinf(J) else f'J_min={J:.4g}'}")

    print("B.3  Effort law vs depth (generous horizon T=2D), superlinear-to-geometric:")
    for c in (0.7, 0.5):
        Js = [min_energy(*cascade(D, c=c), np.eye(D)[-1], 2 * D) for D in range(2, 9)]
        print(f"      c={c}: " + ", ".join(f"{j:.3g}" for j in Js))

    print("B.5  Pressman-Wildavsky scalar shadow (joint success = p^n):")
    for p, n in [(0.99, 70), (0.95, 10), (0.9, 10)]:
        print(f"      p={p} n={n}: {p**n:.3f}")

    print("B.6  Legitimacy coupling (path gain (L*alpha)^D, D=8, alpha=0.8):")
    for L in (1.0, 0.8, 0.5):
        g = (L * 0.8) ** 8
        print(f"      L={L}: gain={g:.5e}  energy x{((0.8**8)/g)**2:.1f} vs full trust")
