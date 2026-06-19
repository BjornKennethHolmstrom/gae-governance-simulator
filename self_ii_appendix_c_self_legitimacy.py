"""
self_ii_appendix_c_self_legitimacy.py

Reproduces the numerical claims in Self II, Appendix C
(Self-Legitimacy Dynamics): the existence bifurcation, hysteresis,
built-vs-borrowed, and the transparency trap.

Flat repo: gae-governance-simulator/
Run: python self_ii_appendix_c_self_legitimacy.py
"""

import numpy as np


def sigma(z):
    return 1.0 / (1.0 + np.exp(-z))


def p_deliver(L, k=8.0, L_half=0.5):
    "Delivery probability rises through L_half; lower L_half = higher competence (C.2)."
    return sigma(k * (L - L_half))


def EdL(L, alpha, gamma, **kw):
    "Expected drift of self-legitimacy (C.2): build with gain alpha, erode with gain gamma."
    p = p_deliver(L, **kw)
    return p * alpha * (1 - L) - (1 - p) * gamma * L


def fixed_points(alpha, gamma, **kw):
    "Fixed points of the legitimacy map and their stability (C.3, C.5)."
    Ls = np.linspace(1e-4, 1 - 1e-4, 400001)
    f = EdL(Ls, alpha, gamma, **kw)
    out = []
    for i in np.where(np.diff(np.sign(f)) != 0)[0]:
        r = Ls[i] - f[i] * (Ls[i + 1] - Ls[i]) / (f[i + 1] - f[i])
        dr = (EdL(r + 1e-4, alpha, gamma, **kw) - EdL(r - 1e-4, alpha, gamma, **kw)) / 2e-4
        out.append((round(r, 3), "stable" if dr < 0 else "unstable"))
    return out


def hysteresis_steps(alpha, gamma, up):
    "Best-case steps to traverse L: 0.3 <-> 0.7 (C.4)."
    L = 0.3 if up else 0.7
    goal = 0.7 if up else 0.3
    n = 0
    while (L < goal if up else L > goal) and n < 10 ** 6:
        L = L + alpha * (1 - L) if up else L - gamma * L
        n += 1
    return n


def shock_trajectory(L0, alpha, gamma, L_half, shock_at=15, shock=0.25, T=120, seed=1):
    "Built-vs-borrowed: response to an identical betrayal shock (C.5)."
    r = np.random.default_rng(seed)
    L = L0
    for t in range(T):
        if t == shock_at:
            L = max(L - shock, 1e-4)
        else:
            L = L + alpha * (1 - L) if r.random() < p_deliver(L, L_half=L_half) else max(L - gamma * L, 1e-4)
    return round(L, 3)


def trap(deceive, alpha=0.05, gamma=0.15, k=8.0, margin=0.15, T=140, reckon=90, seed=3):
    "Transparency trap: commit on perceived P, deliver on true L; deception edits P (C.6)."
    r = np.random.default_rng(seed)
    L = P = 0.70
    gap = 0.0
    betr = 0
    for t in range(T):
        s = max(P - margin, 0.0)
        if r.random() < sigma(k * (L - s)):
            L += alpha * (1 - L); P += alpha * (1 - P)
        else:
            betr += 1
            L = max(L - gamma * L, 1e-4)
            P = P if deceive else max(P - gamma * P, 1e-4)
        gap = max(gap, P - L)
        if t == reckon and deceive:
            P = L
    return round(L, 3), round(P, 3), round(gap, 3), betr


if __name__ == "__main__":
    print("C.3  Existence bifurcation:")
    for Lh, lab in [(0.3, "high competence"), (0.5, "mid competence ")]:
        for g in (0.10, 0.20, 0.30):
            healthy = [r for r, s in fixed_points(0.05, g, L_half=Lh) if s == "stable" and r > 0.4]
            sep = [r for r, s in fixed_points(0.05, g, L_half=Lh) if s == "unstable"]
            msg = f"healthy FP {healthy}, separatrix {sep}" if healthy else "COLLAPSE-ONLY"
            print(f"      {lab} gamma/alpha={g/0.05:>2.0f}: {msg}")

    print("C.4  Hysteresis (recovery/decline step ratio ~ gamma/alpha):")
    for a, g in [(0.05, 0.30), (0.05, 0.15)]:
        nu, nd = hysteresis_steps(a, g, True), hysteresis_steps(a, g, False)
        print(f"      gamma/alpha={g/a:.0f}: climb={nu}, fall={nd}, ratio={nu/nd:.1f}")

    print("C.5  Built vs borrowed (same L0=0.90, same shock):")
    print(f"      built   (L_half=0.40): final L = {shock_trajectory(0.90, 0.05, 0.15, 0.40)}")
    print(f"      borrowed(L_half=0.80): final L = {shock_trajectory(0.90, 0.05, 0.15, 0.80)}")

    print("C.6  Transparency trap (true L, perceived P, max gap, betrayals):")
    print("      honest        :", trap(False))
    print("      self-deceiving:", trap(True))
