import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PARAMS = dict(
    n=0.120, alpha=1.339, beta=0.539,
    s=0.908, lam=2.700, mu=1.841,
    theta=0.196, kb=23.591,
    rhoB=0.154, dB=0.117,
    rhoT=0.546, betaT=0.766, dT=0.067, gamma=0.110,
    rhoE=0.073, eta=2.065, cE=0.489, ke=24.382, dE=0.059
)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

def deriv(x, p, s=None):
    U, B, T, E = x
    if s is None:
        s = p["s"]

    F = s * U / ((1 + p["lam"] * T) * (1 + p["mu"] * B))

    b_drive = sigmoid(p["kb"] * (F - p["theta"]))
    e_drive = sigmoid(
        p["ke"] * (p["alpha"] * U / (1 + p["eta"] * B) - p["cE"])
    )

    dU = p["n"] * (1 - U) - p["alpha"] * E * (1 - p["beta"] * B) * U
    dB = p["rhoB"] * b_drive - p["dB"] * B
    dT = (
        p["rhoT"] * E * (1 - p["betaT"] * B)
        - p["dT"] * T
        - p["gamma"] * B * T
    )
    dE = p["rhoE"] * e_drive - p["dE"] * E

    return np.array([dU, dB, dT, dE]), F

def simulate(x0, t_end=250, dt=0.02, shock=None, p=PARAMS):
    times = np.arange(0, t_end + dt, dt)
    x = np.array(x0, dtype=float)
    rows = []

    for t in times:
        s = p["s"] if shock is None else shock(t)
        dx, F = deriv(x, p, s=s)
        rows.append([t, *x, s, F])
        x = np.clip(x + dt * dx, 0, 1)

    return pd.DataFrame(
        rows, columns=["t", "U", "B", "T", "E", "s", "F"]
    )

def equilibrium_summary(df, tail=500):
    return df.tail(tail)[["U", "B", "T", "E", "F"]].mean()

def shock(t):
    return 1.60 if 60 <= t < 100 else PARAMS["s"]

# Two initial conditions under exactly the same external parameters.
open_run = simulate((0.20, 0.02, 0.95, 0.90))
closed_run = simulate((0.80, 0.90, 0.02, 0.05))
shock_run = simulate((0.20, 0.02, 0.95, 0.90), t_end=220, shock=shock)

print("Open attractor:")
print(equilibrium_summary(open_run).round(3))
print("\nClosed attractor:")
print(equilibrium_summary(closed_run).round(3))
print("\nShock experiment:")
print(shock_run.tail(500)[["B", "T", "E", "F"]].mean().round(3))
