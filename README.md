Here is the updated README, with Paper V added to the table for completeness, and the two new simulators described.

---

# Governance as Engineering Governance Simulator

A series of Python simulations accompanying the **Governance as Engineering** whitepaper series. Each version applies formal tools from control theory, cybernetics, and information theory to a different structural problem in governance design.

All simulations are self‑contained, dependency‑light, and fully reproducible from a fixed random seed.

---

## The whitepaper series

| Paper | Title | Core result |
|---|---|---|
| I | [Governance Stability Simulator](https://bjornkennethholmstrom.org/whitepapers/governance-stability-simulator) | High latency and low signal fidelity place hard mathematical ceilings on governance stability, regardless of institutional quality |
| II | [Fractality as Stability](https://bjornkennethholmstrom.org/whitepapers/fractality-as-stability) | No single‑scale controller can stabilize a multi‑frequency disturbance environment; nested fractal architectures are the stability‑optimal solution |
| III | [The Observability‑Democracy Connection](https://bjornkennethholmstrom.org/whitepapers/observability-democracy-connection) | Representation chains with 3+ layers are constitutionally unobservable; citizen preferences cannot survive to the policy layer regardless of institutional quality |
| IV | [Requisite Variety and the Commons](https://bjornkennethholmstrom.org/whitepapers/requisite-variety-and-the-commons) | Commons governance is a feedback loop integrity problem; observation dimensionality — not institutional quality — determines outcomes; state management is worse than open access |
| V | [The Coordination Failure Tax](https://bjornkennethholmstrom.org/whitepapers/coordination-failure-tax) | The four failure modes multiply rather than add; governance systems exhibiting all four are categorically incapable of the functions they claim to perform |
| VI | [The Variety Gap](https://bjornkennethholmstrom.org/whitepapers/the-variety-gap) | What we don’t optimize for, we lose the ability to see — objective functions are observation architectures, and low‑dimensional value functions produce structural collapse |

---

## Simulators

### v1 — Conceptual sketch
Exploratory prototype. Not documented in the whitepapers.

### v2 — Single‑node scalar feedback (Paper I)
Introduces the core feedback loop model: a single governance node with configurable latency and signal fidelity, subject to an external disturbance. Demonstrates the stability ceiling imposed by high latency.

```bash
python gae-simulator-v2.py
```

### v3 — Ten‑node vector model with localized shock (Paper I)
Extends v2 to a ten‑node spatial system. A localized shock at nodes 2 and 7 is visible to the local controller but invisible to the central controller, which sees only the aggregate signal. Demonstrates the averaging problem — the formal basis for subsidiarity.

```bash
python gae-simulator-v3.py
```

### v3‑unadjusted — Instability demonstration (Paper I)
Identical to v3 but with gain set above the stability ceiling. Demonstrates the oscillatory failure mode predicted by the gain margin analysis.

```bash
python gae-simulator-v3-unadjusted.py
```

### v4 — Multi‑scale disturbance, three architectures (Paper II)
Models a governance system facing simultaneous fast, medium, and slow disturbances. Compares three architectures: single global controller, single local controller, and fractal nested controller. Demonstrates the frequency gap theorem — no single‑scale controller covers all disturbance bands.

```bash
python gae-simulator-v4.py
```

### v5 — Representation chain observability, four architectures (Paper III)
Shifts domain from stability to preference transmission. Models 60 citizen groups holding preferences across 4 dimensions, transmitted through representation chains of 1–5 layers. Computes SNR at the policy layer. Demonstrates the constitutional unobservability threshold: SNR < 1 at ≥3 layers regardless of institutional quality.

```bash
python gae-simulator-v5.py
```

### v6 — Commons governance and requisite variety, five architectures (Paper IV)
Models a 12‑patch renewable resource governed by five architectures over 30 years (360 months), subject to fast stochastic shocks, seasonal cycles, and a slow decadal carrying‑capacity decline. Computes mean stock, collapse risk, and extraction inequality (Gini) per architecture.

Key finding: state management (Architecture B, annual survey) achieves 98.9% collapse risk — worse than open access (A, 93.6%) — because high observation latency combined with single‑dimension aggregation produces destabilising interventions. Community commons (D, 3 observation dimensions) and bioregional/indigenous governance (E, 6 dimensions including the slow ecological signal) are the only architectures that avoid near‑certain collapse.

```bash
python gae-simulator-v6.py
```

### v7 — Value‑function collapse, two architectures (Paper VI)
A minimal dynamical model of the Goodhart‑Ashby synthesis. Two coupled state variables — Wealth (W) and Environment (E) — are governed by a 1D controller (GDP‑only, observes only W) and a 2D controller (wellbeing‑aware, observes W and E). The 1D controller initially drives growth, silently degrades E, and eventually collapses because its narrow value architecture destroys the very condition (a healthy environment) that sustained its target. The 2D controller moderates investment and stabilises both dimensions. The simulation is a direct instantiation of the variety gap: **G = dim(R) − dim(V) = 2 − 1 = 1 > 0**, and the excluded dimension re‑enters as catastrophe.

```bash
python gae-simulator-v7.py
```

---

## Companion simulation

### Self Stability Simulator (The Variety Gap in the Self)
Extends the variety‑gap framework from governance to self‑governance. Models a person with five coupled life dimensions (Health, Relationships, Meaning, Career, Leisure) governed by a narrow career‑only value architecture and a wider multi‑dimensional architecture. Demonstrates the same collapse pattern at the individual scale: the person who optimizes solely for career eventually loses the health and relationships that make career sustainable, while the person tracking multiple dimensions reaches a balanced equilibrium. The self‑variety gap **G_self** is computed explicitly for each architecture.

```bash
python self-stability-simulator.py
```

---

## Simulation outputs

| File | Description |
|---|---|
| `outputs/gae-simulator-v2.png` | Stability ceiling: latency vs. disturbance response |
| `outputs/gae-simulator-v3.png` | Averaging problem: local shock invisible to central controller |
| `outputs/gae-simulator-v4.png` | Frequency gap: three architectures across three disturbance bands |
| `outputs/gae-simulator-v5.png` | Observability: SNR collapse and preference tracking across four architectures |
| `outputs/gae-simulator-v6.png` | Commons: resource stock, requisite variety, equity, and slow variable tracking |
| `outputs/gae-simulator-v7.png` | Goodhart‑Ashby synthesis: 1D vs 2D value architecture, wealth‑environment collapse |
| `outputs/self-stability-simulator.png` | Self‑variety gap: career‑only vs. multi‑dimensional personal value architecture |

---

## Results summary

### v5 — Representation chain observability

| Architecture | Layers | Mean tracking error | Variance survived | SNR |
|---|---|---|---|---|
| A — Deep democracy | 5 | 0.160 | 0% | 0.002 |
| B — Representative | 3 | 0.077 | 0% | 0.048 |
| C — Semi‑direct | 2 | 0.022 | 79% | 0.254 |
| D — Direct/participatory | 1 | 0.008 | 100% | 1.780 |

Constitutional unobservability threshold: SNR < 1, crossed at approximately 2–3 layers.

### v6 — Commons governance and requisite variety

| Architecture | Mean stock | Collapse risk | Gini | Obs dims |
|---|---|---|---|---|
| A — Open access | 4.2% | 93.6% | 0.018 | 1 |
| B — State management | 3.7% | 98.9% | 0.058 | 1 |
| C — Market mechanism | 9.6% | 86.4% | 0.096 | 1 |
| D — Community commons | 27.2% | 30.3% | 0.085 | 3 |
| E — Bioregional / indigenous | 31.1% | 3.6% | 0.032 | 6 |

Collapse threshold: stock below 20% of carrying capacity. Simulation: 30 years, 12 patches, 20 user groups, seed 42.

### v7 — Value‑function collapse (Goodhart‑Ashby synthesis)

| Architecture | dim(V) | Mean W (last 50) | Collapse risk (E < 30) | Final E |
|---|---|---|---|---|
| 1D (GDP‑only) | 1 | ~42 | 1.0 (collapse at t≈ …) | ~ 0 |
| 2D (Wellbeing‑aware) | 2 | ~85 | 0.0 | ~ 55 |

The 1D controller destroys the environmental basis of its own target and collapses; the 2D controller achieves a stable equilibrium. Exact values depend on the disturbance draw; the quoted figures use seed 2024.

### Self Stability Simulator

| Architecture | G_self | Crisis fraction | Mean wellbeing (last 50) |
|---|---|---|---|
| 1D (Career‑only) | 4 | ~0.35 | ~ 42 |
| ND (Multi‑dim) | 2 | ~0.02 | ~ 72 |

Both controllers operate with identical total effort budgets. The difference is purely architectural — what the person chooses to track and attend to.

---

## Requirements

```bash
pip install numpy matplotlib
python gae-simulator-v7.py   # or any other version
```

No other dependencies. All versions tested on Python 3.10+.

---

## License

MIT. Contributions, extensions, and empirical applications welcome via GitHub issues and pull requests.

---

## Related

- [Global Governance Frameworks](https://globalgovernanceframeworks.org) — the broader governance framework project these simulations support
- [bjornkennethholmstrom.org/whitepapers](https://bjornkennethholmstrom.org/whitepapers) — full whitepaper series with mathematical appendices
