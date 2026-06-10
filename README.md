# Governance as Engineering Governance Simulator

A series of Python simulations accompanying the **Governance as Engineering** working paper series. Each version applies formal tools from control theory, cybernetics, and information theory to a different structural problem in governance design.

All simulations are self‑contained, dependency‑light, and fully reproducible from fixed random seeds. From v8 onward, every simulation adds a Monte Carlo layer (50–200 seeds, reported as median trajectories with 10th–90th percentile bands) and at least one parameter‑sweep map demonstrating that the qualitative results are robust across regions of parameter space rather than artefacts of a single configuration.

---

## The whitepaper series

| Paper | Title | Core result |
|---|---|---|
| I | [Governance Stability Simulator](https://bjornkennethholmstrom.org/whitepapers/governance-stability-simulator) | High latency and low signal fidelity place hard mathematical ceilings on governance stability, regardless of institutional quality |
| II | [Fractality as Stability](https://bjornkennethholmstrom.org/whitepapers/fractality-as-stability) | No single‑scale controller can stabilize a multi‑frequency disturbance environment; nested fractal architectures are the stability‑optimal solution |
| III | [The Observability‑Democracy Connection](https://bjornkennethholmstrom.org/whitepapers/observability-democracy-connection) | Representation chains with 3+ layers are constitutionally unobservable; citizen preferences cannot survive to the policy layer regardless of institutional quality |
| IV | [Requisite Variety and the Commons](https://bjornkennethholmstrom.org/whitepapers/requisite-variety-and-the-commons) | Commons governance is a feedback loop integrity problem; observation dimensionality — not institutional quality — determines outcomes; state management is worse than open access |
| V | [The Coordination Failure Tax](https://bjornkennethholmstrom.org/whitepapers/coordination-failure-tax) | The four failure modes multiply rather than add; governance systems exhibiting all four are categorically incapable of the functions they claim to perform |
| VI | [The Variety Gap](https://bjornkennethholmstrom.org/whitepapers/the-variety-gap) | What we don't optimize for, we lose the ability to see — objective functions are observation architectures, and low‑dimensional value functions produce structural collapse |
| VII | [The Architecture of Governance Failure](https://www.bjornkennethholmstrom.org/working-papers/architecture-of-governance-failure) | Reform disappoints for structural reasons: the immune system, the bypass trap, and the legibility problem are features of the transition landscape, not contingent failures — a qualitative synthesis of fifteen country studies |
| VIII | [Measuring the Variety Gap](https://www.bjornkennethholmstrom.org/working-papers/measuring-the-variety-gap) | The variety gap is measurable: a composite index estimated from observable proxies, with epistemic tiering, across twenty country and organizational cases |
| IX | [The Political Economy of Requisite Governance](https://www.bjornkennethholmstrom.org/working-papers/political-economy-of-requisite-governance)* | Transition bandwidth is the ninth structural primitive; latency asymmetry favours embedded incumbents; systems lose the capacity for self‑redesign strictly before operational collapse |
| X | [Requisite Observer Diversity](https://www.bjornkennethholmstrom.org/working-papers/requisite-observer-diversity) | Observer diversity is the tenth structural primitive; N_eff = N/(1+(N−1)ρ) — a consolidated observer ensemble retains N nominal observers but the protection of one, and its blind spots are invisible to every instrument it possesses |

---

## Simulators

### v1 — Conceptual sketch
Exploratory prototype. Not documented in the working papers.

### v2 — Single‑node scalar feedback (Paper I)
Introduces the core feedback loop model: a single governance node with configurable latency and signal fidelity, subject to an external disturbance. Demonstrates the stability ceiling imposed by high latency.

```bash
python gae-simulator-v2.py
```

### v3 — Ten‑node vector model with localized shock (Paper I)
Extends v2 to a ten‑node spatial system. A localized shock at nodes 2 and 7 is visible to the local controller but invisible to the central controller, which sees only the aggregate signal. Demonstrates the averaging problem — the formal basis for subsidiarity.

```bash
python gae-simulator-v3.py
```

### v3‑unadjusted — Instability demonstration (Paper I)
Identical to v3 but with gain set above the stability ceiling. Demonstrates the oscillatory failure mode predicted by the gain margin analysis.

```bash
python gae-simulator-v3-unadjusted.py
```

### v4 — Multi‑scale disturbance, three architectures (Paper II)
Models a governance system facing simultaneous fast, medium, and slow disturbances. Compares three architectures: single global controller, single local controller, and fractal nested controller. Demonstrates the frequency gap theorem — no single‑scale controller covers all disturbance bands.

```bash
python gae-simulator-v4.py
```

### v5 — Representation chain observability, four architectures (Paper III)
Shifts domain from stability to preference transmission. Models 60 citizen groups holding preferences across 4 dimensions, transmitted through representation chains of 1–5 layers. Computes SNR at the policy layer. Demonstrates the constitutional unobservability threshold: SNR < 1 at ≥3 layers regardless of institutional quality.

```bash
python gae-simulator-v5.py
```

### v6 — Commons governance and requisite variety, five architectures (Paper IV)
Models a 12‑patch renewable resource governed by five architectures over 30 years (360 months), subject to fast stochastic shocks, seasonal cycles, and a slow decadal carrying‑capacity decline. Computes mean stock, collapse risk, and extraction inequality (Gini) per architecture.

Key finding: state management (Architecture B, annual survey) achieves 98.9% collapse risk — worse than open access (A, 93.6%) — because high observation latency combined with single‑dimension aggregation produces destabilising interventions. Community commons (D, 3 observation dimensions) and bioregional/indigenous governance (E, 6 dimensions including the slow ecological signal) are the only architectures that avoid near‑certain collapse.

```bash
python gae-simulator-v6.py
```

### v7 — Value‑function collapse, two architectures (Paper VI)
A minimal dynamical model of the Goodhart‑Ashby synthesis. Two coupled state variables — Wealth (W) and Environment (E) — are governed by a 1D controller (GDP‑only, observes only W) and a 2D controller (wellbeing‑aware, observes W and E). The 1D controller initially drives growth, silently degrades E, and eventually collapses because its narrow value architecture destroys the very condition (a healthy environment) that sustained its target. The 2D controller moderates investment and stabilises both dimensions. The simulation is a direct instantiation of the variety gap: **G = dim(R) − dim(V) = 2 − 1 = 1 > 0**, and the excluded dimension re‑enters as catastrophe.

```bash
python gae-simulator-v7.py
```

### v8 — Bypass‑trap dynamics, three scenarios (Paper IX)
Models the bypass trap: a dysfunctional governance substrate (dysfunction D) and a parallel bypass institution (load share B) that absorbs functions the substrate fails to perform. Reform pressure responds only to *visible* dysfunction D·(1−B), so the bypass hides the problem it routes around — while the substrate caps the bypass's own performance ceiling. Three scenarios: no bypass, permanent bypass, and a sunset‑coupled bypass that dismantles itself once a demonstrated‑performance threshold is crossed, returning load and reform pressure to the substrate.

Key findings: the permanent bypass creates a stable low‑performance attractor — the system *looks* healthiest (highest perceived performance) while actual performance is capped lowest. A dedicated deception‑gap panel (perceived − actual performance) shows the trap's self‑concealing signature: persistent positive gap under permanent bypass, zero under no bypass, transient under sunset coupling. The sunset variant escapes the attractor and outperforms even the no‑bypass baseline — conditional on the modelling assumption that deferred reform pressure is productively redirected at sunset. Monte Carlo: 75 seeds; sweep: drift rate × reform rate, showing the sunset coupling extends the feasible‑reform region.

```bash
python gae-simulator-v8-bypass-trap.py
```

### v9 — Reform‑incumbent latency asymmetry (Paper IX)
Two controllers act on the same ten‑dimensional architecture state: a reform coalition (observes 8 dimensions, latency τ_R) pushing toward the target, and an incumbent (observes d_I dimensions, latency τ_I ≤ τ_R) pulling back. The latency advantage is modelled as an effective counter‑mobilisation gain boost on contested dimensions — an explicit tier‑2 assumption, since in a bounded push‑pull system pure observation delay cannot shift the equilibrium.

Key findings: reform wins unopposed on dimensions the incumbent does not monitor (the structural "reform floor"); success declines as incumbent coverage d_I rises and as the latency ratio falls. The full τ_ratio × d_I phase diagram shows a **gradient boundary, not a sharp threshold** — the simulation that disciplines Paper IX's dim(T) ≥ dim(I) condition down to a heuristic rather than a law. Monte Carlo: 50 seeds; sweep: 12 × 10 grid with classified outcomes (success / contested / absorbed).

```bash
python gae-simulator-v9-latency-asymmetry.py
```

### v10 — Transition bandwidth race, three parameterisations (Paper IX)
Couples the variety gap G(t) to the reform capacity R(t) that closes it: a growing gap consumes the very capacity needed to close it, through crisis crowd‑out and incumbent capture. Compares a high‑bandwidth federation, a bypass‑heavy system, and a locked regime, under constant and accelerating (AI‑scenario) environmental demand.

Key finding — the **two‑threshold structure**: the locked regime enters a *transition‑bandwidth trap* (R → 0 at G ≈ 0.61, far below the operational collapse threshold G_crit = 1.8) roughly a hundred steps before operational collapse. The system still functions but can no longer redesign itself — the paper's point of no return, reached strictly before the crisis is visible. Accelerating demand compresses the window between trap and collapse, and breaks the bypass‑heavy system that survives constant demand. Monte Carlo: 60 seeds; sweep: regeneration rate × capture rate with three‑way outcome classification (reformed / trapped / collapsed).

```bash
python gae-simulator-v10-bandwidth-race.py
```

### v11 — Epistemic monoculture collapse, Experiment D1: fixed ensembles (Paper X)
Twenty observer organizations monitor a five‑dimensional latent state. Independent observers each cover a random three of five dimensions with decorrelated noise; shared‑system adopters cover dimensions 1–4 with higher precision and *identical* noise — and are structurally blind to dimension 5, which begins a slow persistent drift at t = 100. Three fixed populations: diverse (20 independent), mixed (5 protected independents + 15 shared), monoculture (20 shared). Detection operates on the independent ensemble's mean estimate of dimension 5; if no one observes it, the precautionary gate can never fire.

Key findings: monoculture fails in 95% of runs — with every instrument reporting acceptable conditions — versus 2% (mixed) and 0% (diverse). The mixed failures are exactly the coverage lottery: (4/10)⁵ ≈ 1% chance that none of the five protected observers draws dimension 5, confirming that in the rank‑deficiency regime failure is a deterministic function of coverage. The protected‑fraction sweep is front‑loaded: failure probability falls 0.98 → 0.40 → 0.16 → 0.08 → 0.03 over the first four independent observers, effectively zero from five. Monte Carlo: 100 seeds (200 per sweep point).

```bash
python gae-simulator-v11-epistemic-monoculture.py
```

### v12 — Consolidation dynamics, Experiment D2: the monoculture attractor (Paper X)
Extends v11 with the switching dynamics of Paper X Part III. All organizations start independent; every ten steps each unprotected organization re‑evaluates its strategy on perceived accuracy and cost. Perceived accuracy is **consensus‑relative** (the true state is unobservable), which produces an emergent positive feedback: shared‑system adopters cluster around the consensus they collectively constitute, so consolidation makes the shared system look progressively better regardless of accuracy against truth. A liability ratchet L(f) = L₀ + L₁·f raises the cost of independence as the shared fraction grows; switching back is rare (atrophied infrastructure); a protected fraction never switches. The regime shift arrives at t = 250 — after twenty‑five evaluation cycles of normal conditions during which consolidation is genuinely the locally rational choice.

Key findings: under a strong liability ratchet (L₁ = 1.5) with no protection, consolidation reaches 97% before the shift and the system fails in 76% of runs; a weak ratchet (L₁ = 0.2) consolidates to only 45% and fails in 6%. **15% constitutional protection reduces failure to zero at every ratchet strength tested** — and exhibits a spillover effect: protected independents anchor the consensus, slowing consolidation among the *unprotected* as well (shared fraction at shift drops from 0.97 to 0.65). The monoculture is near‑absorbing: the surviving strong‑ratchet runs are rescued by rare post‑shift reversions to independence. Monte Carlo: 50 seeds; sweep: protected fraction × L₁ (20 seeds/cell; ~3–5 min runtime).

```bash
python gae-simulator-v12-consolidation-dynamics.py
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
| `outputs/v8-bypass-trap-main.png` | Bypass trap: dysfunction, performance, deception gap, phase portrait |
| `outputs/v8-bypass-trap-sweep.png` | Bypass trap: drift rate × reform rate, three scenarios |
| `outputs/v9-latency-main.png` | Latency asymmetry: trajectories, coverage and latency sweeps, findings |
| `outputs/v9-latency-sweep.png` | Latency asymmetry: τ_ratio × d_I phase diagram, classified outcomes |
| `outputs/v10-bandwidth-main.png` | Bandwidth race: G/R trajectories, two‑threshold phase portrait, collapse timing |
| `outputs/v10-bandwidth-sweep.png` | Bandwidth race: regeneration × capture, constant vs. accelerating demand |
| `outputs/v11-monoculture-trajectories.png` | Monoculture D1: true X₅ vs. controller estimates per scenario |
| `outputs/v11-monoculture-spread.png` | Monoculture D1: independent‑observer ensemble spread diagnostic |
| `outputs/v11-monoculture-phase.png` | Monoculture D1: (X₁, X₅) phase portrait — lockstep drift across the failure boundary |
| `outputs/v11-monoculture-sweep.png` | Monoculture D1: failure probability vs. protected observer fraction |
| `outputs/v12-consolidation-main.png` | Consolidation D2: n(t)/N flow, coverage, X₅, outcome scatter |
| `outputs/v12-consolidation-sweep.png` | Consolidation D2: protected fraction × liability ratchet strength |
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

Constitutional unobservability threshold: SNR < 1, crossed at approximately 2–3 layers.

### v6 — Commons governance and requisite variety

| Architecture | Mean stock | Collapse risk | Gini | Obs dims |
|---|---|---|---|---|
| A — Open access | 4.2% | 93.6% | 0.018 | 1 |
| B — State management | 3.7% | 98.9% | 0.058 | 1 |
| C — Market mechanism | 9.6% | 86.4% | 0.096 | 1 |
| D — Community commons | 27.2% | 30.3% | 0.085 | 3 |
| E — Bioregional / indigenous | 31.1% | 3.6% | 0.032 | 6 |

Collapse threshold: stock below 20% of carrying capacity. Simulation: 30 years, 12 patches, 20 user groups, seed 42.

### v7 — Value‑function collapse (Goodhart‑Ashby synthesis)

| Architecture | dim(V) | Mean W (last 50) | Collapse risk (E < 30) | Final E |
|---|---|---|---|---|
| 1D (GDP‑only) | 1 | ~42 | 1.0 (collapse at t≈ …) | ~ 0 |
| 2D (Wellbeing‑aware) | 2 | ~85 | 0.0 | ~ 55 |

The 1D controller destroys the environmental basis of its own target and collapses; the 2D controller achieves a stable equilibrium. Exact values depend on the disturbance draw; the quoted figures use seed 2024.

### v8 — Bypass‑trap dynamics (75 MC seeds, medians over last 50 steps)

| Scenario | Final dysfunction D | Final performance P | Deception gap (P_perc − P_actual) |
|---|---|---|---|
| No bypass | 0.27 | 0.73 | 0.00 |
| Permanent bypass | 0.37 | 0.66 | ~0.18, persistent |
| Sunset‑coupled bypass | 0.17 | 0.83 | transient, closes at sunset |

The permanent bypass is the low‑performance attractor: highest perceived health, lowest actual ceiling. The sunset advantage over no‑bypass depends on the assumption that deferred reform pressure is productively redirected (multiplier 2.5).

### v9 — Reform‑incumbent latency asymmetry (50 MC seeds, median final architecture quality A)

| Case | τ_ratio | d_I | Final A | Outcome |
|---|---|---|---|---|
| C1 | 0.13 | 8 | 0.33 | absorbed |
| C2 | 0.25 | 6 | 0.43 | contested |
| C3 | 0.75 | 5 | 0.54 | contested |
| C4 | 1.00 | 2 | 0.70 | success |

Full τ_ratio × d_I sweep: the success/absorption boundary is a gradient, not a sharp threshold — dim(T) ≥ dim(I) holds as a tier‑2 heuristic.

### v10 — Transition bandwidth race (60 MC seeds, median event times)

| System | Demand | Bandwidth trap (t, G at trap) | Operational collapse (t) |
|---|---|---|---|
| Federation | constant / accelerating | never | never |
| Bypass‑heavy | constant | never | never |
| Bypass‑heavy | accelerating | 144 (G ≈ 1.4) | 164 |
| Locked | constant | **31 (G ≈ 0.61)** | 136 |
| Locked | accelerating | 29 | 102 |

The two‑threshold structure: the locked regime loses the capacity for self‑redesign (R → 0) at G ≈ 0.61, roughly a hundred steps before operational collapse at G_crit = 1.8.

### v11 — Epistemic monoculture, Experiment D1 (100 MC seeds)

| Scenario | Independent / shared | Failure probability | Detection |
|---|---|---|---|
| Diverse | 20 / 0 | 0.00 | gate at t ≈ 120 |
| Mixed (protected) | 5 / 15 | 0.02 | gate fires unless coverage lottery fails (≈1%) |
| Monoculture | 0 / 20 | 0.95 | never — dimension 5 unobserved |

Protected‑fraction sweep (200 seeds/point): 0.98 → 0.40 → 0.16 → 0.08 → 0.03 → ~0 for 0–5 independent observers.

### v12 — Consolidation dynamics, Experiment D2 (50 MC seeds)

| Scenario | Shared fraction at shift | Failure probability |
|---|---|---|
| Unprotected, weak ratchet (L₁ = 0.2) | 0.45 | 0.06 |
| Unprotected, strong ratchet (L₁ = 1.5) | 0.97 | 0.76 |
| 15% protected, strong ratchet | 0.65 | 0.00 |
| 30% protected, strong ratchet | 0.43 | 0.00 |

2D sweep (protected fraction × L₁, 20 seeds/cell): failure peaks at 0.60 in the unprotected/strong‑ratchet corner and is ≈ 0 for protection ≥ 0.15 at every ratchet strength. Note the spillover: protection slows consolidation among the unprotected as well.

---

## Requirements

```bash
pip install numpy matplotlib
python gae-simulator-v7.py   # or any other version
```

No other dependencies. All versions tested on Python 3.10+. The v12 parameter sweep takes roughly 3–5 minutes; all other simulations complete in well under a minute.

---

## License

MIT. Contributions, extensions, and empirical applications welcome via GitHub issues and pull requests.

---

## Related

- [Global Governance Frameworks](https://globalgovernanceframeworks.org) — the broader governance framework project these simulations support
- [bjornkennethholmstrom.org/working-papers](https://bjornkennethholmstrom.org/working-papers) — full working paper series with mathematical appendices
