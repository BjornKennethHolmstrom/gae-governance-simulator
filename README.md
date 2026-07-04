# Governance as Engineering — Governance Simulator

A series of Python simulations accompanying the **Governance as Engineering** working paper series. Each simulation applies formal tools from control theory, cybernetics, and information theory to a different structural problem in governance design.

All simulations are self-contained, dependency-light, and fully reproducible from fixed random seeds. The later simulations (Paper IX onward) add a Monte Carlo layer (50–200 seeds, reported as median trajectories with 10th–90th percentile bands) and at least one parameter-sweep map demonstrating that the qualitative results are robust across regions of parameter space rather than artefacts of a single configuration.

Files are named by the paper they support: `paper_<roman>_<slug>.py`. Papers with more than one simulation carry a distinguishing slug (e.g. Paper IX has `paper_ix_bypass_trap.py`, `paper_ix_latency_asymmetry.py`, `paper_ix_bandwidth_race.py`). The repository is intentionally flat — every script writes to a relative `outputs/` directory and expects to be run from the repository root. Figures are named to mirror their script (`<script_stem>[_<panel>].png`).

---

## The whitepaper series

| Paper | Title | Core result | Simulator(s) |
|---|---|---|---|
| I | [Governance Stability Simulator](https://bjornkennethholmstrom.org/whitepapers/governance-stability-simulator) | High latency and low signal fidelity place hard mathematical ceilings on governance stability, regardless of institutional quality | `paper_i_single_node_feedback`, `paper_i_multinode_subsidiarity`, `paper_i_multinode_unadjusted` |
| II | [Fractality as Stability](https://bjornkennethholmstrom.org/whitepapers/fractality-as-stability) | No single-scale controller can stabilize a multi-frequency disturbance environment; nested fractal architectures are the stability-optimal solution | `paper_ii_fractal_multiscale` |
| III | [The Observability-Democracy Connection](https://bjornkennethholmstrom.org/whitepapers/observability-democracy-connection) | Representation chains with 3+ layers are constitutionally unobservable; citizen preferences cannot survive to the policy layer regardless of institutional quality | `paper_iii_representation_observability` |
| IV | [Requisite Variety and the Commons](https://bjornkennethholmstrom.org/whitepapers/requisite-variety-and-the-commons) | Commons governance is a feedback loop integrity problem; observation dimensionality — not institutional quality — determines outcomes; state management is worse than open access | `paper_iv_commons_requisite_variety` |
| V | [The Coordination Failure Tax](https://bjornkennethholmstrom.org/whitepapers/coordination-failure-tax) | The four failure modes multiply rather than add; governance systems exhibiting all four are categorically incapable of the functions they claim to perform | *(no simulator)* |
| VI | [The Variety Gap](https://bjornkennethholmstrom.org/whitepapers/the-variety-gap) | What we don't optimize for, we lose the ability to see — objective functions are observation architectures, and low-dimensional value functions produce structural collapse | `paper_vi_value_function_collapse` |
| VII | [The Architecture of Governance Failure](https://www.bjornkennethholmstrom.org/working-papers/architecture-of-governance-failure) | Reform disappoints for structural reasons: the immune system, the bypass trap, and the legibility problem are features of the transition landscape — a qualitative synthesis of fifteen country studies | *(no simulator)* |
| VIII | [Measuring the Variety Gap](https://www.bjornkennethholmstrom.org/working-papers/measuring-the-variety-gap) | The variety gap is measurable: a composite index estimated from observable proxies, with epistemic tiering, across twenty country and organizational cases | *(no simulator)* |
| IX | [The Political Economy of Requisite Governance](https://www.bjornkennethholmstrom.org/working-papers/political-economy-of-requisite-governance) | Transition bandwidth is the ninth structural primitive; latency asymmetry favours embedded incumbents; systems lose the capacity for self-redesign strictly before operational collapse | `paper_ix_bypass_trap`, `paper_ix_latency_asymmetry`, `paper_ix_bandwidth_race` |
| X | [Requisite Observer Diversity](https://www.bjornkennethholmstrom.org/working-papers/requisite-observer-diversity) | Observer diversity is the tenth structural primitive; N_eff = N/(1+(N−1)ρ) — a consolidated ensemble retains N nominal observers but the protection of one, and its blind spots are invisible to every instrument it possesses | `paper_x_epistemic_monoculture`, `paper_x_consolidation_dynamics`, `paper_x_echo_adversarial_fragility` |
| XI | [Reform Exhaustion](https://www.bjornkennethholmstrom.org/working-papers/reform-exhaustion) | The actuation-side dual of constitutional unobservability: as delegation depth grows, the control energy to realise policy intent rises geometrically until the target leaves the reachable set | `paper_xi_chain_prototype` |
| XII | [Boundary Selection Deficits](https://www.bjornkennethholmstrom.org/working-papers/boundary-selection-deficits) | A jurisdiction whose modelled boundary omits active couplings generates unmodelled cross-boundary feedback (an M-Δ loop) that destabilises an internally competent controller — boundary selection is subject to a small-gain condition | `paper_xii_boundary_mismatch` |
| XIII | [Legitimacy as Emergent Gain](https://www.bjornkennethholmstrom.org/working-papers/legitimacy-as-emergent-gain) | Legitimacy is the series' first endogenous coupling state: a gain L that multiplies actuation (B_eff = L·B) and divides observation noise (V = V₀/L), so falling L degrades steering and sensing at once | `paper_xiii_legitimacy_trap` |
| XIV | [Governance as an Adaptive Controller](https://www.bjornkennethholmstrom.org/working-papers/governance-as-adaptive-controller) | Governance is a dual-control problem — act and learn the drifting plant simultaneously; a certainty-equivalent controller that suppresses exploration diverges as the environment drifts, so persistent excitation is structurally required | `paper_xiv_adaptive_controller`, `paper_xiv_sunset` |
| XV | [The Adaptation Bottleneck](https://www.bjornkennethholmstrom.org/working-papers/adaptation-bottleneck) | The dynamic dual of Paper V: the adaptation triad (Sense, Learn, Execute) is a recursive lossy pipeline whose throughput is set by its binding stage | `paper_xv_adaptation_bottleneck` |
| XVI | [Why Diversity Resists Formalization](https://www.bjornkennethholmstrom.org/working-papers/why-diversity-resists-formalization) | Across control theory, evolutionary biology, institutional economics, and decision theory, a quantity representing currently-unused alternatives decays under the primary objective and persists only through a source term the optimizer does not itself set; the corrected order parameter is **source-term locality** — whether that term lies inside or outside the optimizer's control set | `paper_xvi_replenishment_depletion`, `paper_xvi_replenishment_depletion_microfounded`, `paper_xvi_preferential_attachment_fold`, `paper_xvi_switching_barrier_fold`, `paper_xvi_protection_class` |
| XVII | [The Certification Floor](https://www.bjornkennethholmstrom.org/working-papers/certification-floor) | Processing can be made arbitrarily verifiable, but certification of reality cannot be self-verifying — the regress terminates only at a trusted-unverified anchor; automating a coordination boundary relocates its irreducible certification link upstream but never removes it (a scope-bounded relocation invariant, holding for world-coupled but not self-referential coordination) | *(no simulator)* |
| XVIII | [The Boundary Instability Principle](https://www.bjornkennethholmstrom.org/working-papers/boundary-instability) | When learning acts through channels that also carry cross-boundary influence, factorizability becomes a property of the learning trajectory rather than the architecture: generic persistent learning drives the boundary out of every fixed decomposition (Non-Factorizability Theorem), producing a reflexive boundary cycle and a Critical Learning Bandwidth that can pinch shut — and the natural prediction-error early-warning index fails, because local adaptation launders the coupling out of the monitored residuals | `paper_xviii_boundary_instability` |

---

## The Self series (companion papers)

The same control-theoretic grammar applied at the individual scale: the self as a self-governing system.

| Paper | Title | Core result | Simulator(s) |
|---|---|---|---|
| Self I | [The Variety Gap in the Self](https://bjornkennethholmstrom.org/working-papers/self-variety-gap) | Personal values are observation architectures; a low-dimensional value function destroys self-observability and the excluded dimensions re-enter as crisis — the Goodhart–Ashby synthesis for the individual, with the self-variety gap **G_self** | `self_i_variety_gap` |
| Self II | [Adaptive Self-Governance](https://bjornkennethholmstrom.org/working-papers/adaptive-self-governance) | Observer–plant identity: in a self the controller *is* the plant, so self-observation cannot be performed without acting on the observed; self-revision is bounded two-sidedly (calibration below, coherence above) | `self_ii_appendix_a`…`_e` (five appendix sims) |
| Self III | [The Operator](https://bjornkennethholmstrom.org/working-papers/self-operator) | Institutions inherit the perceptual limits of their operators: institutional observability of interior dimensions is upper-bounded by the interior observational capacity of the human nodes through which they pass — *inherited unobservability* | `self_iii_operator`, `self_iii_formation` |

---

## Simulators

### Prototype — Conceptual sketch
`prototype_conceptual_sketch.py` — exploratory prototype predating the paper series. Not documented in the working papers; kept for provenance.

### Paper I — Governance stability
- `paper_i_single_node_feedback.py` — the core feedback loop: a single governance node with configurable latency and signal fidelity under an external disturbance. Demonstrates the stability ceiling imposed by high latency.
- `paper_i_multinode_subsidiarity.py` — extends the model to a ten-node spatial system. A localized shock at nodes 2 and 7 is visible to the local controller but invisible to the central controller, which sees only the aggregate. Demonstrates the averaging problem — the formal basis for subsidiarity.
- `paper_i_multinode_unadjusted.py` — identical to the subsidiarity model but with gain set above the stability ceiling. Demonstrates the oscillatory failure mode predicted by the gain-margin analysis.

### Paper II — Fractality as stability
`paper_ii_fractal_multiscale.py` — a system facing simultaneous fast, medium, and slow disturbances, governed by three architectures (single global, single local, fractal nested). Demonstrates the frequency-gap theorem: no single-scale controller covers all disturbance bands.

### Paper III — Observability and democracy
`paper_iii_representation_observability.py` — 60 citizen groups holding preferences across 4 dimensions, transmitted through representation chains of 1–5 layers. Computes SNR at the policy layer and demonstrates the constitutional unobservability threshold: SNR < 1 at ≥3 layers, regardless of institutional quality.

### Paper IV — Commons and requisite variety
`paper_iv_commons_requisite_variety.py` — a 12-patch renewable resource governed by five architectures over 30 years, subject to fast stochastic shocks, seasonal cycles, and a slow decadal carrying-capacity decline. State management (annual survey) reaches 98.9% collapse risk — worse than open access — because high observation latency combined with single-dimension aggregation produces destabilising interventions. Only the multi-dimensional architectures (community commons, bioregional/indigenous) avoid near-certain collapse.

### Paper VI — Value-function collapse
`paper_vi_value_function_collapse.py` — a minimal Goodhart–Ashby model. Two coupled states, Wealth (W) and Environment (E), governed by a 1D controller (GDP-only, observes W) and a 2D controller (wellbeing-aware, observes W and E). The 1D controller drives growth, silently degrades E, and collapses because its narrow value architecture destroys the condition sustaining its own target. A direct instantiation of the variety gap **G = dim(R) − dim(V) = 2 − 1 = 1 > 0**.

### Paper IX — Political economy of requisite governance
- `paper_ix_bypass_trap.py` — a dysfunctional substrate (dysfunction D) and a parallel bypass institution (load share B). Reform pressure responds only to *visible* dysfunction D·(1−B), so the bypass hides the problem it routes around. The permanent bypass is the stable low-performance attractor (highest perceived health, lowest actual ceiling); a sunset-coupled bypass escapes it. Monte Carlo: 75 seeds; sweep over drift rate × reform rate.
- `paper_ix_latency_asymmetry.py` — two controllers on the same ten-dimensional architecture state: a reform coalition (8 dims, latency τ_R) and an incumbent (d_I dims, latency τ_I ≤ τ_R). The τ_ratio × d_I phase diagram shows a gradient boundary, not a sharp threshold — disciplining the dim(T) ≥ dim(I) condition down to a tier-2 heuristic. Monte Carlo: 50 seeds.
- `paper_ix_bandwidth_race.py` — couples the variety gap G(t) to the reform capacity R(t) that closes it. The two-threshold structure: the locked regime enters a transition-bandwidth trap (R → 0 at G ≈ 0.61) roughly a hundred steps before operational collapse at G_crit = 1.8. Monte Carlo: 60 seeds; sweep over regeneration × capture.

### Paper X — Requisite observer diversity
- `paper_x_epistemic_monoculture.py` — twenty observers monitor a five-dimensional latent state; shared-system adopters are structurally blind to dimension 5, which drifts from t=100. Monoculture fails in 95% of runs with every instrument reporting acceptable conditions, versus 2% (mixed) and 0% (diverse). Monte Carlo: 100 seeds.
- `paper_x_consolidation_dynamics.py` — adds switching dynamics: perceived accuracy is consensus-relative (the true state is unobservable), producing an emergent positive feedback toward consolidation, plus a liability ratchet. 15% constitutional protection reduces failure to zero at every ratchet strength, and exhibits a spillover effect. Monte Carlo: 50 seeds; sweep over protected fraction × L₁ (~3–5 min).
- `paper_x_echo_adversarial_fragility.py` — asks whether the variance-optimal observer allocation builds its own attack surface. It does — but only against dependence the optimizer could not measure (a hidden common-input factor omitted from the defender's estimated covariance). Separates a structural adversary (observable-factor and hidden-spoof moves) from an omniscient bound.

### Paper XI — Reform exhaustion
`paper_xi_chain_prototype.py` — the actuation-side dual of Paper III. Policy intent is transmitted down a delegation chain of increasing depth; each layer projects the directive onto its own repertoire and adds noise, so control energy rises geometrically until the target leaves the reachable set. *Illustrative prototype (v0.1), not the series-convention full simulation.*

### Paper XII — Boundary selection deficits
`paper_xii_boundary_mismatch.py` — coupled subsystems partitioned into jurisdictions (stochastic block model) with perfect internal controllers. Four boundary scenarios (matched, Westphalian, Sykes-Picot, adaptive). When a modelled boundary omits active couplings, the unmodelled cross-boundary feedback forms an M-Δ loop that destabilises an internally well-designed controller — boundary selection is subject to a small-gain condition.

### Paper XIII — The legitimacy trap
`paper_xiii_legitimacy_trap.py` — legitimacy as an endogenous gain L that multiplies actuation (B_eff = L·B) and divides observation noise (V = V₀/L). Four scenarios trace the performance–legitimacy spiral into a low-trust attractor, with hysteresis and a borrowed-legitimacy betrayal mechanism.

### Paper XIV — Governance as an adaptive controller
- `paper_xiv_adaptive_controller.py` — the dual-control problem: regulate a two-dimensional state while learning its drifting parameters. Six scenarios (Appendix B.3) contrast optimal dual control against exploration starvation, over-exploration, forgetting-without-learning, and exploitation lock-in, plus three sweeps.
- `paper_xiv_sunset.py` — the sunset decision as dual control (Section 6.9 / Appendix A.4). A regulator's operational record cannot separate a low threat from an effective regulator; only a protected probe channel identifies the latent rate. A Bayesian grid filter over (log λ, e) shows why a regulator's own record is incapable of certifying its necessity.

### Paper XV — The adaptation bottleneck
`paper_xv_adaptation_bottleneck.py` — the adaptation triad (Sense, Learn, Execute) as a recursive lossy pipeline. Four experiments recover the allocation optimum (equalise efficiency-scaled stage rates), the three backlog regimes, and the closure-delay depression T_eff,rec = T_raw / (1 + τ·T_raw). Overall throughput is set by the binding stage.

### Paper XVI — Why Diversity Resists Formalization
The race between variety **replenishment** (entry of independent observers) and **depletion** (herding toward consensus), and whether it produces a hysteretic collapse of N_eff. The negative controls are load-bearing: the point is that a fold is *not* generic.
- `paper_xvi_replenishment_depletion.py` — the base race with a posited super-linear herding term; finds a fold with hysteresis.
- `paper_xvi_replenishment_depletion_microfounded.py` — graduation test: does the fold survive when herding is derived from a local rule (correlation neglect) rather than a hand-inserted global ρ̄? 
- `paper_xvi_preferential_attachment_fold.py` — preferential-attachment copying; a real size-independent condensation transition (γ>1) but a single attractor — condensation, not a fold.
- `paper_xvi_switching_barrier_fold.py` — preferential copying plus a switching barrier; locates the critical lock-in strength B\* above which a genuine, horizon-robust fold opens.
- `paper_xvi_protection_class.py` — backs §6 (the protection class between physically-uneditable and politically-revocable); three experiments on the shared N_eff order parameter, two of them null controls.

### Paper XVIII — The Boundary Instability Principle
`paper_xviii_boundary_instability.py` — two subsystems whose local controllers believe their jurisdictions are closed, coupled through a channel that depends on boundary clarity and an accumulated coupling stock. Local gradient-descent learning absorbs cross-boundary influence into internal gain, restoring perceived calm without removing real coupling. Tests the reflexive boundary cycle, the Boundary Dissolution Index (which fails as early warning — the coupling is laundered out of the residuals it monitors), and the Critical Learning Bandwidth as registered predictions.

---

## Companion simulations (Self series)

### Self I — The variety gap in the self
`self_i_variety_gap.py` — a person with five coupled life dimensions (Health, Relationships, Meaning, Career, Leisure) under a narrow career-only architecture versus a wider multi-dimensional one. The same collapse pattern at the individual scale; the self-variety gap **G_self** is computed for each architecture.

### Self II — Adaptive self-governance (five appendix sims)
Each isolates one structural result of self-governance:
- `self_ii_appendix_a_correlation_tax.py` — the correlation tax across internal observers.
- `self_ii_appendix_b_actuation_chain.py` — attenuation along the internal actuation chain (the energy law).
- `self_ii_appendix_c_self_legitimacy.py` — self-legitimacy as an internal coupling gain (existence bifurcation, hysteresis, built-vs-borrowed, the transparency trap).
- `self_ii_appendix_d_two_sided_bound.py` — the two-sided bound on self-revision (calibration below, coherence above).
- `self_ii_appendix_e_observer_plant.py` — observer–plant identity: self-observation acting on the observed.

### Self III — The operator
- `self_iii_operator.py` — the bridge back to the governance line. Reuses the Paper XIII legitimacy loop unchanged and adds one operator node on the interior dimension, governed by interior fidelity φ. Sweeping φ downward exhibits a threshold (φ\* ≈ 0.33 at the chosen parameters) below which a well-formed system crosses into the low-legitimacy attractor — *inherited unobservability*. Monte Carlo: 100 seeds.
- `self_iii_formation.py` — the "Formation of the Observer" section. Tests whether early formation reproduces a formative source's *blindness* with higher fidelity than its *sight*, governed by the number of decorrelated alternative observers a child can reach.

### Study I — Observer correlation (empirical)
`study_i_observer_correlation.py` — an empirical protocol (S1-0.3) testing Paper X on a real AI ensemble. Works in log-error space, treats shared bias as correlated error (uncentered second moments), and estimates ρ_eff with the exact identity MSE_ens/mean(MSE_m) = (1−ρ_eff)/N + ρ_eff.

---

## Simulation outputs

Figures are written to `outputs/` and named to mirror their script.

| File | Description |
|---|---|
| `outputs/paper_i_single_node_feedback.png` | Stability ceiling: latency vs. disturbance response |
| `outputs/paper_i_multinode_subsidiarity.png` | Averaging problem: local shock invisible to central controller |
| `outputs/paper_i_multinode_unadjusted.png` | Instability: gain above the stability ceiling |
| `outputs/paper_ii_fractal_multiscale.png` | Frequency gap: three architectures across three disturbance bands |
| `outputs/paper_iii_representation_observability.png` | Observability: SNR collapse and preference tracking across four architectures |
| `outputs/paper_iv_commons_requisite_variety.png` | Commons: resource stock, requisite variety, equity, slow-variable tracking |
| `outputs/paper_vi_value_function_collapse.png` | Goodhart–Ashby: 1D vs 2D value architecture, wealth–environment collapse |
| `outputs/paper_ix_bypass_trap_main.png`, `…_sweep.png` | Bypass trap: dysfunction, performance, deception gap; drift × reform sweep |
| `outputs/paper_ix_latency_asymmetry_main.png`, `…_sweep.png` | Latency asymmetry: trajectories; τ_ratio × d_I phase diagram |
| `outputs/paper_ix_bandwidth_race_main.png`, `…_sweep.png` | Bandwidth race: G/R trajectories; regeneration × capture sweep |
| `outputs/paper_x_epistemic_monoculture_{trajectories,spread,phase,sweep}.png` | Monoculture D1: trajectories, ensemble spread, phase portrait, protected-fraction sweep |
| `outputs/paper_x_consolidation_dynamics_main.png`, `…_sweep.png` | Consolidation D2: flow/coverage/outcomes; protection × ratchet sweep |
| `outputs/paper_x_echo_adversarial_fragility.png` | Observer allocation: attack surface against unmeasured dependence |
| `outputs/paper_xii_boundary_mismatch_{adaptive_trajectory,adaptive_sweep,stability_loopgain}.png` | Boundary mismatch: adaptive-boundary trajectory/sweep, small-gain loop-gain |
| `outputs/paper_xiii_legitimacy_trap_{phase_diagram,trap_and_recovery,borrowed_collapse,collapse_heatmap,asymmetry_sweep}.png` | Legitimacy trap: phase diagram, trap/recovery, borrowed collapse, heatmaps |
| `outputs/paper_xiv_adaptive_controller_{phase_diagram,starvation_vs_optimal,exploitation_lockin,forgetting_sweep}.png`, `…_summary_metrics.csv` | Adaptive controller: phase diagram, starvation vs optimal, sweeps, metrics |
| `outputs/paper_xiv_sunset_time_to_removal.png`, `…_tradeoff.png` | Sunset decision as dual control: time-to-removal, tradeoff |
| `outputs/paper_xv_adaptation_bottleneck_{A_allocation,B_backlogs,C_closure_delay,D_self_blinding}.png` | Adaptation bottleneck: allocation, backlogs, closure delay, self-blinding |
| `outputs/paper_xvi_preferential_attachment_fold.png` | Condensation transition (γ=1 vs γ>1) |
| `outputs/paper_xvi_replenishment_depletion.png`, `…_microfounded.png` | Posited vs derived fold |
| `outputs/paper_xvi_switching_barrier_fold.png` | Critical lock-in strength and hysteresis window |
| `outputs/paper_xvi_protection_class.png` | Protection-class experiments |
| `outputs/paper_xviii_boundary_instability_{A_phase_cycle,A2_regime_map,B_early_warning,C_bandwidth_slice,C2_window_map}.png` | Boundary instability: phase cycle, regime map, early warning, bandwidth slice, window map |
| `outputs/self_i_variety_gap.png` | Self I: career-only vs multi-dimensional personal value architecture |
| `outputs/self_iii_operator_{phi_sweep,legitimacy_trajectories,interior_and_gap}.png` | Self III: φ-sweep separatrix, legitimacy trajectories, interior gap |
| `outputs/study_i_observer_correlation.png` | Study I: observer correlation tax demo |

---

## Requirements

```bash
pip install numpy matplotlib scipy
python paper_vi_value_function_collapse.py   # or any other simulation
```

`numpy` and `matplotlib` are used everywhere; `scipy` is needed by a few simulations (e.g. `paper_xi_chain_prototype.py`, `study_i_observer_correlation.py`). All simulations tested on Python 3.10+. The Paper X consolidation sweep takes roughly 3–5 minutes; all other simulations complete in well under a minute. Run from the repository root so the relative `outputs/` path resolves.

---

## Results summary

### Paper III — Representation chain observability

| Architecture | Layers | Mean tracking error | Variance survived | SNR |
|---|---|---|---|---|
| A — Deep democracy | 5 | 0.160 | 0% | 0.002 |
| B — Representative | 3 | 0.077 | 0% | 0.048 |
| C — Semi-direct | 2 | 0.022 | 79% | 0.254 |
| D — Direct/participatory | 1 | 0.008 | 100% | 1.780 |

Constitutional unobservability threshold: SNR < 1, crossed at approximately 2–3 layers.

### Paper IV — Commons governance and requisite variety

| Architecture | Mean stock | Collapse risk | Gini | Obs dims |
|---|---|---|---|---|
| A — Open access | 4.2% | 93.6% | 0.018 | 1 |
| B — State management | 3.7% | 98.9% | 0.058 | 1 |
| C — Market mechanism | 9.6% | 86.4% | 0.096 | 1 |
| D — Community commons | 27.2% | 30.3% | 0.085 | 3 |
| E — Bioregional / indigenous | 31.1% | 3.6% | 0.032 | 6 |

Collapse threshold: stock below 20% of carrying capacity. 30 years, 12 patches, 20 user groups, seed 42.

### Paper VI — Value-function collapse (Goodhart–Ashby)

| Architecture | dim(V) | Mean W (last 50) | Collapse risk (E < 30) | Final E |
|---|---|---|---|---|
| 1D (GDP-only) | 1 | ~42 | 1.0 | ~0 |
| 2D (Wellbeing-aware) | 2 | ~85 | 0.0 | ~55 |

The 1D controller destroys the environmental basis of its own target and collapses; the 2D controller reaches a stable equilibrium. Figures use seed 2024.

### Paper IX — Bypass-trap dynamics (75 MC seeds)

| Scenario | Final dysfunction D | Final performance P | Deception gap (P_perc − P_actual) |
|---|---|---|---|
| No bypass | 0.27 | 0.73 | 0.00 |
| Permanent bypass | 0.37 | 0.66 | ~0.18, persistent |
| Sunset-coupled bypass | 0.17 | 0.83 | transient, closes at sunset |

### Paper IX — Reform-incumbent latency asymmetry (50 MC seeds)

| Case | τ_ratio | d_I | Final A | Outcome |
|---|---|---|---|---|
| C1 | 0.13 | 8 | 0.33 | absorbed |
| C2 | 0.25 | 6 | 0.43 | contested |
| C3 | 0.75 | 5 | 0.54 | contested |
| C4 | 1.00 | 2 | 0.70 | success |

The success/absorption boundary is a gradient, not a sharp threshold.

### Paper IX — Transition bandwidth race (60 MC seeds)

| System | Demand | Bandwidth trap (t, G) | Operational collapse (t) |
|---|---|---|---|
| Federation | constant / accelerating | never | never |
| Bypass-heavy | constant | never | never |
| Bypass-heavy | accelerating | 144 (G ≈ 1.4) | 164 |
| Locked | constant | **31 (G ≈ 0.61)** | 136 |
| Locked | accelerating | 29 | 102 |

The locked regime loses the capacity for self-redesign (R → 0) at G ≈ 0.61, roughly a hundred steps before operational collapse at G_crit = 1.8.

### Paper X — Epistemic monoculture, Experiment D1 (100 MC seeds)

| Scenario | Independent / shared | Failure probability | Detection |
|---|---|---|---|
| Diverse | 20 / 0 | 0.00 | gate at t ≈ 120 |
| Mixed (protected) | 5 / 15 | 0.02 | gate fires unless coverage lottery fails (≈1%) |
| Monoculture | 0 / 20 | 0.95 | never — dimension 5 unobserved |

Protected-fraction sweep (200 seeds/point): 0.98 → 0.40 → 0.16 → 0.08 → 0.03 → ~0 for 0–5 independent observers.

### Paper X — Consolidation dynamics, Experiment D2 (50 MC seeds)

| Scenario | Shared fraction at shift | Failure probability |
|---|---|---|
| Unprotected, weak ratchet (L₁ = 0.2) | 0.45 | 0.06 |
| Unprotected, strong ratchet (L₁ = 1.5) | 0.97 | 0.76 |
| 15% protected, strong ratchet | 0.65 | 0.00 |
| 30% protected, strong ratchet | 0.43 | 0.00 |

2D sweep (protected fraction × L₁, 20 seeds/cell): failure peaks at 0.60 in the unprotected/strong-ratchet corner and is ≈ 0 for protection ≥ 0.15 at every ratchet strength. Protection also slows consolidation among the unprotected.

---

## License

MIT. Contributions, extensions, and empirical applications welcome via GitHub issues and pull requests.

---

## Related

- [Global Governance Frameworks](https://globalgovernanceframeworks.org) — the broader governance framework project these simulations support
- [bjornkennethholmstrom.org/working-papers](https://bjornkennethholmstrom.org/working-papers) — full working paper series with mathematical appendices
