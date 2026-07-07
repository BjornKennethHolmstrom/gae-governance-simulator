# Preregistration — Learning ≠ adaptation demo (Paper XXI)

**Status:** Written before the run. No pilot; the explorations behind Paper XXI (`10-learning-versus-adaptation.md`) are argument, not simulation. This demo is the paper's single empirical anchor and is registered from scratch.

## What the demo tests

Paper XXI §2 separates two quantities usually conflated. *Learning* improves the fidelity of a controller's internal model (judged by prediction error against the world). *Adaptation* maintains adequate coupling between the controller and its world (judged by whether the controller stays within viability bounds while acting). The claim is that these dissociate, and specifically that learning can improve fidelity while *degrading* coupling, because incorporating a model revision has a cost the action loop must absorb. The sharpened form is the absorptive-capacity inequality:

$$D_\text{revealed} + D_\text{created} \le C_\text{absorb}$$

Adaptation succeeds only when the demand learning makes actionable — mismatch it *reveals* plus incorporation cost it *creates* — stays within what the loop can absorb. Faster learning raises both left-hand terms; past the point where they exceed $C_\text{absorb}$, coupling degrades even as fidelity keeps improving.

## The model

**Note on revision:** a first operationalization (a slew-capped actuator tracking the estimate) was run and failed P2 — coupling rose monotonically with the learning rate, because a slew cap alone does not make incorporation costly when the estimate stays centred on the target. That run is on record as the first attempt. The mechanism below makes incorporation genuinely costly: a model revision injects a decaying disruption into the action loop proportional to the revision size, so churning the model faster than the loop can settle degrades coupling. This is a pre-run redesign; thresholds below are re-derived for it.

A controller tracks a slowly drifting latent target $\theta_t$ and must both (i) estimate it — the model-fidelity channel — and (ii) hold an actuator near it — the coupling channel. Each step:

- The environment drifts: $\theta_{t+1} = \theta_t + \text{drift} + \text{noise}$.
- The controller updates its estimate toward a noisy observation at **learning rate** $\eta$: $\hat\theta \leftarrow \hat\theta + \eta\,(\text{obs} - \hat\theta)$. Higher $\eta$ = faster learning = better fidelity, but larger and more frequent revisions.
- Each revision has size $|\Delta\hat\theta|$ and injects **incorporation disruption** into the action loop: $\text{disrupt} \leftarrow \text{decay}\cdot\text{disrupt} + \text{cost}\cdot|\Delta\hat\theta|$. This is $D_\text{created}$ — the cost the loop must absorb — accumulating across rapid revisions and decaying between them.
- A second-order actuator chases the estimate ($\text{vel} \leftarrow \text{vel} + \text{stiff}(\hat\theta - a) - \text{damp}\cdot\text{vel}$; $a \leftarrow a + \text{vel}$), but its effective position is perturbed by the current disruption: while the loop is still absorbing revisions it cannot settle. The loop's capacity to absorb is $C_\text{absorb}$; when revision demand exceeds it, the actuator is kept out of the viability band.
- **Coupling** at each step is $|a - \theta| \le \tau$. **Fidelity** is $-|\hat\theta - \theta|$.

The tension is now two-sided. At low $\eta$ the estimate lags the drift (poor fidelity) and the actuator faithfully tracks a lagging estimate (poor coupling). At high $\eta$ the estimate tracks well (good fidelity) but churns on noise, and the sustained incorporation disruption keeps the actuator thrashing (poor coupling). Coupling is therefore best at an intermediate $\eta$ and worse at both ends — while fidelity rises monotonically throughout. This is $D_\text{revealed} + D_\text{created} > C_\text{absorb}$ made mechanical at high $\eta$.

Swept variable: learning rate $\eta \in \{0.02, 0.05, 0.1, 0.2, 0.4, 0.7\}$. Per seed (30 seeds), drift, noise, and initial conditions are drawn; drift, noise, viability band $\tau$, actuator stiffness/damping, incorporation cost, and disruption decay are fixed across seeds (values in the script). Each run is 4000 steps; the first 500 are discarded as burn-in.

## Registered predictions

**P1 — Fidelity is monotone in learning rate.** Median model-fidelity (negative estimate-tracking error) improves monotonically with $\eta$ across the swept range: pooled Spearman$(\eta, \text{fidelity}) > 0.9$, and median fidelity at the highest $\eta$ exceeds that at the lowest.
*Null committed to:* if fidelity is not monotone increasing in $\eta$, the model-fidelity channel is not behaving as a learning channel and the demo is mis-specified.

**P2 — Coupling is non-monotone: it peaks at intermediate learning and degrades at high learning.** Mean coupling (fraction of post-burn-in steps within the viability band) rises then falls across $\eta$: the $\eta$ maximizing median coupling is strictly interior to the swept range (not the smallest or largest $\eta$), and median coupling at the highest $\eta$ is below the interior peak by at least 0.10 (10 percentage points), in $\ge 24/30$ seeds.
*Null committed to:* if coupling is monotone in $\eta$ (either direction) in a majority of seeds, learning and adaptation do not dissociate in this model and the central claim is unsupported here.

**P3 — The dissociation regime is explicit: fidelity up while coupling down.** Between the coupling-optimal $\eta$ and the highest $\eta$, fidelity increases while coupling decreases, in $\ge 24/30$ seeds. This is the learning-improves-while-adaptation-degrades signature stated directly.
*Null committed to:* if fidelity and coupling move in the same direction across that interval in a majority, there is no regime where learning helps the map and hurts the coupling, and the paper's key line ("learning updates the map; adaptation requires surviving the redraw") loses its empirical instance.

## Analysis plan

Distributions across 30 seeds: for fidelity and coupling, median and IQR per $\eta$; the three pass/fail counts against threshold. One figure: fidelity and coupling versus $\eta$ (the dissociation scissors — fidelity rising, coupling peaking then falling). No seed excluded except on a crashed run (reported). No post-hoc threshold changes; a misjudged threshold stands and any re-analysis is labeled exploratory. Pure numpy/matplotlib; seconds on a CPU.

## What failure would mean

- P1 fails → the learning channel is mis-specified; fix before trusting P2/P3.
- **P2 fails → learning and adaptation do not dissociate in this model**; the absorptive-capacity mechanism is not demonstrated and §3 reports the separation as conceptual only (§2's definitional argument), with no empirical instance. This is the load-bearing prediction.
- P3 fails → coupling may be non-monotone for a reason other than the fidelity/coupling trade-off; the specific "map improves while coupling degrades" claim is withdrawn even if P2's non-monotonicity holds.
