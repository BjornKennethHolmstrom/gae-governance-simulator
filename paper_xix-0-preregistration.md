# Preregistration — Role-triad replication (Paper XIX)

**Status:** Written before the registered run. The single-zoo results in `03-results.md`, `04-results.md`, and `05-results.md` are the **pilot**: they motivated the predictions and thresholds below but do not count as evidence for the registered claims.

## The registered distinction: phenomenon, not identity

The pilot found that *specific* models play specific roles (`normal_h8` warns, `damped_h8` bridges, `blur_h8` is near dead weight). None of that is registered. Those are properties of seven particular trained artifacts. What is registered is that the **role structure recurs across independently trained model zoos**: that governor value, sentinel value, and bridge value *dissociate* — that the model which governs best is generally not the one that warns best or connects best. Model identities, per-regime bridge names, and the geometry/topology heatmaps remain exploratory throughout.

**Replication unit.** One *zoo* = one seed. Each seed reseeds environment generation, all seven model initializations and their training data, and the evaluation streams. The protocol retrains the whole zoo per seed rather than re-evaluating a fixed zoo. Registered runs use seeds 0–19 (20 zoos). Script: `paper_xix-1-role_triad_replication.py`.

**Zoo composition (fixed across seeds).** Seven models, matching the pilot: `normal_h8`, `normal_h16`, `compressed_h2`, `wind_h8`, `damped_h8`, `blur_h8`, `velocity_aux_h8`. Composition is held fixed so that dissociation cannot be manufactured by changing the roster; only the seed varies.

**Regime-shift stream (fixed structure).** Six segments — normal, wind, damped, blur, normal, wind — each of equal length, concatenated. Identical structure to the pilot; only the seed-driven realization changes.

## Registered scores (computable on any zoo)

For each model $m$ in a zoo, evaluated on that zoo's stream:

- **Governor score** $G_m$: negative mean stream MSE of $m$ used as sole controller (higher = better governor). Equivalently ranked by mean MSE ascending.
- **Sentinel score** $S_m$: from the robust rolling-window detector (below). A true positive is a warning — $m$ consistently outperforming the active model by margin $\epsilon$ over a rolling window while suppressed — that precedes an active-model error spike within horizon $H$ by at least $L_{\min}$ steps. $S_m$ is the count of unique spike episodes $m$ warns for (its coverage), the registered sentinel quantity. Precision is reported alongside but is not the score.
- **Bridge score** $B_m$: betweenness centrality of $m$ in the behavioral-distance graph thresholded at the connectivity threshold $\epsilon_c$ (the minimum threshold at which the graph is connected). Articulation-point status is reported alongside.

Detector parameters (fixed, from the pilot's robust detector): rolling window 20 steps, margin $\epsilon$ = 10% of active error, horizon $H$ = 50 steps, minimum lead $L_{\min}$ = 10 steps, spike = active error exceeding twice the regime's opening-window median.

## Registered predictions

**P1 — Adaptive pluralism approximates the oracle.** Across the 20 zoos, mean stream MSE orders as: WTA-oracle < adaptive-audit < full-pluralism < monoculture < WTA-closed, and adaptive-audit is closer to the oracle than to full-pluralism (i.e. $|\text{MSE}_{\text{adaptive}} - \text{MSE}_{\text{oracle}}| < |\text{MSE}_{\text{adaptive}} - \text{MSE}_{\text{full}}|$) in a majority of zoos.
*Registered pass:* the median ordering holds and the adaptive-closer-to-oracle condition holds in $\ge 12/20$ zoos.
*Null committed to:* if closed-WTA is not worst in a majority, or adaptive does not beat monoculture in a majority, P1 fails and the architecture claim is reported as not replicated.

**P2 — Governor/sentinel dissociation.** The best governor is not the best sentinel. Registered on two operationalizations, both committed:
1. In $\ge 14/20$ zoos, $\arg\max_m G_m \neq \arg\max_m S_m$.
2. The across-zoo rank correlation between $G_m$ and $S_m$ (Spearman, pooled over all model-zoo pairs) is < 0.5.
*Null committed to:* if the top-governor equals the top-sentinel in a majority of zoos, or the rank correlation is $\ge 0.5$, P2 fails and the sentinel role is reported as not distinct from governing value.

**P3 — Diverse portfolios beat top-utility portfolios on coverage.** At portfolio size $K = 4$, a portfolio selected greedily for marginal spike coverage ("diverse") achieves higher unique coverage than a portfolio of the four highest-governor models ("top-utility"), in a majority of zoos.
*Registered pass:* diverse coverage > top-utility coverage in $\ge 12/20$ zoos, and the mean coverage advantage is positive.
*Null committed to:* if top-utility coverage $\ge$ diverse coverage in a majority, P3 fails; sentinel preservation is then reported as a ranking problem after all, not a portfolio problem.

**P4 — Governor/bridge dissociation.** Bridge value is not governing value. Registered:
1. In $\ge 14/20$ zoos, the top-bridge model (max betweenness at $\epsilon_c$) is not the top-governor model.
2. In $\ge 12/20$ zoos, at least one model outside the top-2 governors is an articulation point at $\epsilon_c$ in at least one of the six regime graphs.
*Null committed to:* if the top governor is usually also the top bridge, or non-top-governors are rarely articulation points, P4 fails and the bridge role collapses into governing value.

## Nesting note

P1 is independent (an architecture claim). P2, P3, P4 are the triad. P2 and P4 are the two dissociation legs; P3 is the design consequence of P2 (that the dissociation is exploitable by portfolio selection). P3 can pass only if sentinel coverage varies across models, which P2 presupposes but does not entail; they are logically nested, not identical.

## Analysis plan

Distributions across the 20 zoos, not single trajectories: for each registered quantity, report median and IQR and the pass/fail count against threshold. No zoo excluded except on a crashed run (reported). No post-hoc threshold changes; if a threshold proves ill-chosen the registered result stands and any re-analysis is labeled exploratory. Outputs: `role_triad_results.csv` (one row per model per zoo, with $G_m, S_m, B_m$, precision, articulation flag), `portfolio_results.csv` (per-zoo diverse vs top-utility coverage at K=1..4), and `role_triad_summary.txt` (P1–P4 pass/fail with counts).

## Exploratory (declared, not registered)

Per-regime distance heatmaps and their cross-regime correlation (the environment-dependent-metric result); named model roles; the sentinel precision/recall front; cycle-rank / redundancy trends; translation-burden ($\epsilon_c$) ordering across regimes. These illustrate; they are not tested against thresholds.

## What failure would mean

- P1 fails → the sensing-pluralism-plus-closure architecture is not robustly better than its baselines; §3 of the paper becomes a negative result.
- P2 fails → the sentinel role is not distinct from governing value; the triad collapses toward a dyad and the paper's thesis is substantially weakened.
- P3 fails → dissociation exists but is not exploitable by portfolio construction; the design rule ("select by marginal coverage") is withdrawn.
- P4 fails → the bridge role is not distinct; the triad reduces to governor + sentinel, and the topology material moves to the deferred sibling paper.
