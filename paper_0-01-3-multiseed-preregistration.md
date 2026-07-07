# Preregistration — Multi-seed replication of the capacity-sweep factorization experiment

**Status:** Written before the registered run. The single-seed run reported in `r2_summary.txt` is hereby designated a **pilot**; it informed the thresholds below but does not count as evidence for the registered claims.

**Protocol:** `01-4-multiseed_factorization.py`, seeds 0–19 (20 seeds), hidden sizes {2, 4, 8, 16}. Each seed controls environment generation, train/val split, and model initialization — i.e., replication is over both world-realization and training run. Config as committed in the script: 600 trajectories × 200 steps, input length 20, prediction offsets {5, 10, 20}, 25 epochs with early stopping (patience 5), linear probes from final hidden state to the true latent state (x, y, vx, vy), R² on a held-out validation subset of 2,000 samples.

**Probe commitment:** Linear regression only. A nonlinear probe could recover more; the registered claims are about *linearly decodable* structure, matching the pilot. This is a stated limitation, not an oversight.

**Definitions.** Per-axis recovery:

- A_x = (R²_x + R²_vx) / 2
- A_y = (R²_y + R²_vy) / 2
- Asymmetry: asym = |A_x − A_y|
- Dropped axis at h=2: argmin(A_x, A_y) if asym > 0.3, else classified "uniform degradation".

Negative R² values are clipped to 0 before computing A (a probe worse than the mean predictor recovers nothing).

## Primary predictions

**P1 — Emergence.** At h=8, the median R² across seeds is ≥ 0.7 for each of the four latent variables. The hidden state recovers the full causal state without supervision on it.
*Null committed to:* if any variable's median falls below 0.5, P1 fails and the emergence claim is reported as not replicated at this capacity.

**P2 — Structured blindness.** At h=2, capacity starvation is structured, not uniform: asym > 0.3 in at least 80% of seeds (≥ 16/20).
*Null committed to:* if 6 or more seeds show uniform degradation, P2 fails; the "capacity selects dimensions" claim is then demoted from a mechanism claim to a sometimes-outcome, and Paper 0 Section 3 must say so.

**P3 — Symmetry breaking.** Among h=2 seeds classified as having a dropped axis, both axes occur as the dropped one. The environment is x/y-symmetric, so which subspace goes blind is training-contingent — non-uniqueness appearing inside a single architecture.
*Threshold:* each axis is dropped in at least 3 seeds (of the ≥16 expected under P2).
*Null committed to:* if one axis is dropped in ≥ 90% of dropping seeds, an unnoticed asymmetry exists in the environment or rendering; P3 fails and the symmetry-breaking interpretation is withdrawn pending diagnosis.

## Secondary prediction

**P4 — Diminishing abstraction.** h=16 achieves a lower median validation loss than h=8, but the median latent R² does not improve: for each latent variable, median R²(h16) ≤ median R²(h8) + 0.05. Surplus capacity buys pixel-level prediction, not a cleaner factorization.
*Reported either way; failure of P4 does not affect P1–P3.*

## Analysis plan

Results reported as median and IQR across seeds per (hidden size, variable) — distributions, not trajectories, per series convention. Outputs: `multiseed_results.csv` (one row per seed × hidden size), `multiseed_summary.txt` (prediction pass/fail with counts), boxplots of R² per variable per hidden size, dropped-axis histogram at h=2, and an A_x vs A_y scatter at h=2.

No exclusions of seeds for any reason other than a failed run (crash), which will be reported. No post-hoc threshold adjustment: if the thresholds prove badly chosen, the registered result stands as stated and any re-analysis is labeled exploratory.

## What failure would mean

- P1 fails → bounded prediction does not reliably produce factorization at this scale; the two-ingredient claim loses its evidential core and Paper 0 becomes a negative result (which the series publishes).
- P2 fails → the variety-gap micro-foundation weakens: capacity limits blur rather than select. Paper VI's mechanism reading must be softened.
- P3 fails → the non-uniqueness demonstration moves back to being purely conceptual (exploration 02), losing its empirical miniature.
