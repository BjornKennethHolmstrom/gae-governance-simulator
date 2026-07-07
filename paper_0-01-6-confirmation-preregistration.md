# Preregistration addendum — Confirmation run (seeds 20–39)

**Status:** Written after analyzing seeds 0–19 (`01-5-multiseed-results.md`), before running seeds 20–39. The exploratory findings of the first run are hypotheses here; this run decides whether they graduate to registered evidence.

**Protocol:** `01-7-confirmation_run.py`. Identical environment, model, training, and probe configuration to `01-3-multiseed-preregistration.md`, except: seeds 20–39 (20 fresh seeds), hidden sizes **{2, 3}** only. All definitions carry over (negative R² clipped to 0; axis metric A_x, A_y as before). New registered definition:

- **Type metric:** A_pos = (R²_x + R²_y)/2, A_vel = (R²_vx + R²_vy)/2, type-asym = |A_pos − A_vel|.
- **Mode classification at h=2:** if max(axis-asym, type-asym) ≤ 0.3, *uniform*; else the mode is whichever asymmetry is larger (*larger-asymmetry-wins*). This label is descriptive only. The registered P2′ and P2″ decisions use the raw metrics (type-asym and max(axis-asym, type-asym) respectively), not the label, so a seat with both asymmetries above threshold does not distort them.

## Registered predictions

**P2′ — Type-structured blindness (h=2, primary).** Type-asym > 0.3 in ≥ 80% of seeds (≥ 16/20).
*Null committed to:* < 60% ⇒ the static-observer finding of run 1 was itself unstable, and Paper 0 reports structured blindness as mode-dependent with no dominant mode.

**P2″ — No uniform blur (h=2, primary).** max(axis-asym, type-asym) > 0.3 in ≥ 90% of seeds (≥ 18/20). This is the general form of the structured-blindness claim: some coherent causal subspace is always sacrificed; degradation is never even.
*Null committed to:* ≥ 4 uniform seeds ⇒ the "capacity selects dimensions" mechanism claim is demoted to a frequent-but-not-lawlike outcome.

**P3′ — Forced tie-break at h=3 (primary).** h=3 tests whether a one-dimension capacity increment above the position-only regime produces a *discrete* velocity-selection pattern (one velocity recovered, the other not) rather than an even split. A hidden size of 3 is not literally three clean scalar slots — a GRU state can hold mixed nonlinear combinations — so the question is settled empirically by the linear probes below, not by assuming slot structure. Registered claims, evaluated on seeds 20–39 at h=3:
1. Both positions recovered: median R²_x ≥ 0.7 and median R²_y ≥ 0.7.
2. Velocity selection is asymmetric within-seed: |R²_vx − R²_vy| > 0.25 in ≥ 60% of seeds ("choosing seeds").
3. The favored velocity varies across seeds: among choosing seeds, each of vx and vy is the favored one in ≥ 3 seeds.

*Nulls committed to:* (2) fails ⇒ capacity splits evenly between velocities and the tie-break framing is wrong (blindness need not be discrete at this granularity). (3) fails with one velocity favored in ≥ 90% of choosing seeds ⇒ a hidden x/y asymmetry exists in the environment or rendering; the symmetry-breaking interpretation is withdrawn pending diagnosis of the code.

**Nesting note (carried from run 1):** P2′/P2″ and P3′ are logically nested but not identical. P2′/P2″ ask whether starvation usually becomes *structured*. P3′ presupposes structure and asks whether the *selected* blind variable varies across seeds rather than reflecting a hidden x/y bias.

## Exploratory (declared, not registered)

- Axis-mode frequency at h=2, pooled over seeds 0–39, with a tally of which axis is dropped. Reported descriptively; no threshold.
- Loss ordering between modes at h=2 (run 1 found type mode strictly better); reported descriptively.

## Analysis plan

As in run 1: medians and IQRs across seeds; per-seed table at each hidden size; pass/fail against the thresholds above written to `confirmation_summary.txt`; no exclusions except crashed runs (reported); no post-hoc threshold adjustment — misjudged thresholds stand as registered and any re-analysis is labeled exploratory.

## What failure would mean

- P2′ or P2″ fails ⇒ Section 3 of Paper 0 reports structured blindness at [Exploratory] tier only, and the variety-gap micro-foundation is presented as suggestive rather than demonstrated.
- P3′(3) fails via the ≥90% route ⇒ the code is audited before any further runs; all symmetry claims are suspended.
- P3′(2) fails ⇒ the "factorization is discrete under starvation" framing is dropped; blindness may be graded at fine capacity margins, which Paper 0 must state.
