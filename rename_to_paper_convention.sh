#!/usr/bin/env bash
#
# Rename the Governance-as-Engineering simulator files from the old chronological
# vN naming to the paper_<roman>_<slug> convention, matching the already-migrated
# paper_*/self_ii_*/self_iii_* files. Preserves git history via `git mv`.
#
# Run from the repository root:  bash rename_to_paper_convention.sh
#
set -euo pipefail

# ---- Main governance series --------------------------------------------------
git mv gae-simulator-v1.py                                  prototype_conceptual_sketch.py
git mv gae-simulator-v2.py                                  paper_i_single_node_feedback.py
git mv gae-simulator-v3.py                                  paper_i_multinode_subsidiarity.py
git mv gae-simulator-v3-unadjusted.py                       paper_i_multinode_unadjusted.py
git mv gae-simulator-v4.py                                  paper_ii_fractal_multiscale.py
git mv gae-simulator-v5.py                                  paper_iii_representation_observability.py
git mv gae-simulator-v6.py                                  paper_iv_commons_requisite_variety.py
git mv gae-simulator-v7.py                                  paper_vi_value_function_collapse.py
git mv gae-simulator-v8-bypass-trap.py                      paper_ix_bypass_trap.py
git mv gae-simulator-v9-latency-assymetry.py               paper_ix_latency_asymmetry.py   # typo fixed: assymetry -> asymmetry
git mv gae-simulator-v10-bandwidth-race.py                  paper_ix_bandwidth_race.py
git mv gae-simulator-v11-epistemic-monoculture.py          paper_x_epistemic_monoculture.py
git mv gae-simulator-v12-consolidation-dynamics.py         paper_x_consolidation_dynamics.py
git mv gae-simulator-v13-chain-prototype.py                paper_xi_chain_prototype.py
git mv gae-simulator-v14-boundary-mismatch.py              paper_xii_boundary_mismatch.py
git mv gae-simulator-v15-legitimacy-trap.py               paper_xiii_legitimacy_trap.py
git mv gae-simulator-v16-governance-as-adaptive-controller.py  paper_xiv_adaptive_controller.py
git mv gae-simulator-v17-adaptation-bottleneck.py          paper_xv_adaptation_bottleneck.py

# ---- Empirical study ---------------------------------------------------------
git mv gae-study1-observer-correlation.py                   study_i_observer_correlation.py

# ---- Self series (bring the one straggler into line) -------------------------
git mv self-stability-simulator.py                          self_i_variety_gap.py

# Files already following the convention and left untouched:
#   paper_x_echo_adversarial_fragility.py
#   paper_xiv_sunset.py
#   paper_xvi_replenishment_depletion.py
#   paper_xvi_replenishment_depletion_microfounded.py
#   paper_xvi_preferential_attachment_fold.py
#   paper_xvi_switching_barrier_fold.py
#   paper_xvi_protection_class.py
#   paper_xviii_boundary_instability.py
#   self_ii_appendix_a_correlation_tax.py ... _e_observer_plant.py
#   self_iii_operator.py
#   self_iii_formation.py

echo "Done. Review with 'git status' before committing."
