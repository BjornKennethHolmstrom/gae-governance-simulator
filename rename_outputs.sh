#!/usr/bin/env bash
#
# Rename output figures/CSVs to mirror their script names, and rewrite the
# filename string literals inside every script to match. Handles all three
# save patterns in the repo:  'outputs/x.png' ,  os.path.join(OUT,"x.png") ,
# and  f"{OUTDIR}/x.png"  — the replace keys off the bare filename, so it hits
# all of them.
#
# Run from the repository root, AFTER rename_to_paper_convention.sh:
#   bash rename_outputs.sh
#
# Edit any right-hand name below before running if you prefer different panel
# suffixes. Also fixes the stray 'ggf-simulator-v3-unadjusted.png' output name.
set -euo pipefail

# old_basename  ->  new_basename
MAP=(
  "gae-simulator-v2.png|paper_i_single_node_feedback.png"
  "gae-simulator-v3.png|paper_i_multinode_subsidiarity.png"
  "ggf-simulator-v3-unadjusted.png|paper_i_multinode_unadjusted.png"
  "gae-simulator-v4.png|paper_ii_fractal_multiscale.png"
  "gae-simulator-v5.png|paper_iii_representation_observability.png"
  "gae-simulator-v6.png|paper_iv_commons_requisite_variety.png"
  "appendix-c-simulation.png|paper_vi_value_function_collapse.png"
  "v8-bypass-trap-main.png|paper_ix_bypass_trap_main.png"
  "v8-bypass-trap-sweep.png|paper_ix_bypass_trap_sweep.png"
  "v9-latency-main.png|paper_ix_latency_asymmetry_main.png"
  "v9-latency-sweep.png|paper_ix_latency_asymmetry_sweep.png"
  "v10-bandwidth-main.png|paper_ix_bandwidth_race_main.png"
  "v10-bandwidth-sweep.png|paper_ix_bandwidth_race_sweep.png"
  "v11-monoculture-trajectories.png|paper_x_epistemic_monoculture_trajectories.png"
  "v11-monoculture-spread.png|paper_x_epistemic_monoculture_spread.png"
  "v11-monoculture-phase.png|paper_x_epistemic_monoculture_phase.png"
  "v11-monoculture-sweep.png|paper_x_epistemic_monoculture_sweep.png"
  "v12-consolidation-main.png|paper_x_consolidation_dynamics_main.png"
  "v12-consolidation-sweep.png|paper_x_consolidation_dynamics_sweep.png"
  "echo_adversarial_fragility.png|paper_x_echo_adversarial_fragility.png"
  "v14-adaptive-trajectory.png|paper_xii_boundary_mismatch_adaptive_trajectory.png"
  "v14-adaptive-sweep.png|paper_xii_boundary_mismatch_adaptive_sweep.png"
  "v14-stability-loopgain.png|paper_xii_boundary_mismatch_stability_loopgain.png"
  "v15-phase-diagram.png|paper_xiii_legitimacy_trap_phase_diagram.png"
  "v15-trap-and-recovery.png|paper_xiii_legitimacy_trap_trap_and_recovery.png"
  "v15-borrowed-collapse.png|paper_xiii_legitimacy_trap_borrowed_collapse.png"
  "v15-collapse-heatmap.png|paper_xiii_legitimacy_trap_collapse_heatmap.png"
  "v15-asymmetry-sweep.png|paper_xiii_legitimacy_trap_asymmetry_sweep.png"
  "v16-phase-diagram.png|paper_xiv_adaptive_controller_phase_diagram.png"
  "v16-starvation-vs-optimal.png|paper_xiv_adaptive_controller_starvation_vs_optimal.png"
  "v16-exploitation-lockin.png|paper_xiv_adaptive_controller_exploitation_lockin.png"
  "v16-forgetting-sweep.png|paper_xiv_adaptive_controller_forgetting_sweep.png"
  "v16-summary-metrics.csv|paper_xiv_adaptive_controller_summary_metrics.csv"
  "sunset_time_to_removal.png|paper_xiv_sunset_time_to_removal.png"
  "sunset_tradeoff.png|paper_xiv_sunset_tradeoff.png"
  "xv_A_allocation.png|paper_xv_adaptation_bottleneck_A_allocation.png"
  "xv_B_backlogs.png|paper_xv_adaptation_bottleneck_B_backlogs.png"
  "xv_C_closure_delay.png|paper_xv_adaptation_bottleneck_C_closure_delay.png"
  "xv_D_self_blinding.png|paper_xv_adaptation_bottleneck_D_self_blinding.png"
  "preferential_attachment_fold.png|paper_xvi_preferential_attachment_fold.png"
  "replenishment_depletion_microfounded.png|paper_xvi_replenishment_depletion_microfounded.png"
  "replenishment_depletion.png|paper_xvi_replenishment_depletion.png"
  "switching_barrier_fold.png|paper_xvi_switching_barrier_fold.png"
  "xviii_A2_regime_map.png|paper_xviii_boundary_instability_A2_regime_map.png"
  "xviii_A_phase_cycle.png|paper_xviii_boundary_instability_A_phase_cycle.png"
  "xviii_B_early_warning.png|paper_xviii_boundary_instability_B_early_warning.png"
  "xviii_C2_window_map.png|paper_xviii_boundary_instability_C2_window_map.png"
  "xviii_C_bandwidth_slice.png|paper_xviii_boundary_instability_C_bandwidth_slice.png"
  "self-stability-simulator.png|self_i_variety_gap.png"
  "study1-demo-correlation-tax.png|study_i_observer_correlation.png"
  "self3-phi-sweep.png|self_iii_operator_phi_sweep.png"
  "self3-legitimacy-trajectories.png|self_iii_operator_legitimacy_trajectories.png"
  "self3-interior-and-gap.png|self_iii_operator_interior_and_gap.png"
)
# Note: 'paper_xvi_protection_class.png' already matches the convention — left as-is.

for pair in "${MAP[@]}"; do
  old="${pair%%|*}"
  new="${pair##*|}"

  # 1) move the rendered file if it exists (keeps git history)
  if [ -f "outputs/$old" ]; then
    git mv -f "outputs/$old" "outputs/$new"
  fi

  # 2) rewrite the literal inside the scripts (dots escaped so '.' is literal)
  esc_old="${old//./\\.}"
  sed -i "s|${esc_old}|${new}|g" *.py
done

echo "Done. Review with 'git status' and a quick 'git diff' on the .py files."
echo "Then re-run any sims whose figures were NOT on disk yet (e.g. self_iii_operator.py)."
