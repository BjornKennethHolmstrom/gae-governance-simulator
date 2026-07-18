#!/usr/bin/env python3
"""
Governance as Engineering — The Cost of Returning: D2 Transplant
=================================================================

Purpose
-------
Extend Paper X's Experiment D2 strategy-selection machinery with an explicit
independent-channel competence state.

The earlier deterministic mean-field prototype turned out to be analytically
solvable. Its loop width is:

    W_H = 2 b^2 + sigma_I^2 (1/c_*^2 - 1/c_0^2) + L

where:
    b       = common shared-system bias relative to truth,
    c_0     = intact independent competence,
    c_*     = rho / (rho + delta), the latent-competence fixed point while shared,
    L       = liability-ratchet contribution at full consolidation.

That formula makes three benchmark predictions:

1. entry is invariant to competence decay in the endpoint mean-field model;
2. decay shifts the EXIT threshold by
       -sigma_I^2 (1/c_*^2 - 1/c_0^2);
3. decay and liability are exactly additive in that reduced model.

The present simulation moves to finite heterogeneous agents with stochastic
switching. It asks whether those benchmark predictions survive when exact
additivity is no longer algebraically forced.

What is inherited from Paper X D2
---------------------------------
- binary Independent / Shared strategies;
- consensus-relative evaluation rather than evaluation against truth;
- a shared system whose users make a common error;
- independent systems with decorrelated idiosyncratic errors;
- lower shared-system operating cost;
- an asymmetric liability penalty on independence that rises with the shared
  fraction;
- periodic stochastic strategy revision.

What is changed
---------------
- Paper X's fixed 0.002 switch-back probability is removed;
- every agent carries latent independent competence c_i;
- competence rebuilds while independent and decays while using the shared
  infrastructure;
- switch-back is determined endogenously by the perceived performance of the
  agent's retained competence;
- no explicit restart cost is added;
- no direct +gamma*f network-performance bonus is added.

This is a STRATEGY-LAYER transplant, not a rerun of Paper X's full hidden-state
environment and precautionary gate. The environment/failure layer is omitted
because the present question is entry, exit, and recovery from consolidation.

Finite-agent model
------------------
For seed s and agent i:

    strategy z_i ∈ {I,S}
    competence c_i ∈ (0,1]
    baseline independent error sigma_i

At shared fraction f:

    perceived shared error:
        E_S = (1-f)^2 b^2

    perceived independent error for agent i:
        E_I,i = f^2 b^2 + sigma_i^2 / c_i^2

Utilities:
    U_S   = -E_S   - C_S + theta
    U_I,i = -E_I,i - C_I - L0 - L1 f

theta is an externally swept shared-system advantage used only to reveal the
entry and exit branches.

Competence:
    independent:
        c <- c + eta_I (1-c)

    shared:
        c <- c + rho (1-c) - delta c

Factorial
---------
A: no competence decay, no liability ratchet
B: competence decay, no liability ratchet
C: no competence decay, liability ratchet
D: competence decay, liability ratchet

Primary tests [R within model]
------------------------------
R1 ANALYTIC BENCHMARK
   Median finite-agent thresholds remain near the mean-field formula.

R2 EXIT-LOCALIZED DECAY EFFECT
   Competence decay changes the exit threshold more than the entry threshold.

R3 DELTA/RHO LAW
   Across a frozen delta/rho sweep, observed exit shifts track the analytic
   prediction sigma^2[(1+delta/rho)^2 - 1].

OPEN R4 LIABILITY x DECAY INTERACTION
   In mean field the interaction is exactly zero. In the finite-agent model it
   is genuinely open. Report the paired difference-in-differences and CI.

R5 RECOVERY-REGION GAP
   In a two-dimensional map over lock duration and recovery pressure, maximal
   internal training under fixed rebuilding efficiency recovers a larger region
   than trainer-dependent rebuilding. The difference, not a single boundary
   point, is the second-order mechanism's contribution.

No paper claim follows automatically. If the agent model merely reproduces the
closed form, this is an analytic Paper X extension. Deviations — interaction,
entry movement, or a differently shaped recovery gap — are the candidate paper.

Dependencies
------------
Python 3.9+, NumPy, Matplotlib

Run
---
    python3 gae-sim-cost-of-return-d2-transplant.py

Expected runtime
----------------
Usually below one minute on a typical CPU.

Outputs
-------
    outputs_cost_of_return_d2/d2-branches.png
    outputs_cost_of_return_d2/d2-analytic-vs-agent.png
    outputs_cost_of_return_d2/d2-delta-rho.png
    outputs_cost_of_return_d2/d2-recovery-fixed.png
    outputs_cost_of_return_d2/d2-recovery-trainer.png
    outputs_cost_of_return_d2/d2-recovery-gap.png
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Frozen design
# ---------------------------------------------------------------------------

MASTER_SEED = 20260718

N_AGENTS = 20
N_BRANCH_SEEDS = 200
N_RATIO_SEEDS = 160
N_RECOVERY_SEEDS = 100

COMMON_SHARED_BIAS = 0.45
INDEPENDENT_ERROR_STD = 0.12
INDEPENDENT_ERROR_HETEROGENEITY = 0.15

COST_SHARED = 0.50
COST_INDEPENDENT = 1.00
LIABILITY_BASE = 0.20
LIABILITY_OFF = 0.0
LIABILITY_ON = 0.15

SELECTION_GAIN = 8.0
SWITCH_RATE = 0.25
MUTATION_RATE = 0.0005

INDEPENDENT_REBUILD_RATE = 0.05
PASSIVE_RETENTION = 0.003
DECAY_OFF = 0.0
DECAY_ON = 0.005

THETA_VALUES = np.linspace(-1.20, 0.20, 57)
DWELL_EVALUATIONS = 30
SHARED_THRESHOLD = 0.50

DELTA_RHO_RATIOS = np.array([0.0, 0.25, 0.50, 1.0, 1.5, 2.0, 3.0, 4.0])

# Recovery map
LOCK_THETA = 0.10
LOCK_DURATIONS = np.arange(0, 301, 30)
RECOVERY_THETA_VALUES = np.linspace(-1.20, -0.40, 17)
RECOVERY_HORIZON = 200
MAX_INTERNAL_TRAINING = 0.30
TRAINING_EFFECTIVENESS = 0.25
RECOVERY_SHARED_TARGET = 0.20
RECOVERY_COMPETENCE_TARGET = 0.70

TRAINER_REFERENCE_MASS = 0.10
MIN_TRAINER_MASS = 1.0 / N_AGENTS

# Paper-gate tolerances
ANALYTIC_THRESHOLD_TOLERANCE = 0.06
EXIT_LOCALIZATION_MARGIN = 0.04
MIN_RATIO_CORRELATION = 0.95
MIN_RECOVERY_GAP_AREA = 0.01

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_cost_of_return_d2"


@dataclass(frozen=True)
class Cell:
    code: str
    label: str
    decay: float
    liability_ratchet: float


CELLS = (
    Cell("A", "no decay / no ratchet", DECAY_OFF, LIABILITY_OFF),
    Cell("B", "decay / no ratchet", DECAY_ON, LIABILITY_OFF),
    Cell("C", "no decay / ratchet", DECAY_OFF, LIABILITY_ON),
    Cell("D", "decay / ratchet", DECAY_ON, LIABILITY_ON),
)


@dataclass
class BranchBatch:
    cell: Cell
    upward: np.ndarray
    downward: np.ndarray
    entry: np.ndarray
    exit: np.ndarray
    width: np.ndarray
    analytic_entry: float
    analytic_exit: float


@dataclass
class Population:
    strategy_shared: np.ndarray
    competence: np.ndarray
    sigma_independent: np.ndarray


def expected_sigma_squared() -> float:
    """
    sigma_i = sigma_0 exp(N(0,h^2)).
    Therefore E[sigma_i^2] = sigma_0^2 exp(2 h^2).
    """
    return float(
        INDEPENDENT_ERROR_STD**2
        * math.exp(2.0 * INDEPENDENT_ERROR_HETEROGENEITY**2)
    )


SIGMA2_EXPECTED = expected_sigma_squared()
COST_GAP = COST_INDEPENDENT + LIABILITY_BASE - COST_SHARED


def competence_fixed_point(decay: float) -> float:
    if decay <= 0.0:
        return 1.0
    return PASSIVE_RETENTION / (PASSIVE_RETENTION + decay)


def analytic_entry_threshold() -> float:
    return (
        COMMON_SHARED_BIAS**2
        - SIGMA2_EXPECTED
        - COST_GAP
    )


def analytic_exit_threshold(
    decay: float,
    liability_ratchet: float,
) -> float:
    c_star = competence_fixed_point(decay)
    return (
        -COMMON_SHARED_BIAS**2
        - SIGMA2_EXPECTED / (c_star**2)
        - COST_GAP
        - liability_ratchet
    )


def analytic_loop_width(
    decay: float,
    liability_ratchet: float,
) -> float:
    return (
        analytic_entry_threshold()
        - analytic_exit_threshold(decay, liability_ratchet)
    )


def initialize_population(
    n_seeds: int,
    seed: int,
) -> tuple[np.random.Generator, Population]:
    rng = np.random.default_rng(seed)
    sigma = INDEPENDENT_ERROR_STD * np.exp(
        rng.normal(
            0.0,
            INDEPENDENT_ERROR_HETEROGENEITY,
            size=(n_seeds, N_AGENTS),
        )
    )
    return rng, Population(
        strategy_shared=np.zeros((n_seeds, N_AGENTS), dtype=bool),
        competence=np.ones((n_seeds, N_AGENTS), dtype=float),
        sigma_independent=sigma,
    )


def trainer_multiplier(
    population: Population,
    rebuilding_law: Literal["none", "fixed", "trainer"],
) -> np.ndarray:
    n_seeds = population.strategy_shared.shape[0]

    if rebuilding_law == "none":
        return np.zeros(n_seeds)
    if rebuilding_law == "fixed":
        return np.ones(n_seeds)

    trainer_mass = np.mean(
        (~population.strategy_shared) * population.competence,
        axis=1,
    )
    return np.where(
        trainer_mass < MIN_TRAINER_MASS,
        0.0,
        np.minimum(1.0, trainer_mass / TRAINER_REFERENCE_MASS),
    )


def one_evaluation(
    population: Population,
    rng: np.random.Generator,
    theta: float,
    decay: float,
    liability_ratchet: float,
    training_allocation: float = 0.0,
    rebuilding_law: Literal["none", "fixed", "trainer"] = "none",
) -> None:
    """
    Mutates population in place for one strategy-evaluation period.
    """
    shared = population.strategy_shared
    competence = population.competence

    multiplier = trainer_multiplier(population, rebuilding_law)
    training_rate = (
        TRAINING_EFFECTIVENESS
        * training_allocation
        * multiplier[:, None]
    )

    competence[:] = np.where(
        shared,
        competence
        + PASSIVE_RETENTION * (1.0 - competence)
        - decay * competence
        + training_rate * (1.0 - competence),
        competence
        + (
            INDEPENDENT_REBUILD_RATE
            + training_rate
        )
        * (1.0 - competence),
    )
    np.clip(competence, 1e-3, 1.0, out=competence)

    shared_fraction = np.mean(shared, axis=1)
    shared_error = (
        (1.0 - shared_fraction)
        * COMMON_SHARED_BIAS
    ) ** 2

    independent_error = (
        (shared_fraction[:, None] * COMMON_SHARED_BIAS) ** 2
        + population.sigma_independent**2 / competence**2
    )

    utility_shared = (
        -shared_error
        - COST_SHARED
        + theta
    )
    utility_independent = (
        -independent_error
        - COST_INDEPENDENT
        - LIABILITY_BASE
        - liability_ratchet * shared_fraction[:, None]
    )

    switch_advantage = np.where(
        shared,
        utility_independent - utility_shared[:, None],
        utility_shared[:, None] - utility_independent,
    )

    switch_probability = (
        MUTATION_RATE
        + SWITCH_RATE
        * np.maximum(
            0.0,
            np.tanh(SELECTION_GAIN * switch_advantage),
        )
    )
    switch_probability = np.clip(switch_probability, 0.0, 1.0)

    switches = (
        rng.random(size=shared.shape)
        < switch_probability
    )
    shared[:] = np.where(switches, ~shared, shared)


def threshold_from_branch(branch: np.ndarray) -> np.ndarray:
    """
    Branch is shape (seed, theta), aligned in ascending theta order.
    Returns the first theta at which f >= SHARED_THRESHOLD.
    """
    mask = branch >= SHARED_THRESHOLD
    indices = np.argmax(mask, axis=1)
    exists = np.any(mask, axis=1)
    result = np.full(branch.shape[0], np.nan)
    result[exists] = THETA_VALUES[indices[exists]]
    return result


def simulate_branch_batch(cell: Cell) -> BranchBatch:
    # Resetting the seed across cells supplies common random numbers.
    rng, population = initialize_population(
        N_BRANCH_SEEDS,
        MASTER_SEED,
    )

    upward_points = []
    for theta in THETA_VALUES:
        for _ in range(DWELL_EVALUATIONS):
            one_evaluation(
                population,
                rng,
                float(theta),
                cell.decay,
                cell.liability_ratchet,
            )
        upward_points.append(
            np.mean(population.strategy_shared, axis=1)
        )

    downward_descending = []
    for theta in THETA_VALUES[::-1]:
        for _ in range(DWELL_EVALUATIONS):
            one_evaluation(
                population,
                rng,
                float(theta),
                cell.decay,
                cell.liability_ratchet,
            )
        downward_descending.append(
            np.mean(population.strategy_shared, axis=1)
        )

    upward = np.stack(upward_points, axis=1)
    downward = np.stack(
        downward_descending[::-1],
        axis=1,
    )
    entry = threshold_from_branch(upward)
    exit_ = threshold_from_branch(downward)
    width = entry - exit_

    return BranchBatch(
        cell=cell,
        upward=upward,
        downward=downward,
        entry=entry,
        exit=exit_,
        width=width,
        analytic_entry=analytic_entry_threshold(),
        analytic_exit=analytic_exit_threshold(
            cell.decay,
            cell.liability_ratchet,
        ),
    )


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    mean = float(np.mean(finite))
    if len(finite) < 2:
        return mean, 0.0
    se = float(np.std(finite, ddof=1) / math.sqrt(len(finite)))
    return mean, 1.96 * se


def run_delta_rho_sweep() -> dict[str, np.ndarray]:
    observed_exit = []
    observed_shift = []
    predicted_exit = []
    predicted_shift = []

    no_decay_exit: np.ndarray | None = None
    no_decay_prediction: float | None = None

    for ratio_index, ratio in enumerate(DELTA_RHO_RATIOS):
        decay = float(ratio * PASSIVE_RETENTION)
        cell = Cell(
            code=f"R{ratio_index}",
            label=f"delta/rho={ratio:.2f}",
            decay=decay,
            liability_ratchet=LIABILITY_OFF,
        )

        # A smaller but still paired batch for the ratio sweep.
        original_seed_count = globals()["N_BRANCH_SEEDS"]
        globals()["N_BRANCH_SEEDS"] = N_RATIO_SEEDS
        try:
            batch = simulate_branch_batch(cell)
        finally:
            globals()["N_BRANCH_SEEDS"] = original_seed_count

        median_exit = float(np.nanmedian(batch.exit))
        prediction = analytic_exit_threshold(decay, LIABILITY_OFF)

        if no_decay_exit is None:
            no_decay_exit = batch.exit.copy()
            no_decay_prediction = prediction

        observed_exit.append(median_exit)
        observed_shift.append(
            float(np.nanmedian(batch.exit - no_decay_exit))
        )
        predicted_exit.append(prediction)
        predicted_shift.append(prediction - float(no_decay_prediction))

    return {
        "ratio": DELTA_RHO_RATIOS.copy(),
        "observed_exit": np.asarray(observed_exit),
        "observed_shift": np.asarray(observed_shift),
        "predicted_exit": np.asarray(predicted_exit),
        "predicted_shift": np.asarray(predicted_shift),
    }


def snapshot_lock_states() -> tuple[
    np.ndarray,
    dict[int, tuple[np.ndarray, np.ndarray]],
]:
    """
    One paired lock history, reused for every recovery theta and rebuilding law.
    Returns the fixed sigma array and snapshots of strategy/competence.
    """
    rng, population = initialize_population(
        N_RECOVERY_SEEDS,
        MASTER_SEED + 50_000,
    )
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {
        0: (
            population.strategy_shared.copy(),
            population.competence.copy(),
        )
    }

    maximum_lock = int(np.max(LOCK_DURATIONS))
    for time_index in range(1, maximum_lock + 1):
        one_evaluation(
            population,
            rng,
            LOCK_THETA,
            DECAY_ON,
            LIABILITY_ON,
        )
        if time_index in LOCK_DURATIONS:
            snapshots[time_index] = (
                population.strategy_shared.copy(),
                population.competence.copy(),
            )

    return population.sigma_independent.copy(), snapshots


def recovery_success(population: Population) -> np.ndarray:
    shared_fraction = np.mean(
        population.strategy_shared,
        axis=1,
    )
    independent_mask = ~population.strategy_shared
    independent_count = np.sum(independent_mask, axis=1)
    independent_competence = np.divide(
        np.sum(
            independent_mask * population.competence,
            axis=1,
        ),
        np.maximum(independent_count, 1),
    )

    return (
        (shared_fraction <= RECOVERY_SHARED_TARGET)
        & (
            independent_competence
            >= RECOVERY_COMPETENCE_TARGET
        )
    )


def recovery_map(
    rebuilding_law: Literal["fixed", "trainer"],
    sigma_independent: np.ndarray,
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    result = np.zeros(
        (
            len(LOCK_DURATIONS),
            len(RECOVERY_THETA_VALUES),
        ),
        dtype=float,
    )

    for lock_index, lock_duration in enumerate(LOCK_DURATIONS):
        strategy_snapshot, competence_snapshot = snapshots[
            int(lock_duration)
        ]

        for theta_index, theta_recovery in enumerate(
            RECOVERY_THETA_VALUES
        ):
            rng = np.random.default_rng(
                MASTER_SEED
                + 100_000
                + 1_000 * lock_index
                + theta_index
            )
            population = Population(
                strategy_shared=strategy_snapshot.copy(),
                competence=competence_snapshot.copy(),
                sigma_independent=sigma_independent,
            )

            for _ in range(RECOVERY_HORIZON):
                one_evaluation(
                    population,
                    rng,
                    float(theta_recovery),
                    DECAY_ON,
                    LIABILITY_ON,
                    training_allocation=MAX_INTERNAL_TRAINING,
                    rebuilding_law=rebuilding_law,
                )

            result[lock_index, theta_index] = float(
                np.mean(recovery_success(population))
            )

    return result


def spearman_rank_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    No SciPy dependency. Values in this frozen sweep are expected to be unique.
    """
    rank_x = np.argsort(np.argsort(x)).astype(float)
    rank_y = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    filename: str,
    colorbar_label: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=(
            RECOVERY_THETA_VALUES[0],
            RECOVERY_THETA_VALUES[-1],
            LOCK_DURATIONS[0],
            LOCK_DURATIONS[-1],
        ),
        vmin=0.0 if np.min(matrix) >= 0 else None,
        vmax=1.0 if np.max(matrix) <= 1 else None,
    )
    ax.set_xlabel("recovery shared-system advantage theta")
    ax.set_ylabel("time held in consolidated regime")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    batches = {
        cell.code: simulate_branch_batch(cell)
        for cell in CELLS
    }

    ratio_results = run_delta_rho_sweep()

    sigma_recovery, lock_snapshots = snapshot_lock_states()
    fixed_map = recovery_map(
        "fixed",
        sigma_recovery,
        lock_snapshots,
    )
    trainer_map = recovery_map(
        "trainer",
        sigma_recovery,
        lock_snapshots,
    )
    recovery_gap = fixed_map - trainer_map

    # ------------------------------------------------------------------
    # Contrasts and registered checks
    # ------------------------------------------------------------------

    median_entry = {
        code: float(np.nanmedian(batch.entry))
        for code, batch in batches.items()
    }
    median_exit = {
        code: float(np.nanmedian(batch.exit))
        for code, batch in batches.items()
    }

    benchmark_deviations = []
    for batch in batches.values():
        benchmark_deviations.extend(
            [
                abs(
                    float(np.nanmedian(batch.entry))
                    - batch.analytic_entry
                ),
                abs(
                    float(np.nanmedian(batch.exit))
                    - batch.analytic_exit
                ),
            ]
        )
    r1 = max(benchmark_deviations) <= ANALYTIC_THRESHOLD_TOLERANCE

    entry_shift_no_ratchet = (
        median_entry["B"] - median_entry["A"]
    )
    exit_shift_no_ratchet = (
        median_exit["B"] - median_exit["A"]
    )
    entry_shift_ratchet = (
        median_entry["D"] - median_entry["C"]
    )
    exit_shift_ratchet = (
        median_exit["D"] - median_exit["C"]
    )

    r2 = (
        abs(exit_shift_no_ratchet)
        >= abs(entry_shift_no_ratchet)
        + EXIT_LOCALIZATION_MARGIN
        and abs(exit_shift_ratchet)
        >= abs(entry_shift_ratchet)
        + EXIT_LOCALIZATION_MARGIN
    )

    ratio_correlation = spearman_rank_correlation(
        ratio_results["predicted_shift"],
        ratio_results["observed_shift"],
    )
    r3 = ratio_correlation >= MIN_RATIO_CORRELATION

    interaction_per_seed = (
        batches["D"].width
        - batches["B"].width
        - batches["C"].width
        + batches["A"].width
    )
    interaction_mean, interaction_half_ci = mean_ci(
        interaction_per_seed
    )

    recovery_gap_area = float(np.mean(recovery_gap))
    recoverable_fixed_area = float(np.mean(fixed_map >= 0.50))
    recoverable_trainer_area = float(
        np.mean(trainer_map >= 0.50)
    )
    r5 = recovery_gap_area >= MIN_RECOVERY_GAP_AREA

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    print("=" * 100)
    print("THE COST OF RETURNING — PAPER X D2 STRATEGY-LAYER TRANSPLANT")
    print("=" * 100)
    print(f"master seed                         : {MASTER_SEED}")
    print(f"agents per population               : {N_AGENTS}")
    print(f"paired branch seeds                 : {N_BRANCH_SEEDS}")
    print(f"theta points                        : {len(THETA_VALUES)}")
    print(f"dwell evaluations per theta         : {DWELL_EVALUATIONS}")
    print(f"E[sigma_i^2]                        : {SIGMA2_EXPECTED:.6f}")
    print(f"mean-field theta_entry              : {analytic_entry_threshold():.4f}")
    print()

    print("ANALYTIC BENCHMARK VS FINITE AGENTS")
    print(
        f"{'cell':<6}"
        f"{'condition':<27}"
        f"{'entry obs':>11}"
        f"{'entry pred':>12}"
        f"{'exit obs':>11}"
        f"{'exit pred':>11}"
        f"{'width obs':>12}"
        f"{'width pred':>12}"
    )
    print("-" * 102)
    for cell in CELLS:
        batch = batches[cell.code]
        observed_width = float(np.nanmedian(batch.width))
        print(
            f"{cell.code:<6}"
            f"{cell.label:<27}"
            f"{np.nanmedian(batch.entry):11.3f}"
            f"{batch.analytic_entry:12.3f}"
            f"{np.nanmedian(batch.exit):11.3f}"
            f"{batch.analytic_exit:11.3f}"
            f"{observed_width:12.3f}"
            f"{analytic_loop_width(cell.decay, cell.liability_ratchet):12.3f}"
        )

    print()
    print("DECAY LOCALIZATION")
    print(
        f"entry shift B-A / exit shift B-A   : "
        f"{entry_shift_no_ratchet:+.3f} / {exit_shift_no_ratchet:+.3f}"
    )
    print(
        f"entry shift D-C / exit shift D-C   : "
        f"{entry_shift_ratchet:+.3f} / {exit_shift_ratchet:+.3f}"
    )

    print()
    print("OPEN LIABILITY x DECAY INTERACTION")
    print(
        f"I_H                                 : "
        f"{interaction_mean:+.5f} +/- {interaction_half_ci:.5f} (95% CI)"
    )
    print(
        "Mean-field prediction               : exactly 0 by algebraic additivity"
    )

    print()
    print("DELTA/RHO SWEEP")
    print(
        f"{'delta/rho':>10}"
        f"{'exit observed':>16}"
        f"{'exit predicted':>17}"
        f"{'shift observed':>17}"
        f"{'shift predicted':>18}"
    )
    for index, ratio in enumerate(ratio_results["ratio"]):
        print(
            f"{ratio:10.2f}"
            f"{ratio_results['observed_exit'][index]:16.3f}"
            f"{ratio_results['predicted_exit'][index]:17.3f}"
            f"{ratio_results['observed_shift'][index]:17.3f}"
            f"{ratio_results['predicted_shift'][index]:18.3f}"
        )
    print(
        f"Spearman prediction/observation     : {ratio_correlation:.3f}"
    )

    print()
    print("RECOVERY REGION")
    print(
        f"fixed-law recoverable map fraction  : {recoverable_fixed_area:.3f}"
    )
    print(
        f"trainer-law recoverable fraction    : {recoverable_trainer_area:.3f}"
    )
    print(
        f"mean probability gap                : {recovery_gap_area:.3f}"
    )

    print()
    print("PAPER-GATE CHECKS")
    print(
        f"R1 analytic thresholds approximate agents: "
        f"{'PASS' if r1 else 'FAIL'} "
        f"(max deviation={max(benchmark_deviations):.3f})"
    )
    print(
        f"R2 decay acts mainly on exit           : "
        f"{'PASS' if r2 else 'FAIL'}"
    )
    print(
        f"R3 delta/rho law tracks exit shift     : "
        f"{'PASS' if r3 else 'FAIL'}"
    )
    print(
        f"OPEN R4 finite-agent interaction       : "
        f"{interaction_mean:+.5f} +/- {interaction_half_ci:.5f}"
    )
    print(
        f"R5 trainer ecology shrinks recovery set: "
        f"{'PASS' if r5 else 'FAIL'}"
    )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Median branches
    fig, ax = plt.subplots(figsize=(10.3, 6.0))
    for cell in CELLS:
        batch = batches[cell.code]
        ax.plot(
            THETA_VALUES,
            np.median(batch.upward, axis=0),
            label=f"{cell.code} upward",
        )
        ax.plot(
            THETA_VALUES,
            np.median(batch.downward, axis=0),
            linestyle="--",
            label=f"{cell.code} downward",
        )
    ax.axhline(SHARED_THRESHOLD, linestyle=":")
    ax.set_xlabel("shared-system advantage theta")
    ax.set_ylabel("median shared-system fraction")
    ax.set_title(
        "Finite-agent branches: decay primarily displaces the return threshold"
    )
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "d2-branches.png",
        dpi=170,
    )
    plt.close(fig)

    # 2. Analytic vs observed thresholds
    positions = np.arange(len(CELLS))
    observed_entry = [
        np.nanmedian(batches[cell.code].entry)
        for cell in CELLS
    ]
    predicted_entry = [
        batches[cell.code].analytic_entry
        for cell in CELLS
    ]
    observed_exit = [
        np.nanmedian(batches[cell.code].exit)
        for cell in CELLS
    ]
    predicted_exit = [
        batches[cell.code].analytic_exit
        for cell in CELLS
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(
        positions,
        observed_entry,
        marker="o",
        label="entry observed",
    )
    ax.plot(
        positions,
        predicted_entry,
        marker="s",
        linestyle="--",
        label="entry analytic",
    )
    ax.plot(
        positions,
        observed_exit,
        marker="o",
        label="exit observed",
    )
    ax.plot(
        positions,
        predicted_exit,
        marker="s",
        linestyle="--",
        label="exit analytic",
    )
    ax.set_xticks(
        positions,
        [cell.code for cell in CELLS],
    )
    ax.set_xlabel("factorial cell")
    ax.set_ylabel("theta threshold")
    ax.set_title(
        "Closed-form benchmark versus finite heterogeneous agents"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "d2-analytic-vs-agent.png",
        dpi=170,
    )
    plt.close(fig)

    # 3. delta/rho law
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(
        ratio_results["ratio"],
        ratio_results["observed_shift"],
        marker="o",
        label="finite-agent exit shift",
    )
    ax.plot(
        ratio_results["ratio"],
        ratio_results["predicted_shift"],
        marker="s",
        linestyle="--",
        label="analytic exit shift",
    )
    ax.set_xlabel("competence-decay ratio delta/rho")
    ax.set_ylabel("exit-threshold shift from no decay")
    ax.set_title(
        "The analytic decay law predicts the finite-agent return penalty"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "d2-delta-rho.png",
        dpi=170,
    )
    plt.close(fig)

    plot_heatmap(
        fixed_map,
        "Recovery probability with fixed rebuilding efficiency",
        "d2-recovery-fixed.png",
        "P(recovered)",
    )
    plot_heatmap(
        trainer_map,
        "Recovery probability when surviving channels must train replacements",
        "d2-recovery-trainer.png",
        "P(recovered)",
    )
    plot_heatmap(
        recovery_gap,
        "Recovery-set gap: fixed efficiency minus trainer-dependent rebuilding",
        "d2-recovery-gap.png",
        "probability difference",
    )

    print()
    print("Saved:")
    for filename in (
        "d2-branches.png",
        "d2-analytic-vs-agent.png",
        "d2-delta-rho.png",
        "d2-recovery-fixed.png",
        "d2-recovery-trainer.png",
        "d2-recovery-gap.png",
    ):
        print(f"  {OUTPUT_DIR / filename}")

    if not all((r1, r2, r3, r5)):
        raise SystemExit(
            "\nOne or more paper-gate checks failed. "
            "Treat this as a scientific result, not a software error."
        )


if __name__ == "__main__":
    main()
