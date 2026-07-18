#!/usr/bin/env python3
"""
Governance as Engineering — The Cost of Returning, Cycle 3
==========================================================

TAIL-NUCLEATED REVERSIBILITY IN A FINITE EPISTEMIC POPULATION

This cycle follows the analytic mean-field result and the first Paper X D2
strategy-layer transplant.

The mean-field benchmark
------------------------
At the consolidated endpoint, the reduced model predicts:

    theta_exit_mean =
        -b^2
        - E[sigma_i^2] / c_*^2
        - (C_I + L0 - C_S)
        - L1

with:

    c_* = rho / (rho + delta)

and loop width:

    W_H =
        2 b^2
        + E[sigma_i^2] (1/c_*^2 - 1/c_0^2)
        + L1

The first agent simulation exited systematically earlier / shallower than this
mean-field prediction, with the discrepancy growing as delta/rho increased.

Candidate mechanism: exit is nucleated by the best-preserved channel
--------------------------------------------------------------------
At f approximately 1, the first profitable independent defection is made by the
agent with the smallest retained independence penalty:

    p_min = min_i sigma_i^2 / c_i^2

This yields an order-statistic prediction:

    theta_exit_tail =
        -b^2
        - p_min
        - (C_I + L0 - C_S)
        - L1

A single unusually capable channel can nucleate a cascade. Reversibility is
therefore potentially controlled by the favorable tail of retained competence,
not by the population mean.

This cycle tests that candidate rather than treating the structured residual as
numerical error.

Studies
-------
1. Main 2 x 2 factorial with continuous threshold interpolation.
2. Mean-field versus realized order-statistic exit prediction.
3. delta/rho sweep.
4. population-size N sweep at fixed delta/rho.
5. dwell-time sweep to expose first-passage / metastability dependence.
6. recovery maps with:
       a. fixed rebuilding efficiency;
       b. trainer-dependent subsidy only (main second-order arm);
       c. fully trainer-gated rebuilding (strong critical-mass ablation).

The full-gating arm is NOT treated as the baseline. It deliberately assumes
that surviving competent channels are required even for natural competence
rebuilding, and is included only to show what additional assumption is required
for a strict internal absorbing region.

Design-frozen predictions [R within model]
-------------------------------------------
P1 ORDER-STATISTIC CORRECTION
   The realized-tail prediction has lower exit-threshold MAE than the
   mean-field prediction in the main factorial.

P2 CONTINUOUS INTERACTION
   The absolute liability x decay interaction estimated from linearly
   interpolated per-seed crossings is smaller than the coarse-grid estimate.
   If not, the apparent subadditivity survives the quantization challenge.

P3 DELTA/RHO CLOSURE
   Across the frozen delta/rho sweep, the order-statistic prediction has lower
   RMSE than the mean-field prediction.

P4 N DEPENDENCE
   Larger populations exit at shallower theta because the favorable order
   statistic improves. The order-statistic prediction captures the direction
   and reduces RMSE relative to the mean-field prediction.

P5 DWELL DEPENDENCE
   Median exit becomes shallower as dwell increases from 30 to 100 to 300
   evaluation periods, confirming that the measured exit is a first-passage
   quantity on a frozen metastable timescale rather than an infinite-time
   equilibrium spinodal.

P6 RECOVERY BOUNDARY BAND
   The trainer-dependent-subsidy law produces a positive recovery-probability
   gap concentrated in the fixed-law transition band (defined independently as
   0.2 <= P_fixed <= 0.8). Report the gap by lock duration rather than only as
   a whole-map average.

OPEN P7 FULL-GATING ABLATION
   Determine whether the stronger trainer-gated-natural-rebuilding assumption
   creates a strict internally unrecoverable region. Any such result is
   conditional on that explicit critical-mass assumption.

Interpretation gate
-------------------
If the tail correction closes the structured exit residual across delta/rho and
N, the candidate independent contribution is:

    the reversibility of epistemic monoculture is governed by the favorable
    tail of retained independent capability, not by its mean.

If it does not, the work returns to Paper X as an analytic appendix plus a
finite-agent robustness check.

Dependencies
------------
Python 3.9+, NumPy, Matplotlib

Run
---
    python3 gae-sim-cost-of-return-tail-cycle.py

Expected runtime
----------------
Roughly 1–3 minutes on a typical CPU.

Outputs
-------
    outputs_cost_of_return_tail/tail-main-branches.png
    outputs_cost_of_return_tail/tail-prediction-errors.png
    outputs_cost_of_return_tail/tail-delta-rho.png
    outputs_cost_of_return_tail/tail-N-sweep.png
    outputs_cost_of_return_tail/tail-dwell-sweep.png
    outputs_cost_of_return_tail/tail-interaction.png
    outputs_cost_of_return_tail/recovery-fixed.png
    outputs_cost_of_return_tail/recovery-subsidy-gated.png
    outputs_cost_of_return_tail/recovery-full-gated.png
    outputs_cost_of_return_tail/recovery-boundary-gap.png
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Frozen design
# ---------------------------------------------------------------------------

MASTER_SEED = 20260718

# Main model
N_MAIN = 20
N_MAIN_SEEDS = 240
N_SWEEP_SEEDS = 140
N_DWELL_SEEDS = 140
N_RECOVERY_SEEDS = 200

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

THETA_VALUES = np.linspace(-1.35, 0.20, 63)  # step 0.025
DWELL_MAIN = 30
DWELL_VALUES = np.array([30, 100, 300])
SHARED_THRESHOLD = 0.50

DELTA_RHO_RATIOS = np.array([0.0, 0.25, 0.50, 1.0, 1.5, 2.0, 3.0, 4.0])
N_VALUES = np.array([10, 20, 50, 100])
N_SWEEP_RATIO = 2.0

# Recovery map
LOCK_THETA = 0.10
LOCK_DURATIONS = np.arange(0, 301, 30)
RECOVERY_THETA_VALUES = np.linspace(-1.30, -0.40, 37)
RECOVERY_HORIZON = 220
MAX_INTERNAL_TRAINING = 0.30
TRAINING_EFFECTIVENESS = 0.25
RECOVERY_SHARED_TARGET = 0.20
RECOVERY_COMPETENCE_TARGET = 0.70

# Critical trainer mass is a population fraction, not "one agent".
TRAINER_REFERENCE_MASS = 0.10
MIN_TRAINER_MASS = 0.05

# Registered thresholds
MIN_TAIL_MAE_IMPROVEMENT = 0.05
MIN_BOUNDARY_GAP = 0.02

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_cost_of_return_tail"


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
class Population:
    strategy_shared: np.ndarray
    competence: np.ndarray
    sigma_independent: np.ndarray


@dataclass
class BranchBatch:
    cell: Cell
    theta_values: np.ndarray
    upward: np.ndarray
    downward: np.ndarray
    entry_coarse: np.ndarray
    exit_coarse: np.ndarray
    entry_continuous: np.ndarray
    exit_continuous: np.ndarray
    top_competence: np.ndarray
    sigma_independent: np.ndarray
    meanfield_exit: float
    tail_exit_realized: np.ndarray


def expected_sigma_squared() -> float:
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


def analytic_exit_mean(
    decay: float,
    liability_ratchet: float,
) -> float:
    c_star = competence_fixed_point(decay)
    return (
        -COMMON_SHARED_BIAS**2
        -SIGMA2_EXPECTED / c_star**2
        -COST_GAP
        -liability_ratchet
    )


def approximate_expected_min_sigma2(n_agents: int) -> float:
    """
    Quantile approximation for the favorable order statistic.

    sigma_i = sigma0 * exp(h Z)
    sigma_i^2 = sigma0^2 * exp(2 h Z)

    The minimum is approximated by the p=1/(N+1) quantile.
    """
    p = 1.0 / (n_agents + 1.0)
    z = NormalDist().inv_cdf(p)
    return float(
        INDEPENDENT_ERROR_STD**2
        * math.exp(2.0 * INDEPENDENT_ERROR_HETEROGENEITY * z)
    )


def analytic_exit_tail_quantile(
    n_agents: int,
    decay: float,
    liability_ratchet: float,
) -> float:
    c_star = competence_fixed_point(decay)
    min_sigma2 = approximate_expected_min_sigma2(n_agents)
    return (
        -COMMON_SHARED_BIAS**2
        -min_sigma2 / c_star**2
        -COST_GAP
        -liability_ratchet
    )


def initialize_population(
    n_seeds: int,
    n_agents: int,
    seed: int,
) -> tuple[np.random.Generator, Population]:
    rng = np.random.default_rng(seed)
    sigma = INDEPENDENT_ERROR_STD * np.exp(
        rng.normal(
            0.0,
            INDEPENDENT_ERROR_HETEROGENEITY,
            size=(n_seeds, n_agents),
        )
    )
    return rng, Population(
        strategy_shared=np.zeros((n_seeds, n_agents), dtype=bool),
        competence=np.ones((n_seeds, n_agents), dtype=float),
        sigma_independent=sigma,
    )


def trainer_multiplier(
    population: Population,
) -> np.ndarray:
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
    rebuilding_law: Literal[
        "none",
        "fixed",
        "subsidy_gated",
        "full_gated",
    ] = "none",
) -> None:
    shared = population.strategy_shared
    competence = population.competence

    if rebuilding_law in ("subsidy_gated", "full_gated"):
        trainer = trainer_multiplier(population)
    elif rebuilding_law == "fixed":
        trainer = np.ones(shared.shape[0])
    else:
        trainer = np.zeros(shared.shape[0])

    training_rate = (
        TRAINING_EFFECTIVENESS
        * training_allocation
        * trainer[:, None]
    )

    if rebuilding_law == "full_gated":
        natural_independent_rate = (
            INDEPENDENT_REBUILD_RATE * trainer[:, None]
        )
    else:
        natural_independent_rate = INDEPENDENT_REBUILD_RATE

    competence[:] = np.where(
        shared,
        competence
        + PASSIVE_RETENTION * (1.0 - competence)
        - decay * competence
        + training_rate * (1.0 - competence),
        competence
        + (
            natural_independent_rate
            + training_rate
        )
        * (1.0 - competence),
    )
    np.clip(competence, 1e-3, 1.0, out=competence)

    shared_fraction = np.mean(shared, axis=1)
    shared_error = (
        (1.0 - shared_fraction) * COMMON_SHARED_BIAS
    ) ** 2
    independent_error = (
        (shared_fraction[:, None] * COMMON_SHARED_BIAS) ** 2
        + population.sigma_independent**2 / competence**2
    )

    utility_shared = (
        -shared_error
        -COST_SHARED
        +theta
    )
    utility_independent = (
        -independent_error
        -COST_INDEPENDENT
        -LIABILITY_BASE
        -liability_ratchet * shared_fraction[:, None]
    )

    switch_advantage = np.where(
        shared,
        utility_independent - utility_shared[:, None],
        utility_shared[:, None] - utility_independent,
    )
    switch_probability = (
        MUTATION_RATE
        +SWITCH_RATE
        * np.maximum(
            0.0,
            np.tanh(SELECTION_GAIN * switch_advantage),
        )
    )
    switch_probability = np.clip(
        switch_probability,
        0.0,
        1.0,
    )

    switches = (
        rng.random(size=shared.shape)
        < switch_probability
    )
    shared[:] = np.where(switches, ~shared, shared)


def coarse_crossing(
    branch: np.ndarray,
    theta_values: np.ndarray,
) -> np.ndarray:
    mask = branch >= SHARED_THRESHOLD
    indices = np.argmax(mask, axis=1)
    exists = np.any(mask, axis=1)
    result = np.full(branch.shape[0], np.nan)
    result[exists] = theta_values[indices[exists]]
    return result


def continuous_crossing(
    branch: np.ndarray,
    theta_values: np.ndarray,
) -> np.ndarray:
    """
    First linearly interpolated f=SHARED_THRESHOLD crossing for each seed.

    The branch is aligned to ascending theta. Nonmonotonic finite-agent paths
    are permitted; the first low-to-high crossing is used.
    """
    result = np.full(branch.shape[0], np.nan)

    for seed_index in range(branch.shape[0]):
        values = branch[seed_index]
        above = values >= SHARED_THRESHOLD
        crossing_indices = np.where(
            (~above[:-1]) & above[1:]
        )[0]

        if len(crossing_indices) == 0:
            if above[0]:
                result[seed_index] = theta_values[0]
            continue

        index = int(crossing_indices[0])
        y0 = values[index]
        y1 = values[index + 1]
        x0 = theta_values[index]
        x1 = theta_values[index + 1]

        if abs(y1 - y0) < 1e-12:
            result[seed_index] = x1
        else:
            fraction = (
                SHARED_THRESHOLD - y0
            ) / (y1 - y0)
            result[seed_index] = (
                x0 + fraction * (x1 - x0)
            )

    return result


def simulate_branch_batch(
    cell: Cell,
    n_agents: int,
    n_seeds: int,
    dwell: int,
    seed: int,
    theta_values: np.ndarray = THETA_VALUES,
) -> BranchBatch:
    rng, population = initialize_population(
        n_seeds,
        n_agents,
        seed,
    )

    upward_points = []
    for theta in theta_values:
        for _ in range(dwell):
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

    top_competence = population.competence.copy()
    sigma_copy = population.sigma_independent.copy()

    downward_descending = []
    for theta in theta_values[::-1]:
        for _ in range(dwell):
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

    tail_penalty = np.min(
        sigma_copy**2 / top_competence**2,
        axis=1,
    )
    tail_exit_realized = (
        -COMMON_SHARED_BIAS**2
        -tail_penalty
        -COST_GAP
        -cell.liability_ratchet
    )

    return BranchBatch(
        cell=cell,
        theta_values=theta_values.copy(),
        upward=upward,
        downward=downward,
        entry_coarse=coarse_crossing(
            upward,
            theta_values,
        ),
        exit_coarse=coarse_crossing(
            downward,
            theta_values,
        ),
        entry_continuous=continuous_crossing(
            upward,
            theta_values,
        ),
        exit_continuous=continuous_crossing(
            downward,
            theta_values,
        ),
        top_competence=top_competence,
        sigma_independent=sigma_copy,
        meanfield_exit=analytic_exit_mean(
            cell.decay,
            cell.liability_ratchet,
        ),
        tail_exit_realized=tail_exit_realized,
    )


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = finite(values)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    se = float(
        np.std(values, ddof=1)
        / math.sqrt(len(values))
    )
    return mean, 1.96 * se


def rmse(
    observed: np.ndarray,
    predicted: np.ndarray | float,
) -> float:
    mask = np.isfinite(observed)
    return float(
        np.sqrt(
            np.mean(
                (
                    observed[mask]
                    - np.asarray(predicted)[mask]
                    if np.asarray(predicted).ndim > 0
                    else observed[mask] - float(predicted)
                ) ** 2
            )
        )
    )


def mae(
    observed: np.ndarray,
    predicted: np.ndarray | float,
) -> float:
    mask = np.isfinite(observed)
    if np.asarray(predicted).ndim > 0:
        difference = (
            observed[mask]
            - np.asarray(predicted)[mask]
        )
    else:
        difference = observed[mask] - float(predicted)
    return float(np.mean(np.abs(difference)))


def run_main_factorial() -> dict[str, BranchBatch]:
    return {
        cell.code: simulate_branch_batch(
            cell,
            N_MAIN,
            N_MAIN_SEEDS,
            DWELL_MAIN,
            MASTER_SEED,
        )
        for cell in CELLS
    }


def run_ratio_sweep() -> dict[str, np.ndarray]:
    observed = []
    mean_prediction = []
    tail_quantile_prediction = []
    realized_tail_prediction = []
    mean_rmse = []
    tail_rmse = []

    for index, ratio in enumerate(DELTA_RHO_RATIOS):
        decay = float(
            ratio * PASSIVE_RETENTION
        )
        cell = Cell(
            f"R{index}",
            f"delta/rho={ratio:.2f}",
            decay,
            LIABILITY_OFF,
        )
        batch = simulate_branch_batch(
            cell,
            N_MAIN,
            N_SWEEP_SEEDS,
            DWELL_MAIN,
            MASTER_SEED + 10_000,
        )
        observed.append(
            float(np.nanmedian(batch.exit_continuous))
        )
        mean_prediction.append(batch.meanfield_exit)
        tail_quantile_prediction.append(
            analytic_exit_tail_quantile(
                N_MAIN,
                decay,
                LIABILITY_OFF,
            )
        )
        realized_tail_prediction.append(
            float(
                np.nanmedian(
                    batch.tail_exit_realized
                )
            )
        )
        mean_rmse.append(
            rmse(
                batch.exit_continuous,
                batch.meanfield_exit,
            )
        )
        tail_rmse.append(
            rmse(
                batch.exit_continuous,
                batch.tail_exit_realized,
            )
        )

    return {
        "ratio": DELTA_RHO_RATIOS.copy(),
        "observed": np.asarray(observed),
        "mean_prediction": np.asarray(mean_prediction),
        "tail_quantile_prediction": np.asarray(
            tail_quantile_prediction
        ),
        "realized_tail_prediction": np.asarray(
            realized_tail_prediction
        ),
        "mean_rmse": np.asarray(mean_rmse),
        "tail_rmse": np.asarray(tail_rmse),
    }


def run_n_sweep() -> dict[str, np.ndarray]:
    observed = []
    mean_prediction = []
    tail_quantile_prediction = []
    realized_tail_prediction = []
    mean_rmse = []
    tail_rmse = []

    decay = float(
        N_SWEEP_RATIO * PASSIVE_RETENTION
    )
    cell = Cell(
        "N",
        f"N sweep at delta/rho={N_SWEEP_RATIO:.2f}",
        decay,
        LIABILITY_OFF,
    )

    for n_agents in N_VALUES:
        batch = simulate_branch_batch(
            cell,
            int(n_agents),
            N_SWEEP_SEEDS,
            DWELL_MAIN,
            MASTER_SEED + 20_000,
        )
        observed.append(
            float(np.nanmedian(batch.exit_continuous))
        )
        mean_prediction.append(batch.meanfield_exit)
        tail_quantile_prediction.append(
            analytic_exit_tail_quantile(
                int(n_agents),
                decay,
                LIABILITY_OFF,
            )
        )
        realized_tail_prediction.append(
            float(
                np.nanmedian(
                    batch.tail_exit_realized
                )
            )
        )
        mean_rmse.append(
            rmse(
                batch.exit_continuous,
                batch.meanfield_exit,
            )
        )
        tail_rmse.append(
            rmse(
                batch.exit_continuous,
                batch.tail_exit_realized,
            )
        )

    return {
        "N": N_VALUES.copy(),
        "observed": np.asarray(observed),
        "mean_prediction": np.asarray(mean_prediction),
        "tail_quantile_prediction": np.asarray(
            tail_quantile_prediction
        ),
        "realized_tail_prediction": np.asarray(
            realized_tail_prediction
        ),
        "mean_rmse": np.asarray(mean_rmse),
        "tail_rmse": np.asarray(tail_rmse),
    }


def run_dwell_sweep() -> dict[str, np.ndarray]:
    cell = CELLS[3]  # decay + ratchet, strongest metastability case
    exit_median = []
    exit_mean = []
    exit_ci = []

    for dwell in DWELL_VALUES:
        batch = simulate_branch_batch(
            cell,
            N_MAIN,
            N_DWELL_SEEDS,
            int(dwell),
            MASTER_SEED + 30_000,
        )
        values = batch.exit_continuous
        mean, half_ci = mean_ci(values)
        exit_median.append(float(np.nanmedian(values)))
        exit_mean.append(mean)
        exit_ci.append(half_ci)

    return {
        "dwell": DWELL_VALUES.copy(),
        "median_exit": np.asarray(exit_median),
        "mean_exit": np.asarray(exit_mean),
        "half_ci": np.asarray(exit_ci),
    }


def snapshot_lock_states() -> tuple[
    np.ndarray,
    dict[int, tuple[np.ndarray, np.ndarray]],
]:
    rng, population = initialize_population(
        N_RECOVERY_SEEDS,
        N_MAIN,
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


def recovery_success(
    population: Population,
) -> np.ndarray:
    shared_fraction = np.mean(
        population.strategy_shared,
        axis=1,
    )
    independent_mask = ~population.strategy_shared
    independent_count = np.sum(
        independent_mask,
        axis=1,
    )
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
    rebuilding_law: Literal[
        "fixed",
        "subsidy_gated",
        "full_gated",
    ],
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

    for lock_index, lock_duration in enumerate(
        LOCK_DURATIONS
    ):
        strategy_snapshot, competence_snapshot = snapshots[
            int(lock_duration)
        ]

        for theta_index, theta_recovery in enumerate(
            RECOVERY_THETA_VALUES
        ):
            # Common random numbers across rebuilding laws isolate the
            # treatment difference from Monte Carlo noise.
            rng = np.random.default_rng(
                MASTER_SEED
                +100_000
                +1_000 * lock_index
                +theta_index
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

            result[
                lock_index,
                theta_index,
            ] = float(
                np.mean(
                    recovery_success(population)
                )
            )

    return result


def boundary_band_statistics(
    fixed_map: np.ndarray,
    comparison_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Boundary band is defined only from the fixed-law baseline:
        0.1 <= P_fixed <= 0.9.

    Returns gap by lock duration, count of band cells, and pooled gap.
    """
    gap = fixed_map - comparison_map
    band = (
        (fixed_map >= 0.10)
        & (fixed_map <= 0.90)
    )

    gap_by_lock = np.full(
        len(LOCK_DURATIONS),
        np.nan,
    )
    counts = np.zeros(
        len(LOCK_DURATIONS),
        dtype=int,
    )

    for lock_index in range(len(LOCK_DURATIONS)):
        mask = band[lock_index]
        counts[lock_index] = int(np.sum(mask))
        if counts[lock_index] > 0:
            gap_by_lock[lock_index] = float(
                np.mean(gap[lock_index, mask])
            )

    pooled = float(
        np.mean(gap[band])
    ) if np.any(band) else float("nan")
    return gap_by_lock, counts, pooled


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    filename: str,
    colorbar_label: str,
    diverging: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    if diverging:
        vmax = float(np.max(np.abs(matrix)))
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
            vmin=-vmax,
            vmax=vmax,
        )
    else:
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
            vmin=0.0,
            vmax=1.0,
        )
    ax.set_xlabel(
        "recovery shared-system advantage theta"
    )
    ax.set_ylabel(
        "time held in consolidated regime"
    )
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=170,
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    main_batches = run_main_factorial()
    ratio = run_ratio_sweep()
    n_sweep = run_n_sweep()
    dwell = run_dwell_sweep()

    sigma_recovery, snapshots = snapshot_lock_states()
    fixed_map = recovery_map(
        "fixed",
        sigma_recovery,
        snapshots,
    )
    subsidy_map = recovery_map(
        "subsidy_gated",
        sigma_recovery,
        snapshots,
    )
    full_gated_map = recovery_map(
        "full_gated",
        sigma_recovery,
        snapshots,
    )

    subsidy_gap_by_lock, subsidy_band_counts, subsidy_band_gap = (
        boundary_band_statistics(
            fixed_map,
            subsidy_map,
        )
    )
    full_gap_by_lock, full_band_counts, full_band_gap = (
        boundary_band_statistics(
            fixed_map,
            full_gated_map,
        )
    )

    # ------------------------------------------------------------------
    # P1: tail correction
    # ------------------------------------------------------------------

    main_mean_mae = []
    main_tail_mae = []

    for cell in CELLS:
        batch = main_batches[cell.code]
        main_mean_mae.append(
            mae(
                batch.exit_continuous,
                batch.meanfield_exit,
            )
        )
        main_tail_mae.append(
            mae(
                batch.exit_continuous,
                batch.tail_exit_realized,
            )
        )

    pooled_mean_mae = float(
        np.mean(main_mean_mae)
    )
    pooled_tail_mae = float(
        np.mean(main_tail_mae)
    )
    tail_improvement = (
        pooled_mean_mae - pooled_tail_mae
    ) / pooled_mean_mae
    p1 = (
        tail_improvement
        >= MIN_TAIL_MAE_IMPROVEMENT
    )

    # ------------------------------------------------------------------
    # P2: interaction under continuous thresholds
    # ------------------------------------------------------------------

    interaction_coarse = (
        (
            main_batches["D"].entry_coarse
            - main_batches["D"].exit_coarse
        )
        -(
            main_batches["B"].entry_coarse
            - main_batches["B"].exit_coarse
        )
        -(
            main_batches["C"].entry_coarse
            - main_batches["C"].exit_coarse
        )
        +(
            main_batches["A"].entry_coarse
            - main_batches["A"].exit_coarse
        )
    )
    interaction_continuous = (
        (
            main_batches["D"].entry_continuous
            - main_batches["D"].exit_continuous
        )
        -(
            main_batches["B"].entry_continuous
            - main_batches["B"].exit_continuous
        )
        -(
            main_batches["C"].entry_continuous
            - main_batches["C"].exit_continuous
        )
        +(
            main_batches["A"].entry_continuous
            - main_batches["A"].exit_continuous
        )
    )
    coarse_mean, coarse_ci = mean_ci(
        interaction_coarse
    )
    continuous_mean, continuous_ci = mean_ci(
        interaction_continuous
    )
    p2 = abs(continuous_mean) < abs(coarse_mean)

    # ------------------------------------------------------------------
    # P3/P4/P5/P6/P7
    # ------------------------------------------------------------------

    ratio_mean_rmse = float(
        np.sqrt(
            np.mean(
                (
                    ratio["observed"]
                    -ratio["mean_prediction"]
                ) ** 2
            )
        )
    )
    ratio_tail_rmse = float(
        np.sqrt(
            np.mean(
                (
                    ratio["observed"]
                    -ratio["realized_tail_prediction"]
                ) ** 2
            )
        )
    )
    p3 = ratio_tail_rmse < ratio_mean_rmse

    n_mean_rmse = float(
        np.sqrt(
            np.mean(
                (
                    n_sweep["observed"]
                    -n_sweep["mean_prediction"]
                ) ** 2
            )
        )
    )
    n_tail_rmse = float(
        np.sqrt(
            np.mean(
                (
                    n_sweep["observed"]
                    -n_sweep["realized_tail_prediction"]
                ) ** 2
            )
        )
    )
    observed_n_direction = (
        n_sweep["observed"][-1]
        > n_sweep["observed"][0]
    )
    p4 = (
        observed_n_direction
        and n_tail_rmse < n_mean_rmse
    )

    p5 = bool(
        np.all(
            np.diff(
                dwell["median_exit"]
            ) >= -1e-9
        )
    )

    p6 = (
        math.isfinite(subsidy_band_gap)
        and subsidy_band_gap
        >= MIN_BOUNDARY_GAP
    )

    strict_unreachable_region = (
        (fixed_map >= 0.50)
        & (full_gated_map <= 0.05)
    )
    p7 = bool(
        np.any(strict_unreachable_region)
    )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    print("=" * 104)
    print("THE COST OF RETURNING — CYCLE 3")
    print("TAIL-NUCLEATED REVERSIBILITY IN A FINITE EPISTEMIC POPULATION")
    print("=" * 104)
    print(f"master seed                          : {MASTER_SEED}")
    print(f"main agents / paired seeds           : {N_MAIN} / {N_MAIN_SEEDS}")
    print(f"theta step                           : {THETA_VALUES[1]-THETA_VALUES[0]:.3f}")
    print(f"main dwell                           : {DWELL_MAIN}")
    print(f"mutation probability/evaluation      : {MUTATION_RATE}")
    print()

    print("MAIN FACTORIAL: CONTINUOUS THRESHOLDS AND EXIT PREDICTIONS")
    print(
        f"{'cell':<6}"
        f"{'entry':>10}"
        f"{'exit':>10}"
        f"{'mean pred':>12}"
        f"{'tail pred':>12}"
        f"{'MAE mean':>11}"
        f"{'MAE tail':>11}"
    )
    print("-" * 76)

    for cell in CELLS:
        batch = main_batches[cell.code]
        print(
            f"{cell.code:<6}"
            f"{np.nanmedian(batch.entry_continuous):10.3f}"
            f"{np.nanmedian(batch.exit_continuous):10.3f}"
            f"{batch.meanfield_exit:12.3f}"
            f"{np.nanmedian(batch.tail_exit_realized):12.3f}"
            f"{mae(batch.exit_continuous, batch.meanfield_exit):11.3f}"
            f"{mae(batch.exit_continuous, batch.tail_exit_realized):11.3f}"
        )

    print()
    print("P1 ORDER-STATISTIC CORRECTION")
    print(f"pooled mean-field MAE               : {pooled_mean_mae:.4f}")
    print(f"pooled realized-tail MAE            : {pooled_tail_mae:.4f}")
    print(f"relative MAE improvement             : {tail_improvement:.3f}")

    print()
    print("P2 LIABILITY x DECAY INTERACTION")
    print(
        f"coarse-grid I_H                      : "
        f"{coarse_mean:+.5f} +/- {coarse_ci:.5f}"
    )
    print(
        f"continuous-crossing I_H              : "
        f"{continuous_mean:+.5f} +/- {continuous_ci:.5f}"
    )

    print()
    print("DELTA/RHO SWEEP")
    print(
        f"{'ratio':>8}"
        f"{'observed':>12}"
        f"{'mean pred':>12}"
        f"{'tail q pred':>13}"
        f"{'tail real':>12}"
    )
    for index, value in enumerate(ratio["ratio"]):
        print(
            f"{value:8.2f}"
            f"{ratio['observed'][index]:12.3f}"
            f"{ratio['mean_prediction'][index]:12.3f}"
            f"{ratio['tail_quantile_prediction'][index]:13.3f}"
            f"{ratio['realized_tail_prediction'][index]:12.3f}"
        )
    print(f"ratio-sweep mean-field RMSE          : {ratio_mean_rmse:.4f}")
    print(f"ratio-sweep realized-tail RMSE       : {ratio_tail_rmse:.4f}")

    print()
    print("POPULATION-SIZE SWEEP")
    print(
        f"{'N':>8}"
        f"{'observed':>12}"
        f"{'mean pred':>12}"
        f"{'tail q pred':>13}"
        f"{'tail real':>12}"
    )
    for index, value in enumerate(n_sweep["N"]):
        print(
            f"{int(value):8d}"
            f"{n_sweep['observed'][index]:12.3f}"
            f"{n_sweep['mean_prediction'][index]:12.3f}"
            f"{n_sweep['tail_quantile_prediction'][index]:13.3f}"
            f"{n_sweep['realized_tail_prediction'][index]:12.3f}"
        )
    print(f"N-sweep mean-field RMSE              : {n_mean_rmse:.4f}")
    print(f"N-sweep realized-tail RMSE           : {n_tail_rmse:.4f}")

    print()
    print("DWELL-TIME SWEEP, CELL D")
    print(
        f"{'dwell':>8}"
        f"{'median exit':>14}"
        f"{'mean exit':>12}"
        f"{'95% half-CI':>14}"
    )
    for index, dwell_value in enumerate(dwell["dwell"]):
        print(
            f"{int(dwell_value):8d}"
            f"{dwell['median_exit'][index]:14.3f}"
            f"{dwell['mean_exit'][index]:12.3f}"
            f"{dwell['half_ci'][index]:14.3f}"
        )

    print()
    print("RECOVERY BOUNDARY BAND")
    print(f"whole-map fixed mean                 : {np.mean(fixed_map):.3f}")
    print(f"whole-map subsidy-gated mean         : {np.mean(subsidy_map):.3f}")
    print(f"whole-map full-gated mean            : {np.mean(full_gated_map):.3f}")
    print(f"subsidy-gated pooled boundary gap    : {subsidy_band_gap:.3f}")
    print(f"full-gated pooled boundary gap       : {full_band_gap:.3f}")
    print(
        f"strict full-gating unreachable cells : "
        f"{int(np.sum(strict_unreachable_region))}"
    )
    print(
        f"{'T_lock':>8}"
        f"{'band n':>10}"
        f"{'subsidy gap':>14}"
        f"{'full gap':>12}"
    )
    for index, lock_duration in enumerate(LOCK_DURATIONS):
        subsidy_text = (
            f"{subsidy_gap_by_lock[index]:.3f}"
            if math.isfinite(subsidy_gap_by_lock[index])
            else "-"
        )
        full_text = (
            f"{full_gap_by_lock[index]:.3f}"
            if math.isfinite(full_gap_by_lock[index])
            else "-"
        )
        print(
            f"{int(lock_duration):8d}"
            f"{int(subsidy_band_counts[index]):10d}"
            f"{subsidy_text:>14}"
            f"{full_text:>12}"
        )

    print()
    print("DESIGN-FROZEN CHECKS")
    print(f"P1 tail prediction improves MAE      : {'PASS' if p1 else 'FAIL'}")
    print(f"P2 continuous I_H shrinks            : {'PASS' if p2 else 'FAIL'}")
    print(f"P3 tail closes delta/rho gap          : {'PASS' if p3 else 'FAIL'}")
    print(f"P4 tail captures N dependence         : {'PASS' if p4 else 'FAIL'}")
    print(f"P5 exit shallows with dwell           : {'PASS' if p5 else 'FAIL'}")
    print(f"P6 boundary-band recovery gap         : {'PASS' if p6 else 'FAIL'}")
    print(
        f"OPEN P7 full-gating absorbing region  : "
        f"{'PRESENT' if p7 else 'ABSENT'}"
    )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Main branches
    fig, ax = plt.subplots(figsize=(10.3, 6.0))
    for cell in CELLS:
        batch = main_batches[cell.code]
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
    ax.set_ylabel("median shared fraction")
    ax.set_title("Finite-agent hysteresis branches")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "tail-main-branches.png",
        dpi=170,
    )
    plt.close(fig)

    # Prediction errors
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    positions = np.arange(len(CELLS))
    width = 0.35
    ax.bar(
        positions - width / 2,
        main_mean_mae,
        width,
        label="mean-field MAE",
    )
    ax.bar(
        positions + width / 2,
        main_tail_mae,
        width,
        label="realized-tail MAE",
    )
    ax.set_xticks(
        positions,
        [cell.code for cell in CELLS],
    )
    ax.set_xlabel("factorial cell")
    ax.set_ylabel("absolute exit-threshold error")
    ax.set_title(
        "Does the best-preserved channel predict return better than the mean?"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "tail-prediction-errors.png",
        dpi=170,
    )
    plt.close(fig)

    # Ratio sweep
    fig, ax = plt.subplots(figsize=(9.3, 5.4))
    ax.plot(
        ratio["ratio"],
        ratio["observed"],
        marker="o",
        label="observed exit",
    )
    ax.plot(
        ratio["ratio"],
        ratio["mean_prediction"],
        marker="s",
        linestyle="--",
        label="mean-field prediction",
    )
    ax.plot(
        ratio["ratio"],
        ratio["tail_quantile_prediction"],
        marker="^",
        linestyle=":",
        label="order-statistic quantile",
    )
    ax.plot(
        ratio["ratio"],
        ratio["realized_tail_prediction"],
        marker="x",
        label="realized-tail prediction",
    )
    ax.set_xlabel("delta/rho")
    ax.set_ylabel("exit threshold theta")
    ax.set_title("Competence decay: mean versus tail prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "tail-delta-rho.png",
        dpi=170,
    )
    plt.close(fig)

    # N sweep
    fig, ax = plt.subplots(figsize=(9.3, 5.4))
    ax.plot(
        n_sweep["N"],
        n_sweep["observed"],
        marker="o",
        label="observed exit",
    )
    ax.plot(
        n_sweep["N"],
        n_sweep["mean_prediction"],
        marker="s",
        linestyle="--",
        label="mean-field prediction",
    )
    ax.plot(
        n_sweep["N"],
        n_sweep["tail_quantile_prediction"],
        marker="^",
        linestyle=":",
        label="order-statistic quantile",
    )
    ax.plot(
        n_sweep["N"],
        n_sweep["realized_tail_prediction"],
        marker="x",
        label="realized-tail prediction",
    )
    ax.set_xlabel("number of observer organizations N")
    ax.set_ylabel("exit threshold theta")
    ax.set_title("Larger populations improve the favorable recovery tail")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "tail-N-sweep.png",
        dpi=170,
    )
    plt.close(fig)

    # Dwell sweep
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.errorbar(
        dwell["dwell"],
        dwell["mean_exit"],
        yerr=dwell["half_ci"],
        marker="o",
        capsize=4,
        label="mean exit with 95% CI",
    )
    ax.plot(
        dwell["dwell"],
        dwell["median_exit"],
        marker="s",
        linestyle="--",
        label="median exit",
    )
    ax.set_xlabel("dwell evaluations per theta")
    ax.set_ylabel("exit threshold theta")
    ax.set_title(
        "Exit is a protocol-dependent first-passage quantity"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "tail-dwell-sweep.png",
        dpi=170,
    )
    plt.close(fig)

    # Interaction
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.bar(
        ["coarse grid", "continuous crossing"],
        [coarse_mean, continuous_mean],
        yerr=[coarse_ci, continuous_ci],
        capsize=5,
    )
    ax.axhline(0.0, linestyle=":")
    ax.set_ylabel("liability x decay interaction I_H")
    ax.set_title(
        "Does the apparent subadditivity survive de-quantization?"
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "tail-interaction.png",
        dpi=170,
    )
    plt.close(fig)

    plot_heatmap(
        fixed_map,
        "Recovery with fixed rebuilding efficiency",
        "recovery-fixed.png",
        "P(recovered)",
    )
    plot_heatmap(
        subsidy_map,
        "Recovery when only the training subsidy needs surviving trainers",
        "recovery-subsidy-gated.png",
        "P(recovered)",
    )
    plot_heatmap(
        full_gated_map,
        "Strong ablation: all rebuilding requires a surviving trainer ecology",
        "recovery-full-gated.png",
        "P(recovered)",
    )

    # Boundary gap by lock duration
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    ax.plot(
        LOCK_DURATIONS,
        subsidy_gap_by_lock,
        marker="o",
        label="subsidy-gated gap",
    )
    ax.plot(
        LOCK_DURATIONS,
        full_gap_by_lock,
        marker="s",
        label="fully gated ablation gap",
    )
    ax.axhline(0.0, linestyle=":")
    ax.set_xlabel("time held in consolidated regime")
    ax.set_ylabel(
        "mean recovery-probability gap in fixed-law boundary band"
    )
    ax.set_title(
        "The second-order mechanism is concentrated near the recovery boundary"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "recovery-boundary-gap.png",
        dpi=170,
    )
    plt.close(fig)

    print()
    print("Saved:")
    for filename in (
        "tail-main-branches.png",
        "tail-prediction-errors.png",
        "tail-delta-rho.png",
        "tail-N-sweep.png",
        "tail-dwell-sweep.png",
        "tail-interaction.png",
        "recovery-fixed.png",
        "recovery-subsidy-gated.png",
        "recovery-full-gated.png",
        "recovery-boundary-gap.png",
    ):
        print(f"  {OUTPUT_DIR / filename}")

    if not all((p1, p3, p4, p5, p6)):
        raise SystemExit(
            "\nOne or more core paper-gate predictions failed. "
            "Treat this as a scientific result, not a software error."
        )


if __name__ == "__main__":
    main()
