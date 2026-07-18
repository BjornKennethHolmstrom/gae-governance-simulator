#!/usr/bin/env python3
"""
Governance as Engineering — Paper XXVI Final Confirmation
=========================================================

THE ESCAPE LADDER AND THE HAZARD OF RETURN

This is the final registered simulation cycle for the proposed Paper XXVI.
It adds no new institutional mechanisms.

It tests only:

1. whether the deterministic escape-ladder phase structure is reproduced by
   finite heterogeneous populations with zero fitted parameters; and

2. whether fixed-theta escape hazards measured independently compose into the
   dwell-dependent exit thresholds observed under a downward parameter sweep.

The interaction between liability and competence decay is retired from the
paper's claims. The ladder predicts it is subadditive but far below the current
dynamic noise floor.

Relationship to Paper X
-----------------------
Paper X models epistemic consolidation using:
    - consensus-relative evaluation;
    - lower shared-system cost;
    - a liability ratchet;
    - periodic stochastic switching;
    - and a small fixed switch-back probability representing atrophied
      independent infrastructure.

This extension replaces the fixed switch-back probability with an explicit
retained independent competence state.

Deterministic escape ladder
---------------------------
For agent i at shared fraction f:

    A_i(f, theta)
        = b^2(1 - 2f) - L1 f - x_i - DeltaC - theta

where:

    x_i = sigma_i^2 / c_i^2
    DeltaC = C_I + L0 - C_S

One defection lowers f by 1/N and raises every remaining agent's defection
advantage by:

    Delta = (2b^2 + L1) / N

Order the consolidated penalties:

    y_1 <= y_2 <= ... <= y_N

Let K = ceil(N/2). The system crosses below half shared adoption iff:

    G(theta) > M_N

where:

    G(theta) = -theta - b^2 - L1 - DeltaC

    M_N = max_{k <= K} [y_k - (k-1) Delta]

Therefore:

    theta_exit^det
        = -(b^2 + L1 + DeltaC) - M_N

The binding rank k* distinguishes regimes:

    k* = 1:
        the best-preserved channel can nucleate and propagate escape alone;

    k* > 1:
        the first defector can leave before enough followers can complete the
        cascade, producing partial-defection plateaus.

Zero-parameter phase criterion
------------------------------
For each (N,h), the analytic phase prediction uses the lognormal quantiles:

    sigma^2(q) = sigma_0^2 exp(2 h Phi^{-1}(q))

scaled by the consolidated competence fixed point c_*.

No parameter is fitted to the Monte Carlo phase map. The predicted phase is
simply whether the quantile ladder has k*>1.

High-heterogeneity interpretation
---------------------------------
Extreme heterogeneity does not make the first defector's recruitment kick
stronger. The kick Delta is fixed by N, b, and L1.

Instead, for q < 1/2 the absolute lower-tail penalties compress toward zero at
large h. Their spacings can again fall below the linear ladder credit, returning
the system toward k*=1.

Stochastic first-passage layer
------------------------------
Below or near the deterministic threshold, escape is protocol-dependent.
Mutation-assisted defectors may persist long enough to recruit further
defectors.

For fixed theta, waiting times to half-exit are simulated directly. An
exponential hazard is estimated by the censored maximum-likelihood estimator:

    lambda_hat = events / total exposure time

The downward-sweep median is then predicted without fitting another curve:

    cumulative hazard
        = sum_j dwell * lambda(theta_j)

The predicted median exit is the first theta step where cumulative hazard
reaches ln 2.

Development honesty
-------------------
The theta range and Monte Carlo sizes were selected in pilot runs. All reported
confirmation results use a separate seed offset.

Registered checks [R within model]
-----------------------------------
P1 PHASE REPRODUCTION
   The zero-parameter quantile-ladder classification agrees with the empirical
   majority phase, P(k*>1) > 0.5, in at least 85% of the frozen (N,h) cells.

P2 THREE ANCHOR REGIMES
   The phase test reproduces:
       - tail-dominated original regime: N=20,h=0.15;
       - ladder regime: N=1000,h=0.15;
       - high-h return toward tail: N=1000,h=0.80.

P3 HAZARD VALIDATION
   Exponential survival curves estimated from half the fixed-theta trials have
   mean absolute survival error <= 0.12 on the held-out half.

P4 SWEEP COMPOSITION
   The hazard-composed prediction of median exit across dwell values has mean
   absolute error <= 0.012 in theta on new population draws.

P5 DWELL ORDERING
   Both predicted and observed median exits become monotonically shallower as
   dwell time increases.

Interpretation boundary
-----------------------
The exponential hazard is an operational approximation over the tested
timescale, not a universal waiting-time theorem.

All findings are [R within the model]. Governance transfer remains [IP].

Dependencies
------------
Python 3.9+, NumPy, Matplotlib

Run
---
    python3 gae-sim-paper-26-final-confirmation.py

Outputs
-------
    outputs_paper_26_final/final-phase-map.png
    outputs_paper_26_final/final-binding-rank-map.png
    outputs_paper_26_final/final-hazard-curves.png
    outputs_paper_26_final/final-hazard-validation.png
    outputs_paper_26_final/final-sweep-composition.png
    outputs_paper_26_final/final-results.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

# ---------------------------------------------------------------------------
# Frozen confirmation design
# ---------------------------------------------------------------------------

MASTER_SEED = 20260718
CONFIRMATION_SEED_OFFSET = 1_000_000

COMMON_SHARED_BIAS = 0.45
INDEPENDENT_ERROR_STD = 0.12

COST_SHARED = 0.50
COST_INDEPENDENT = 1.00
LIABILITY_BASE = 0.20
LIABILITY_RATCHET = 0.15

PASSIVE_RETENTION = 0.003
COMPETENCE_DECAY = 0.005
INDEPENDENT_REBUILD_RATE = 0.05

SELECTION_GAIN = 8.0
SWITCH_RATE = 0.25
MUTATION_RATE = 0.0005

COST_GAP = COST_INDEPENDENT + LIABILITY_BASE - COST_SHARED
SHARED_EXIT_THRESHOLD = 0.50

# Phase-map confirmation
MAP_N_VALUES = np.array([20, 50, 100, 200, 500, 1000, 2000])
MAP_H_VALUES = np.array([0.00, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80])
MAP_SAMPLES = 1500

# Fixed-theta hazard measurement
HAZARD_N = 20
HAZARD_H = 0.15
HAZARD_POPULATIONS = 18
HAZARD_TRIALS_PER_POPULATION = 36
HAZARD_ESTIMATION_TRIALS = HAZARD_TRIALS_PER_POPULATION // 2
HAZARD_MAX_TIME = 1600
HAZARD_THETA_VALUES = np.arange(-1.17, -1.019, 0.01)
SURVIVAL_CHECK_TIMES = np.array([50, 100, 300, 700, 1200, 1600])

# Out-of-sample sweep composition
DWELL_VALUES = np.array([10, 30, 100, 300])
SWEEP_TRIALS_PER_POPULATION = 36
NEW_SWEEP_POPULATIONS = 24

# Registered tolerances
MIN_PHASE_ACCURACY = 0.85
MAX_SURVIVAL_MAE = 0.12
MAX_SWEEP_THETA_MAE = 0.012

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_paper_26_final"


@dataclass
class Population:
    strategy_shared: np.ndarray
    competence: np.ndarray
    sigma_independent: np.ndarray


def competence_fixed_point() -> float:
    return PASSIVE_RETENTION / (
        PASSIVE_RETENTION + COMPETENCE_DECAY
    )


C_STAR = competence_fixed_point()


def cascade_credit(n_agents: int) -> float:
    return (
        2.0 * COMMON_SHARED_BIAS**2
        + LIABILITY_RATCHET
    ) / n_agents


def base_exit_constant() -> float:
    return -(
        COMMON_SHARED_BIAS**2
        + LIABILITY_RATCHET
        + COST_GAP
    )


def draw_sigma(
    rng: np.random.Generator,
    n_populations: int,
    n_agents: int,
    h: float,
) -> np.ndarray:
    return (
        INDEPENDENT_ERROR_STD
        * np.exp(
            rng.normal(
                0.0,
                h,
                size=(n_populations, n_agents),
            )
        )
    )


def ladder_statistics_from_penalties(
    penalties_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    penalties_sorted shape: (population, N), ascending.

    Returns:
        M_N,
        binding rank k* (1-indexed),
        ladder gap M_N-y_1.
    """
    n_agents = penalties_sorted.shape[1]
    k_max = math.ceil(n_agents / 2)
    credit = cascade_credit(n_agents)

    ladder_values = (
        penalties_sorted[:, :k_max]
        - np.arange(k_max, dtype=float)[None, :] * credit
    )
    binding_zero = np.argmax(ladder_values, axis=1)
    m_value = ladder_values[
        np.arange(len(ladder_values)),
        binding_zero,
    ]
    y1 = penalties_sorted[:, 0]

    return (
        m_value,
        binding_zero + 1,
        m_value - y1,
    )


def realized_ladder_statistics(
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    penalties = np.sort(
        sigma**2 / C_STAR**2,
        axis=1,
    )
    m_value, binding_rank, gap = ladder_statistics_from_penalties(
        penalties
    )
    exit_threshold = base_exit_constant() - m_value
    return exit_threshold, binding_rank, gap, penalties


def quantile_penalties(
    n_agents: int,
    h: float,
) -> np.ndarray:
    probabilities = np.arange(
        1,
        n_agents + 1,
        dtype=float,
    ) / (n_agents + 1.0)

    z = np.asarray(
        [
            NormalDist().inv_cdf(float(probability))
            for probability in probabilities
        ]
    )
    sigma_squared = (
        INDEPENDENT_ERROR_STD**2
        * np.exp(2.0 * h * z)
    )
    return sigma_squared / C_STAR**2


def quantile_binding_rank(
    n_agents: int,
    h: float,
) -> tuple[int, float]:
    penalties = np.sort(
        quantile_penalties(n_agents, h)
    )[None, :]
    _, binding_rank, gap = ladder_statistics_from_penalties(
        penalties
    )
    return int(binding_rank[0]), float(gap[0])


# ---------------------------------------------------------------------------
# Phase-map confirmation
# ---------------------------------------------------------------------------

def run_phase_map() -> dict[str, np.ndarray | float | bool]:
    empirical_probability = np.zeros(
        (len(MAP_N_VALUES), len(MAP_H_VALUES))
    )
    empirical_median_rank = np.zeros_like(
        empirical_probability
    )
    predicted_ladder_phase = np.zeros_like(
        empirical_probability,
        dtype=bool,
    )
    predicted_binding_rank = np.zeros_like(
        empirical_probability
    )

    for n_index, n_agents in enumerate(MAP_N_VALUES):
        for h_index, h in enumerate(MAP_H_VALUES):
            rng = np.random.default_rng(
                MASTER_SEED
                + CONFIRMATION_SEED_OFFSET
                + 10_000 * int(n_agents)
                + h_index
            )
            sigma = draw_sigma(
                rng,
                MAP_SAMPLES,
                int(n_agents),
                float(h),
            )
            _, binding_rank, _, _ = realized_ladder_statistics(
                sigma
            )

            empirical_probability[
                n_index,
                h_index,
            ] = float(np.mean(binding_rank > 1))
            empirical_median_rank[
                n_index,
                h_index,
            ] = float(np.median(binding_rank))

            predicted_rank, _gap = quantile_binding_rank(
                int(n_agents),
                float(h),
            )
            predicted_binding_rank[
                n_index,
                h_index,
            ] = predicted_rank
            predicted_ladder_phase[
                n_index,
                h_index,
            ] = predicted_rank > 1

    empirical_majority_phase = empirical_probability > 0.50
    classification_accuracy = float(
        np.mean(
            empirical_majority_phase
            == predicted_ladder_phase
        )
    )

    def cell_value(n_value: int, h_value: float) -> tuple[float, bool, int]:
        n_index = int(np.where(MAP_N_VALUES == n_value)[0][0])
        h_index = int(
            np.where(np.isclose(MAP_H_VALUES, h_value))[0][0]
        )
        return (
            float(empirical_probability[n_index, h_index]),
            bool(predicted_ladder_phase[n_index, h_index]),
            int(predicted_binding_rank[n_index, h_index]),
        )

    original = cell_value(20, 0.15)
    ladder = cell_value(1000, 0.15)
    high_h = cell_value(1000, 0.80)

    p1 = classification_accuracy >= MIN_PHASE_ACCURACY
    p2 = (
        original[0] < 0.50
        and original[1] is False
        and ladder[0] > 0.50
        and ladder[1] is True
        and high_h[0] < 0.50
        and high_h[1] is False
    )

    return {
        "empirical_probability": empirical_probability,
        "empirical_median_rank": empirical_median_rank,
        "predicted_ladder_phase": predicted_ladder_phase,
        "predicted_binding_rank": predicted_binding_rank,
        "classification_accuracy": classification_accuracy,
        "original": original,
        "ladder": ladder,
        "high_h": high_h,
        "p1": p1,
        "p2": p2,
    }


# ---------------------------------------------------------------------------
# Agent dynamics
# ---------------------------------------------------------------------------

def update_one_evaluation(
    population: Population,
    rng: np.random.Generator,
    theta: float,
) -> None:
    shared = population.strategy_shared
    competence = population.competence

    competence[:] = np.where(
        shared,
        competence
        + PASSIVE_RETENTION * (1.0 - competence)
        - COMPETENCE_DECAY * competence,
        competence
        + INDEPENDENT_REBUILD_RATE
        * (1.0 - competence),
    )
    np.clip(competence, 1e-4, 1.0, out=competence)

    shared_fraction = np.mean(shared, axis=1)
    shared_error = (
        (1.0 - shared_fraction)
        * COMMON_SHARED_BIAS
    ) ** 2
    independent_error = (
        (
            shared_fraction[:, None]
            * COMMON_SHARED_BIAS
        ) ** 2
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
        - LIABILITY_RATCHET * shared_fraction[:, None]
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
            np.tanh(
                SELECTION_GAIN * switch_advantage
            ),
        )
    )
    np.clip(
        switch_probability,
        0.0,
        1.0,
        out=switch_probability,
    )

    switches = (
        rng.random(size=shared.shape)
        < switch_probability
    )
    shared[:] = np.where(
        switches,
        ~shared,
        shared,
    )


def make_repeated_population(
    sigma_by_population: np.ndarray,
    trials_per_population: int,
) -> Population:
    sigma = np.repeat(
        sigma_by_population,
        trials_per_population,
        axis=0,
    )
    n_trajectories, n_agents = sigma.shape
    return Population(
        strategy_shared=np.ones(
            (n_trajectories, n_agents),
            dtype=bool,
        ),
        competence=np.full(
            (n_trajectories, n_agents),
            C_STAR,
            dtype=float,
        ),
        sigma_independent=sigma,
    )


def fixed_theta_waiting_times(
    sigma_by_population: np.ndarray,
    theta: float,
    trials_per_population: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    population = make_repeated_population(
        sigma_by_population,
        trials_per_population,
    )

    n_trajectories = population.strategy_shared.shape[0]
    wait = np.full(
        n_trajectories,
        HAZARD_MAX_TIME + 1,
        dtype=int,
    )
    active = np.ones(
        n_trajectories,
        dtype=bool,
    )

    for time_index in range(1, HAZARD_MAX_TIME + 1):
        update_one_evaluation(
            population,
            rng,
            theta,
        )
        escaped = (
            (
                np.mean(
                    population.strategy_shared,
                    axis=1,
                )
                <= SHARED_EXIT_THRESHOLD
            )
            & active
        )
        wait[escaped] = time_index
        active[escaped] = False

        if not np.any(active):
            break

    return wait.reshape(
        len(sigma_by_population),
        trials_per_population,
    )


def estimate_exponential_hazard(
    waiting_times: np.ndarray,
) -> np.ndarray:
    events = waiting_times <= HAZARD_MAX_TIME
    exposure = np.minimum(
        waiting_times,
        HAZARD_MAX_TIME,
    )
    return (
        np.sum(events, axis=1)
        / np.maximum(
            np.sum(exposure, axis=1),
            1,
        )
    )


def run_hazard_measurement() -> dict[str, np.ndarray | float | bool]:
    rng = np.random.default_rng(
        MASTER_SEED
        + CONFIRMATION_SEED_OFFSET
        + 200_000
    )
    sigma_training = draw_sigma(
        rng,
        HAZARD_POPULATIONS,
        HAZARD_N,
        HAZARD_H,
    )

    lambda_by_population = np.zeros(
        (
            HAZARD_POPULATIONS,
            len(HAZARD_THETA_VALUES),
        )
    )
    heldout_survival_mae = np.zeros_like(
        lambda_by_population
    )
    heldout_event_fraction = np.zeros_like(
        lambda_by_population
    )

    for theta_index, theta in enumerate(
        HAZARD_THETA_VALUES
    ):
        waiting = fixed_theta_waiting_times(
            sigma_training,
            float(theta),
            HAZARD_TRIALS_PER_POPULATION,
            seed=(
                MASTER_SEED
                + CONFIRMATION_SEED_OFFSET
                + 300_000
                + theta_index
            ),
        )

        estimation = waiting[
            :,
            :HAZARD_ESTIMATION_TRIALS,
        ]
        validation = waiting[
            :,
            HAZARD_ESTIMATION_TRIALS:,
        ]

        hazard = estimate_exponential_hazard(
            estimation
        )
        lambda_by_population[
            :,
            theta_index,
        ] = hazard

        for population_index in range(
            HAZARD_POPULATIONS
        ):
            observed_survival = np.asarray(
                [
                    np.mean(
                        validation[population_index]
                        > check_time
                    )
                    for check_time in SURVIVAL_CHECK_TIMES
                ]
            )
            predicted_survival = np.exp(
                -hazard[population_index]
                * SURVIVAL_CHECK_TIMES
            )
            heldout_survival_mae[
                population_index,
                theta_index,
            ] = float(
                np.mean(
                    np.abs(
                        observed_survival
                        - predicted_survival
                    )
                )
            )
            heldout_event_fraction[
                population_index,
                theta_index,
            ] = float(
                np.mean(
                    validation[population_index]
                    <= HAZARD_MAX_TIME
                )
            )

    survival_mae = float(
        np.mean(heldout_survival_mae)
    )
    p3 = survival_mae <= MAX_SURVIVAL_MAE

    return {
        "sigma_training": sigma_training,
        "lambda_by_population": lambda_by_population,
        "heldout_survival_mae": heldout_survival_mae,
        "heldout_event_fraction": heldout_event_fraction,
        "survival_mae": survival_mae,
        "p3": p3,
    }


# ---------------------------------------------------------------------------
# Hazard composition into downward sweeps
# ---------------------------------------------------------------------------

def predict_exit_from_hazard(
    theta_values_deep_to_shallow: np.ndarray,
    hazard_values_deep_to_shallow: np.ndarray,
    dwell: int,
) -> float:
    theta_descending = theta_values_deep_to_shallow[::-1]
    hazard_descending = hazard_values_deep_to_shallow[::-1]

    cumulative = np.cumsum(
        dwell * hazard_descending
    )
    crossing = np.where(
        cumulative >= math.log(2.0)
    )[0]

    if len(crossing) == 0:
        return float("nan")

    return float(
        theta_descending[int(crossing[0])]
    )


def simulate_downward_sweep(
    sigma_by_population: np.ndarray,
    dwell: int,
    trials_per_population: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    population = make_repeated_population(
        sigma_by_population,
        trials_per_population,
    )

    n_trajectories = population.strategy_shared.shape[0]
    exit_theta = np.full(
        n_trajectories,
        np.nan,
    )
    active = np.ones(
        n_trajectories,
        dtype=bool,
    )

    for theta in HAZARD_THETA_VALUES[::-1]:
        for _ in range(dwell):
            update_one_evaluation(
                population,
                rng,
                float(theta),
            )
            escaped = (
                (
                    np.mean(
                        population.strategy_shared,
                        axis=1,
                    )
                    <= SHARED_EXIT_THRESHOLD
                )
                & active
            )
            exit_theta[escaped] = theta
            active[escaped] = False

        if not np.any(active):
            break

    return exit_theta.reshape(
        len(sigma_by_population),
        trials_per_population,
    )


def run_sweep_composition(
    hazard_result: dict[str, np.ndarray | float | bool],
) -> dict[str, np.ndarray | float | bool]:
    lambda_by_population = np.asarray(
        hazard_result["lambda_by_population"]
    )

    # Population-conditional predictions on new stochastic trials.
    conditional_prediction = np.zeros(
        (
            HAZARD_POPULATIONS,
            len(DWELL_VALUES),
        )
    )
    conditional_observed = np.zeros_like(
        conditional_prediction
    )

    sigma_training = np.asarray(
        hazard_result["sigma_training"]
    )

    for dwell_index, dwell in enumerate(DWELL_VALUES):
        for population_index in range(
            HAZARD_POPULATIONS
        ):
            conditional_prediction[
                population_index,
                dwell_index,
            ] = predict_exit_from_hazard(
                HAZARD_THETA_VALUES,
                lambda_by_population[
                    population_index
                ],
                int(dwell),
            )

        observed_trials = simulate_downward_sweep(
            sigma_training,
            int(dwell),
            SWEEP_TRIALS_PER_POPULATION,
            seed=(
                MASTER_SEED
                + CONFIRMATION_SEED_OFFSET
                + 400_000
                + dwell_index
            ),
        )
        conditional_observed[
            :,
            dwell_index,
        ] = np.nanmedian(
            observed_trials,
            axis=1,
        )

    conditional_mask = (
        np.isfinite(conditional_prediction)
        & np.isfinite(conditional_observed)
    )
    conditional_mae = float(
        np.mean(
            np.abs(
                conditional_prediction[
                    conditional_mask
                ]
                - conditional_observed[
                    conditional_mask
                ]
            )
        )
    )

    # New-population generalization: use the median hazard curve from the
    # training populations and test against entirely new population draws.
    median_hazard_curve = np.median(
        lambda_by_population,
        axis=0,
    )

    rng = np.random.default_rng(
        MASTER_SEED
        + CONFIRMATION_SEED_OFFSET
        + 500_000
    )
    sigma_new = draw_sigma(
        rng,
        NEW_SWEEP_POPULATIONS,
        HAZARD_N,
        HAZARD_H,
    )

    generalized_prediction = np.zeros(
        len(DWELL_VALUES)
    )
    generalized_observed = np.zeros(
        len(DWELL_VALUES)
    )

    for dwell_index, dwell in enumerate(DWELL_VALUES):
        generalized_prediction[
            dwell_index
        ] = predict_exit_from_hazard(
            HAZARD_THETA_VALUES,
            median_hazard_curve,
            int(dwell),
        )

        new_trials = simulate_downward_sweep(
            sigma_new,
            int(dwell),
            SWEEP_TRIALS_PER_POPULATION,
            seed=(
                MASTER_SEED
                + CONFIRMATION_SEED_OFFSET
                + 600_000
                + dwell_index
            ),
        )
        generalized_observed[
            dwell_index
        ] = float(
            np.nanmedian(new_trials)
        )

    generalized_mae = float(
        np.mean(
            np.abs(
                generalized_prediction
                - generalized_observed
            )
        )
    )

    p4 = generalized_mae <= MAX_SWEEP_THETA_MAE
    p5 = (
        np.all(
            np.diff(
                generalized_prediction
            ) > 0.0
        )
        and np.all(
            np.diff(
                generalized_observed
            ) > 0.0
        )
    )

    return {
        "conditional_prediction": conditional_prediction,
        "conditional_observed": conditional_observed,
        "conditional_mae": conditional_mae,
        "median_hazard_curve": median_hazard_curve,
        "generalized_prediction": generalized_prediction,
        "generalized_observed": generalized_observed,
        "generalized_mae": generalized_mae,
        "p4": p4,
        "p5": p5,
    }


# ---------------------------------------------------------------------------
# Reporting and figures
# ---------------------------------------------------------------------------

def save_results_markdown(
    phase: dict[str, np.ndarray | float | bool],
    hazard: dict[str, np.ndarray | float | bool],
    sweep: dict[str, np.ndarray | float | bool],
) -> None:
    original = phase["original"]
    ladder = phase["ladder"]
    high_h = phase["high_h"]

    lines = [
        "# Paper XXVI final confirmation",
        "",
        "## Phase criterion",
        "",
        f"- Zero-parameter classification accuracy: "
        f"{phase['classification_accuracy']:.3f}",
        f"- Original regime N=20,h=0.15: "
        f"P(k*>1)={original[0]:.3f}, predicted ladder={original[1]}",
        f"- Ladder regime N=1000,h=0.15: "
        f"P(k*>1)={ladder[0]:.3f}, predicted ladder={ladder[1]}",
        f"- High-h regime N=1000,h=0.80: "
        f"P(k*>1)={high_h[0]:.3f}, predicted ladder={high_h[1]}",
        "",
        "## Hazard validation",
        "",
        f"- Held-out exponential survival MAE: "
        f"{hazard['survival_mae']:.4f}",
        "",
        "## Sweep composition",
        "",
        f"- Population-conditional theta MAE: "
        f"{sweep['conditional_mae']:.4f}",
        f"- New-population median theta MAE: "
        f"{sweep['generalized_mae']:.4f}",
        "",
        "| Dwell | Hazard prediction | Observed median |",
        "|---:|---:|---:|",
    ]

    for dwell, predicted, observed in zip(
        DWELL_VALUES,
        sweep["generalized_prediction"],
        sweep["generalized_observed"],
    ):
        lines.append(
            f"| {int(dwell)} | {predicted:.3f} | {observed:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Registered checks",
            "",
            f"- P1 phase reproduction: "
            f"{'PASS' if phase['p1'] else 'FAIL'}",
            f"- P2 anchor regimes: "
            f"{'PASS' if phase['p2'] else 'FAIL'}",
            f"- P3 hazard validation: "
            f"{'PASS' if hazard['p3'] else 'FAIL'}",
            f"- P4 sweep composition: "
            f"{'PASS' if sweep['p4'] else 'FAIL'}",
            f"- P5 dwell ordering: "
            f"{'PASS' if sweep['p5'] else 'FAIL'}",
            "",
            "All claims are [R within the model].",
        ]
    )

    (
        OUTPUT_DIR
        / "final-results.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase = run_phase_map()
    hazard = run_hazard_measurement()
    sweep = run_sweep_composition(hazard)

    print("=" * 100)
    print("PAPER XXVI FINAL CONFIRMATION")
    print("THE ESCAPE LADDER AND THE HAZARD OF RETURN")
    print("=" * 100)
    print(f"master seed                          : {MASTER_SEED}")
    print(f"confirmation seed offset             : {CONFIRMATION_SEED_OFFSET}")
    print(f"consolidated competence c*           : {C_STAR:.3f}")
    print()

    print("ZERO-PARAMETER PHASE CRITERION")
    print(
        f"classification accuracy               : "
        f"{phase['classification_accuracy']:.3f}"
    )
    print(
        f"original N=20,h=.15 P(k*>1)          : "
        f"{phase['original'][0]:.3f}; "
        f"predicted={phase['original'][1]}"
    )
    print(
        f"ladder N=1000,h=.15 P(k*>1)          : "
        f"{phase['ladder'][0]:.3f}; "
        f"predicted={phase['ladder'][1]}"
    )
    print(
        f"high-h N=1000,h=.80 P(k*>1)          : "
        f"{phase['high_h'][0]:.3f}; "
        f"predicted={phase['high_h'][1]}"
    )
    print()

    print("FIXED-THETA HAZARD")
    print(
        f"held-out exponential survival MAE    : "
        f"{hazard['survival_mae']:.4f}"
    )
    print(
        f"{'theta':>9}"
        f"{'median lambda':>16}"
        f"{'median P(event)':>18}"
        f"{'median surv MAE':>18}"
    )
    lambda_by_population = np.asarray(
        hazard["lambda_by_population"]
    )
    event_fraction = np.asarray(
        hazard["heldout_event_fraction"]
    )
    survival_error = np.asarray(
        hazard["heldout_survival_mae"]
    )

    for theta_index, theta in enumerate(
        HAZARD_THETA_VALUES
    ):
        print(
            f"{theta:9.2f}"
            f"{np.median(lambda_by_population[:, theta_index]):16.5f}"
            f"{np.median(event_fraction[:, theta_index]):18.3f}"
            f"{np.median(survival_error[:, theta_index]):18.3f}"
        )
    print()

    print("OUT-OF-SAMPLE SWEEP COMPOSITION")
    print(
        f"population-conditional theta MAE     : "
        f"{sweep['conditional_mae']:.4f}"
    )
    print(
        f"new-population median theta MAE      : "
        f"{sweep['generalized_mae']:.4f}"
    )
    print(
        f"{'dwell':>8}"
        f"{'predicted':>13}"
        f"{'observed':>13}"
        f"{'error':>11}"
    )
    for dwell, predicted, observed in zip(
        DWELL_VALUES,
        sweep["generalized_prediction"],
        sweep["generalized_observed"],
    ):
        print(
            f"{int(dwell):8d}"
            f"{predicted:13.3f}"
            f"{observed:13.3f}"
            f"{abs(predicted-observed):11.3f}"
        )
    print()

    print("REGISTERED CHECKS")
    print(
        f"P1 phase reproduction                : "
        f"{'PASS' if phase['p1'] else 'FAIL'}"
    )
    print(
        f"P2 three anchor regimes              : "
        f"{'PASS' if phase['p2'] else 'FAIL'}"
    )
    print(
        f"P3 hazard validation                 : "
        f"{'PASS' if hazard['p3'] else 'FAIL'}"
    )
    print(
        f"P4 sweep composition                 : "
        f"{'PASS' if sweep['p4'] else 'FAIL'}"
    )
    print(
        f"P5 dwell ordering                    : "
        f"{'PASS' if sweep['p5'] else 'FAIL'}"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Phase probability with zero-parameter boundary overlay.
    empirical_probability = np.asarray(
        phase["empirical_probability"]
    )
    predicted_phase = np.asarray(
        phase["predicted_ladder_phase"],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(9.4, 5.9))
    image = ax.imshow(
        empirical_probability,
        origin="lower",
        aspect="auto",
        extent=(
            MAP_H_VALUES[0],
            MAP_H_VALUES[-1],
            MAP_N_VALUES[0],
            MAP_N_VALUES[-1],
        ),
        vmin=0.0,
        vmax=1.0,
    )
    ax.contour(
        MAP_H_VALUES,
        MAP_N_VALUES,
        predicted_phase,
        levels=[0.5],
    )
    ax.set_xlabel("lognormal heterogeneity h")
    ax.set_ylabel("observer population N")
    ax.set_title(
        "Empirical ladder phase with zero-parameter analytic boundary"
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("P(binding rank k*>1)")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "final-phase-map.png",
        dpi=170,
    )
    plt.close(fig)

    # Median binding rank map.
    fig, ax = plt.subplots(figsize=(9.4, 5.9))
    rank_image = ax.imshow(
        np.asarray(phase["empirical_median_rank"]),
        origin="lower",
        aspect="auto",
        extent=(
            MAP_H_VALUES[0],
            MAP_H_VALUES[-1],
            MAP_N_VALUES[0],
            MAP_N_VALUES[-1],
        ),
    )
    ax.set_xlabel("lognormal heterogeneity h")
    ax.set_ylabel("observer population N")
    ax.set_title(
        "Median binding rank of the escape ladder"
    )
    rank_bar = fig.colorbar(rank_image, ax=ax)
    rank_bar.set_label("median k*")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "final-binding-rank-map.png",
        dpi=170,
    )
    plt.close(fig)

    # Hazard curves.
    fig, ax = plt.subplots(figsize=(9.1, 5.4))
    median_hazard = np.median(
        lambda_by_population,
        axis=0,
    )
    lower_hazard = np.quantile(
        lambda_by_population,
        0.25,
        axis=0,
    )
    upper_hazard = np.quantile(
        lambda_by_population,
        0.75,
        axis=0,
    )
    ax.plot(
        HAZARD_THETA_VALUES,
        median_hazard,
        marker="o",
        label="median estimated hazard",
    )
    ax.fill_between(
        HAZARD_THETA_VALUES,
        lower_hazard,
        upper_hazard,
        alpha=0.25,
        label="interquartile range",
    )
    ax.set_xlabel("fixed theta")
    ax.set_ylabel("escape hazard per evaluation")
    ax.set_title(
        "Fixed-theta escape hazard"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "final-hazard-curves.png",
        dpi=170,
    )
    plt.close(fig)

    # Held-out survival validation.
    fig, ax = plt.subplots(figsize=(9.1, 5.3))
    ax.plot(
        HAZARD_THETA_VALUES,
        np.median(
            survival_error,
            axis=0,
        ),
        marker="o",
        label="median population MAE",
    )
    ax.plot(
        HAZARD_THETA_VALUES,
        np.mean(
            survival_error,
            axis=0,
        ),
        marker="s",
        linestyle="--",
        label="mean population MAE",
    )
    ax.axhline(
        MAX_SURVIVAL_MAE,
        linestyle=":",
        label="registered limit",
    )
    ax.set_xlabel("fixed theta")
    ax.set_ylabel("held-out survival-curve MAE")
    ax.set_title(
        "Exponential hazard is an adequate operational approximation"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "final-hazard-validation.png",
        dpi=170,
    )
    plt.close(fig)

    # Sweep composition.
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ax.plot(
        DWELL_VALUES,
        sweep["generalized_prediction"],
        marker="o",
        label="hazard-composed prediction",
    )
    ax.plot(
        DWELL_VALUES,
        sweep["generalized_observed"],
        marker="s",
        linestyle="--",
        label="new-population observed median",
    )
    ax.set_xscale("log")
    ax.set_xlabel("dwell evaluations per theta")
    ax.set_ylabel("median exit theta")
    ax.set_title(
        "Fixed-theta hazards compose into swept exit thresholds"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "final-sweep-composition.png",
        dpi=170,
    )
    plt.close(fig)

    save_results_markdown(
        phase,
        hazard,
        sweep,
    )

    print()
    print("Saved:")
    for filename in (
        "final-phase-map.png",
        "final-binding-rank-map.png",
        "final-hazard-curves.png",
        "final-hazard-validation.png",
        "final-sweep-composition.png",
        "final-results.md",
    ):
        print(f"  {OUTPUT_DIR / filename}")

    if not all(
        (
            phase["p1"],
            phase["p2"],
            hazard["p3"],
            sweep["p4"],
            sweep["p5"],
        )
    ):
        raise SystemExit(
            "\nOne or more final registered checks failed. "
            "Treat this as a scientific result, not a software error."
        )


if __name__ == "__main__":
    main()
