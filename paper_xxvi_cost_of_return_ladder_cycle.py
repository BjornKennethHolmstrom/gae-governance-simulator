#!/usr/bin/env python3
"""
Governance as Engineering — The Cost of Returning, Cycle 4
==========================================================

THE ESCAPE LADDER:
tail nucleation, cascade propagation, and institutional time

Background
----------
Paper X models epistemic consolidation through consensus-relative evaluation,
lower shared-system cost, and a liability ratchet. Its rare fixed switch-back
probability stands in for atrophied independent infrastructure.

Earlier extensions replaced that fixed probability with explicit competence
decay. A reduced mean-field model produced a closed-form return threshold, but
finite-agent simulations exited systematically earlier than the population-mean
formula predicted.

Claude's proposed finite-population correction is the escape ladder.

Agent advantage
---------------
For an agent i at shared fraction f:

    A_i(f, theta) = U_I,i - U_S
                  = b^2(1 - 2f) - L1 f - x_i - DeltaC - theta

where:

    x_i = sigma_i^2 / c_i^2
    DeltaC = C_I + L0 - C_S

Each independent defection lowers f by 1/N and raises every remaining agent's
defection advantage by:

    kick = (2b^2 + L1) / N

Sort the consolidated independence penalties:

    y_1 <= y_2 <= ... <= y_N

To reach K = ceil(N/2) defectors deterministically, every step in the cascade
must be viable. The exact frozen-competence ladder threshold is:

    theta_ladder =
        -(b^2 + L1 + DeltaC) - M_N

    M_N =
        max_{k <= K} [y_k - (k-1) kick]

Special cases:
    population mean: replace M_N with E[x]
    pure tail:       replace M_N with y_1
    ladder:          retains both the favorable nucleus and recruitment burden

Important design correction
---------------------------
At the original N=20 and h=0.15, the cascade kick is so large that k*=1 binds
in almost every population. An h-only sweep at N=20 therefore cannot cleanly
separate pure-tail and ladder theories.

This cycle uses:
    1. the original N=20 regime as a tail-dominated control;
    2. an N x h analytic separation map;
    3. large-N stochastic confirmation arms where k*>1 is predicted;
    4. a dwell sweep for the protocol-dependent nucleation layer.

Competence timing correction
----------------------------
Every downward-branch experiment begins fully consolidated with competence at
the exact shared-state fixed point:

    c_* = rho / (rho + delta)

Therefore the realized diagnostic uses the competence relevant at exit, not
competence captured at the end of an earlier upward sweep.

Prediction roles
----------------
A-priori predictions:
    - population-mean formula;
    - quantile pure-tail formula;
    - quantile ladder formula.

Mechanism diagnostics:
    - realized pure-tail formula using each seed's sigma sample;
    - realized ladder formula using each seed's ordered penalties.

Registered predictions [R within model]
----------------------------------------
P1 LADDER SEPARATION
   In the large-N intermediate-heterogeneity arm, realized ladder MAE is lower
   than both mean-field and realized pure-tail MAE for the half-exit threshold.

P2 REGIME SPECIFICITY
   At N=20,h=0.15, pure-tail and ladder predictions are nearly identical.
   At large N and intermediate h, their gap becomes positive and the observed
   threshold lies closer to the ladder.

P3 STAIRCASE / PLATEAU
   In regimes with median binding rank k*>1, the observed first-defection
   threshold is shallower than the half-exit threshold. The gap is positively
   associated with the realized ladder gap M_N-y_1.

P4 N x h PHASE MAP
   The fraction of populations with k*>1 is negligible in the original regime
   and non-zero in the predeclared large-N intermediate-h regime. It returns
   toward zero at very high h when one exceptional channel again suffices.

P5 LOG-DWELL LAW
   At fixed N=20,h=0.15, the exit threshold is approximately affine in log
   dwell time over the frozen dwell values.

OPEN P6 SUBADDITIVITY
   Compute the ladder-predicted liability x decay interaction before dynamic
   confirmation. Compare its sign and magnitude with the stochastic branch
   estimate. A large unexplained residual falsifies the claim that the ladder
   alone explains the interaction.

Paper gate
----------
A separate Paper XXVI is supported only if:
    - the ladder materially improves prediction where pure-tail and mean-field
      differ;
    - intermediate shared-fraction plateaus appear in the predicted regime;
    - and the N/h/dwell dependencies are explained rather than merely fitted.

Otherwise the result remains an analytic extension to Paper X.

Dependencies
------------
Python 3.9+, NumPy, Matplotlib

Run
---
    python3 gae-sim-cost-of-return-ladder-cycle.py

Outputs
-------
    outputs_cost_of_return_ladder/ladder-phase-map.png
    outputs_cost_of_return_ladder/ladder-prediction-errors.png
    outputs_cost_of_return_ladder/ladder-h-sweep.png
    outputs_cost_of_return_ladder/ladder-N-sweep.png
    outputs_cost_of_return_ladder/ladder-staircase.png
    outputs_cost_of_return_ladder/ladder-dwell-law.png
    outputs_cost_of_return_ladder/ladder-interaction.png
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Frozen parameters
# ---------------------------------------------------------------------------

MASTER_SEED = 20260718

COMMON_SHARED_BIAS = 0.45
INDEPENDENT_ERROR_STD = 0.12

COST_SHARED = 0.50
COST_INDEPENDENT = 1.00
LIABILITY_BASE = 0.20
LIABILITY_OFF = 0.0
LIABILITY_ON = 0.15

PASSIVE_RETENTION = 0.003
DECAY_OFF = 0.0
DECAY_ON = 0.005
INDEPENDENT_REBUILD_RATE = 0.05

SELECTION_GAIN = 8.0
SWITCH_RATE = 0.25
MUTATION_RATE = 0.0005

SHARED_THRESHOLD = 0.50
COST_GAP = COST_INDEPENDENT + LIABILITY_BASE - COST_SHARED

# Analytic separation map
MAP_N_VALUES = np.array([20, 50, 100, 200, 500, 1000, 2000])
MAP_H_VALUES = np.array([0.00, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80])
MAP_SAMPLES = 300

# Stochastic confirmation arms.
# Original regime, intermediate ladder regime, and high-h return-to-tail regime.
CONFIRMATION_ARMS = (
    (20, 0.15, "original tail-dominated"),
    (1000, 0.15, "large-N ladder regime"),
    (1000, 0.80, "large-N high-h tail regime"),
)
CONFIRMATION_SEEDS = {
    20: 140,
    1000: 24,
}
CONFIRMATION_DWELL = 50
THETA_STEP = 0.0025
THETA_MARGIN = 0.045

# h sweep at large N
H_SWEEP_N = 1000
H_SWEEP_VALUES = np.array([0.00, 0.10, 0.15, 0.25, 0.80])
H_SWEEP_SEEDS = 18
H_SWEEP_DWELL = 40

# N sweep at intermediate h
N_SWEEP_VALUES = np.array([20, 100, 500, 1000])
N_SWEEP_H = 0.15
N_SWEEP_SEEDS = {
    20: 100,
    100: 50,
    500: 24,
    1000: 18,
}
N_SWEEP_DWELL = 40

# Dwell test in original regime
DWELL_VALUES = np.array([10, 30, 100, 300])
DWELL_N = 20
DWELL_H = 0.15
DWELL_SEEDS = 100

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_cost_of_return_ladder"


@dataclass(frozen=True)
class Cell:
    code: str
    decay: float
    liability_ratchet: float


CELLS = (
    Cell("A", DECAY_OFF, LIABILITY_OFF),
    Cell("B", DECAY_ON, LIABILITY_OFF),
    Cell("C", DECAY_OFF, LIABILITY_ON),
    Cell("D", DECAY_ON, LIABILITY_ON),
)


@dataclass
class Theory:
    mean_exit: float
    quantile_tail_exit: float
    quantile_ladder_exit: float
    quantile_binding_rank: int
    quantile_ladder_gap: float


@dataclass
class DynamicBatch:
    n_agents: int
    h: float
    cell: Cell
    theta_descending: np.ndarray
    branch_descending: np.ndarray
    half_exit: np.ndarray
    first_defection: np.ndarray
    realized_tail_exit: np.ndarray
    realized_ladder_exit: np.ndarray
    realized_binding_rank: np.ndarray
    realized_ladder_gap: np.ndarray
    theory: Theory


def competence_fixed_point(decay: float) -> float:
    if decay <= 0.0:
        return 1.0
    return PASSIVE_RETENTION / (PASSIVE_RETENTION + decay)


def expected_sigma_squared(h: float) -> float:
    return float(
        INDEPENDENT_ERROR_STD**2
        * math.exp(2.0 * h**2)
    )


def base_exit_constant(liability_ratchet: float) -> float:
    return -(
        COMMON_SHARED_BIAS**2
        + liability_ratchet
        + COST_GAP
    )


def cascade_kick(
    n_agents: int,
    liability_ratchet: float,
) -> float:
    return (
        2.0 * COMMON_SHARED_BIAS**2
        + liability_ratchet
    ) / n_agents


def quantile_penalties(
    n_agents: int,
    h: float,
    decay: float,
) -> np.ndarray:
    c_star = competence_fixed_point(decay)
    probabilities = np.arange(
        1,
        n_agents + 1,
        dtype=float,
    ) / (n_agents + 1.0)

    z = np.asarray(
        [
            NormalDist().inv_cdf(float(p))
            for p in probabilities
        ]
    )
    sigma_squared = (
        INDEPENDENT_ERROR_STD**2
        * np.exp(2.0 * h * z)
    )
    return sigma_squared / c_star**2


def ladder_from_penalties(
    penalties_sorted: np.ndarray,
    liability_ratchet: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    penalties_sorted shape: (seed, N), ascending.

    Returns:
        M_N,
        binding rank k* (1-indexed),
        gap M_N-y_1.
    """
    n_agents = penalties_sorted.shape[1]
    k_max = math.ceil(n_agents / 2)
    kick = cascade_kick(
        n_agents,
        liability_ratchet,
    )
    ladder_values = (
        penalties_sorted[:, :k_max]
        - np.arange(k_max, dtype=float)[None, :] * kick
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


def build_theory(
    n_agents: int,
    h: float,
    cell: Cell,
) -> Theory:
    c_star = competence_fixed_point(cell.decay)
    mean_penalty = (
        expected_sigma_squared(h) / c_star**2
    )

    quantile = quantile_penalties(
        n_agents,
        h,
        cell.decay,
    )
    quantile_sorted = np.sort(quantile)[None, :]
    m_value, binding_rank, gap = ladder_from_penalties(
        quantile_sorted,
        cell.liability_ratchet,
    )

    constant = base_exit_constant(
        cell.liability_ratchet
    )
    return Theory(
        mean_exit=constant - mean_penalty,
        quantile_tail_exit=constant - quantile_sorted[0, 0],
        quantile_ladder_exit=constant - float(m_value[0]),
        quantile_binding_rank=int(binding_rank[0]),
        quantile_ladder_gap=float(gap[0]),
    )


def draw_sigma(
    rng: np.random.Generator,
    n_seeds: int,
    n_agents: int,
    h: float,
) -> np.ndarray:
    return (
        INDEPENDENT_ERROR_STD
        * np.exp(
            rng.normal(
                0.0,
                h,
                size=(n_seeds, n_agents),
            )
        )
    )


def realized_theory(
    sigma: np.ndarray,
    cell: Cell,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Competence is evaluated at the exact consolidated fixed point.
    """
    c_star = competence_fixed_point(cell.decay)
    penalties = np.sort(
        sigma**2 / c_star**2,
        axis=1,
    )
    m_value, binding_rank, gap = ladder_from_penalties(
        penalties,
        cell.liability_ratchet,
    )
    constant = base_exit_constant(
        cell.liability_ratchet
    )
    tail_exit = constant - penalties[:, 0]
    ladder_exit = constant - m_value
    return (
        tail_exit,
        ladder_exit,
        binding_rank,
        gap,
    )


def theta_grid_for_arm(
    theory: Theory,
) -> np.ndarray:
    shallow = max(
        theory.quantile_tail_exit,
        theory.quantile_ladder_exit,
    ) + THETA_MARGIN
    deep = min(
        theory.mean_exit,
        theory.quantile_tail_exit,
        theory.quantile_ladder_exit,
    ) - THETA_MARGIN

    # Descending theta: favorable to shared at first, then progressively favors exit.
    count = int(
        math.ceil(
            (shallow - deep) / THETA_STEP
        )
    ) + 1
    return shallow - THETA_STEP * np.arange(count)


def update_competence(
    shared: np.ndarray,
    competence: np.ndarray,
    cell: Cell,
) -> None:
    competence[:] = np.where(
        shared,
        competence
        + PASSIVE_RETENTION * (1.0 - competence)
        - cell.decay * competence,
        competence
        + INDEPENDENT_REBUILD_RATE
        * (1.0 - competence),
    )
    np.clip(
        competence,
        1e-4,
        1.0,
        out=competence,
    )


def one_evaluation(
    shared: np.ndarray,
    competence: np.ndarray,
    sigma: np.ndarray,
    rng: np.random.Generator,
    theta: float,
    cell: Cell,
    mutation_rate: float,
) -> None:
    update_competence(
        shared,
        competence,
        cell,
    )

    f = np.mean(shared, axis=1)
    shared_error = (
        (1.0 - f)
        * COMMON_SHARED_BIAS
    ) ** 2
    independent_error = (
        (f[:, None] * COMMON_SHARED_BIAS) ** 2
        + sigma**2 / competence**2
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
        - cell.liability_ratchet * f[:, None]
    )

    advantage = np.where(
        shared,
        utility_independent - utility_shared[:, None],
        utility_shared[:, None] - utility_independent,
    )
    switch_probability = (
        mutation_rate
        + SWITCH_RATE
        * np.maximum(
            0.0,
            np.tanh(
                SELECTION_GAIN * advantage
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


def interpolate_descending_crossing(
    theta_descending: np.ndarray,
    values: np.ndarray,
    threshold: float,
    direction: str,
) -> np.ndarray:
    """
    Per-seed first crossing while theta moves from shallow to deep.

    direction='below': values cross from > threshold to <= threshold.
    """
    result = np.full(
        values.shape[0],
        np.nan,
    )

    for seed_index in range(values.shape[0]):
        series = values[seed_index]
        if direction == "below":
            condition = series <= threshold
        else:
            condition = series >= threshold

        indices = np.where(condition)[0]
        if len(indices) == 0:
            continue

        index = int(indices[0])
        if index == 0:
            result[seed_index] = theta_descending[0]
            continue

        x0 = theta_descending[index - 1]
        x1 = theta_descending[index]
        y0 = series[index - 1]
        y1 = series[index]

        if abs(y1 - y0) < 1e-12:
            result[seed_index] = x1
            continue

        fraction = (
            threshold - y0
        ) / (y1 - y0)
        result[seed_index] = (
            x0 + fraction * (x1 - x0)
        )

    return result


def simulate_downward_batch(
    n_agents: int,
    h: float,
    cell: Cell,
    n_seeds: int,
    dwell: int,
    seed: int,
    mutation_rate: float = MUTATION_RATE,
    theta_descending_override: np.ndarray | None = None,
) -> DynamicBatch:
    rng = np.random.default_rng(seed)
    sigma = draw_sigma(
        rng,
        n_seeds,
        n_agents,
        h,
    )
    theory = build_theory(
        n_agents,
        h,
        cell,
    )
    (
        realized_tail,
        realized_ladder,
        binding_rank,
        ladder_gap,
    ) = realized_theory(
        sigma,
        cell,
    )

    theta_descending = (
        theta_grid_for_arm(theory)
        if theta_descending_override is None
        else theta_descending_override.copy()
    )

    shared = np.ones(
        (n_seeds, n_agents),
        dtype=bool,
    )
    competence = np.full(
        (n_seeds, n_agents),
        competence_fixed_point(cell.decay),
        dtype=float,
    )

    branch_points = []

    for theta in theta_descending:
        for _ in range(dwell):
            one_evaluation(
                shared,
                competence,
                sigma,
                rng,
                float(theta),
                cell,
                mutation_rate,
            )
        branch_points.append(
            np.mean(shared, axis=1)
        )

    branch_descending = np.stack(
        branch_points,
        axis=1,
    )
    half_exit = interpolate_descending_crossing(
        theta_descending,
        branch_descending,
        SHARED_THRESHOLD,
        "below",
    )

    # One defector. The half-step threshold avoids floating-point ambiguity.
    first_defection_level = (
        1.0 - 0.5 / n_agents
    )
    first_defection = interpolate_descending_crossing(
        theta_descending,
        branch_descending,
        first_defection_level,
        "below",
    )

    return DynamicBatch(
        n_agents=n_agents,
        h=h,
        cell=cell,
        theta_descending=theta_descending,
        branch_descending=branch_descending,
        half_exit=half_exit,
        first_defection=first_defection,
        realized_tail_exit=realized_tail,
        realized_ladder_exit=realized_ladder,
        realized_binding_rank=binding_rank,
        realized_ladder_gap=ladder_gap,
        theory=theory,
    )


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def mae(
    observed: np.ndarray,
    predicted: np.ndarray | float,
) -> float:
    mask = np.isfinite(observed)
    if np.asarray(predicted).ndim == 0:
        residual = (
            observed[mask] - float(predicted)
        )
    else:
        residual = (
            observed[mask]
            - np.asarray(predicted)[mask]
        )
    return float(
        np.mean(
            np.abs(residual)
        )
    )


def mean_ci(
    values: np.ndarray,
) -> tuple[float, float]:
    values = finite(values)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    se = float(
        np.std(values, ddof=1)
        / math.sqrt(len(values))
    )
    return mean, 1.96 * se


def r_squared(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    coefficients = np.polyfit(
        x,
        y,
        1,
    )
    prediction = np.polyval(
        coefficients,
        x,
    )
    residual = float(
        np.sum(
            (y - prediction) ** 2
        )
    )
    total = float(
        np.sum(
            (y - np.mean(y)) ** 2
        )
    )
    if total <= 1e-15:
        return 1.0
    return 1.0 - residual / total


# ---------------------------------------------------------------------------
# Analytic phase map
# ---------------------------------------------------------------------------

def run_phase_map() -> dict[str, np.ndarray]:
    probability_ladder = np.zeros(
        (
            len(MAP_N_VALUES),
            len(MAP_H_VALUES),
        )
    )
    median_binding_rank = np.zeros_like(
        probability_ladder
    )
    median_ladder_gap = np.zeros_like(
        probability_ladder
    )

    for n_index, n_agents in enumerate(
        MAP_N_VALUES
    ):
        rng = np.random.default_rng(
            MASTER_SEED
            + 1_000 * int(n_agents)
        )

        for h_index, h in enumerate(
            MAP_H_VALUES
        ):
            sigma = draw_sigma(
                rng,
                MAP_SAMPLES,
                int(n_agents),
                float(h),
            )
            _, _, binding, gap = realized_theory(
                sigma,
                CELLS[3],
            )
            probability_ladder[
                n_index,
                h_index,
            ] = float(
                np.mean(binding > 1)
            )
            median_binding_rank[
                n_index,
                h_index,
            ] = float(
                np.median(binding)
            )
            median_ladder_gap[
                n_index,
                h_index,
            ] = float(
                np.median(gap)
            )

    return {
        "probability_ladder": probability_ladder,
        "median_binding_rank": median_binding_rank,
        "median_ladder_gap": median_ladder_gap,
    }


# ---------------------------------------------------------------------------
# Dynamic studies
# ---------------------------------------------------------------------------

def run_confirmation_arms() -> list[DynamicBatch]:
    results = []
    for index, (
        n_agents,
        h,
        _label,
    ) in enumerate(CONFIRMATION_ARMS):
        results.append(
            simulate_downward_batch(
                n_agents=int(n_agents),
                h=float(h),
                cell=CELLS[3],
                n_seeds=CONFIRMATION_SEEDS[
                    int(n_agents)
                ],
                dwell=CONFIRMATION_DWELL,
                seed=MASTER_SEED
                + 10_000
                + index,
                mutation_rate=0.0,
            )
        )
    return results


def run_h_sweep() -> list[DynamicBatch]:
    results = []
    for index, h in enumerate(
        H_SWEEP_VALUES
    ):
        results.append(
            simulate_downward_batch(
                n_agents=H_SWEEP_N,
                h=float(h),
                cell=CELLS[3],
                n_seeds=H_SWEEP_SEEDS,
                dwell=H_SWEEP_DWELL,
                seed=MASTER_SEED
                + 20_000
                + index,
                mutation_rate=0.0,
            )
        )
    return results


def run_n_sweep() -> list[DynamicBatch]:
    results = []
    for index, n_agents in enumerate(
        N_SWEEP_VALUES
    ):
        results.append(
            simulate_downward_batch(
                n_agents=int(n_agents),
                h=N_SWEEP_H,
                cell=CELLS[3],
                n_seeds=N_SWEEP_SEEDS[
                    int(n_agents)
                ],
                dwell=N_SWEEP_DWELL,
                seed=MASTER_SEED
                + 30_000
                + index,
                mutation_rate=0.0,
            )
        )
    return results


def run_dwell_sweep() -> dict[str, np.ndarray]:
    median_exit = []
    mean_exit = []
    half_ci = []

    for index, dwell in enumerate(
        DWELL_VALUES
    ):
        batch = simulate_downward_batch(
            n_agents=DWELL_N,
            h=DWELL_H,
            cell=CELLS[3],
            n_seeds=DWELL_SEEDS,
            dwell=int(dwell),
            seed=MASTER_SEED
            + 40_000,
            mutation_rate=MUTATION_RATE,
        )
        mean, ci = mean_ci(
            batch.half_exit
        )
        median_exit.append(
            float(
                np.nanmedian(
                    batch.half_exit
                )
            )
        )
        mean_exit.append(mean)
        half_ci.append(ci)

    return {
        "dwell": DWELL_VALUES.copy(),
        "median_exit": np.asarray(
            median_exit
        ),
        "mean_exit": np.asarray(
            mean_exit
        ),
        "half_ci": np.asarray(
            half_ci
        ),
    }


def run_interaction_study() -> dict[str, object]:
    n_seeds = 160
    n_agents = 20
    h = 0.15

    # Common sigma draws for the analytic interaction.
    rng = np.random.default_rng(
        MASTER_SEED + 50_000
    )
    sigma = draw_sigma(
        rng,
        n_seeds,
        n_agents,
        h,
    )

    ladder_exit = {}
    for cell in CELLS:
        _, ladder, _, _ = realized_theory(
            sigma,
            cell,
        )
        ladder_exit[cell.code] = ladder

    # Entry is invariant across cells in this downward-only comparison,
    # so loop-width interaction is minus the exit-threshold interaction.
    predicted_interaction = (
        -ladder_exit["D"]
        + ladder_exit["B"]
        + ladder_exit["C"]
        - ladder_exit["A"]
    )
    predicted_mean, predicted_ci = mean_ci(
        predicted_interaction
    )

    # A common theta grid and synchronized random streams are required for a
    # valid paired interaction contrast across cells.
    interaction_theories = [
        build_theory(n_agents, h, cell) for cell in CELLS
    ]
    common_shallow = max(
        max(t.quantile_tail_exit, t.quantile_ladder_exit)
        for t in interaction_theories
    ) + THETA_MARGIN
    common_deep = min(
        min(t.mean_exit, t.quantile_tail_exit, t.quantile_ladder_exit)
        for t in interaction_theories
    ) - THETA_MARGIN
    common_count = int(math.ceil((common_shallow-common_deep)/THETA_STEP))+1
    common_theta = common_shallow - THETA_STEP*np.arange(common_count)

    dynamic_exit = {}
    for cell in CELLS:
        batch = simulate_downward_batch(
            n_agents=n_agents,
            h=h,
            cell=cell,
            n_seeds=n_seeds,
            dwell=CONFIRMATION_DWELL,
            seed=MASTER_SEED + 50_000,
            mutation_rate=MUTATION_RATE,
            theta_descending_override=common_theta,
        )
        dynamic_exit[cell.code] = batch.half_exit

    dynamic_interaction = (
        -dynamic_exit["D"]
        + dynamic_exit["B"]
        + dynamic_exit["C"]
        - dynamic_exit["A"]
    )
    dynamic_mean, dynamic_ci = mean_ci(
        dynamic_interaction
    )

    return {
        "predicted_values": predicted_interaction,
        "predicted_mean": predicted_mean,
        "predicted_ci": predicted_ci,
        "dynamic_values": dynamic_interaction,
        "dynamic_mean": dynamic_mean,
        "dynamic_ci": dynamic_ci,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_phase_map(
    matrix: np.ndarray,
    title: str,
    filename: str,
    colorbar_label: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(9.2, 5.8)
    )
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks(
        np.arange(len(MAP_H_VALUES)),
        [f"{value:.2f}" for value in MAP_H_VALUES],
    )
    ax.set_yticks(
        np.arange(len(MAP_N_VALUES)),
        [str(int(value)) for value in MAP_N_VALUES],
    )
    ax.set_xlabel(
        "lognormal heterogeneity h"
    )
    ax.set_ylabel(
        "observer population N"
    )
    ax.set_title(title)
    colorbar = fig.colorbar(
        image,
        ax=ax,
    )
    colorbar.set_label(
        colorbar_label
    )
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

    phase_map = run_phase_map()
    confirmation = run_confirmation_arms()
    h_sweep = run_h_sweep()
    n_sweep = run_n_sweep()
    dwell = run_dwell_sweep()
    interaction = run_interaction_study()

    # ------------------------------------------------------------------
    # Registered checks
    # ------------------------------------------------------------------

    large_ladder_arm = confirmation[1]
    p1 = (
        mae(
            large_ladder_arm.half_exit,
            large_ladder_arm.realized_ladder_exit,
        )
        < mae(
            large_ladder_arm.half_exit,
            large_ladder_arm.realized_tail_exit,
        )
        and mae(
            large_ladder_arm.half_exit,
            large_ladder_arm.realized_ladder_exit,
        )
        < mae(
            large_ladder_arm.half_exit,
            large_ladder_arm.theory.mean_exit,
        )
    )

    original_gap = float(
        np.median(
            confirmation[0].realized_ladder_gap
        )
    )
    large_gap = float(
        np.median(
            large_ladder_arm.realized_ladder_gap
        )
    )
    high_h_gap = float(
        np.median(
            confirmation[2].realized_ladder_gap
        )
    )
    p2 = (
        original_gap < 0.001
        and large_gap > original_gap
        and large_gap > high_h_gap
    )

    plateau_observed = (
        large_ladder_arm.first_defection
        - large_ladder_arm.half_exit
    )
    valid_plateau = (
        np.isfinite(plateau_observed)
        & np.isfinite(
            large_ladder_arm.realized_ladder_gap
        )
    )
    if np.sum(valid_plateau) >= 3:
        plateau_correlation = float(
            np.corrcoef(
                plateau_observed[
                    valid_plateau
                ],
                large_ladder_arm.realized_ladder_gap[
                    valid_plateau
                ],
            )[0, 1]
        )
    else:
        plateau_correlation = float("nan")
    p3 = (
        float(
            np.nanmedian(
                plateau_observed
            )
        ) > 0.0
        and math.isfinite(
            plateau_correlation
        )
        and plateau_correlation > 0.0
    )

    original_map_probability = float(
        phase_map["probability_ladder"][
            np.where(MAP_N_VALUES == 20)[0][0],
            np.where(
                np.isclose(
                    MAP_H_VALUES,
                    0.15,
                )
            )[0][0],
        ]
    )
    large_map_probability = float(
        phase_map["probability_ladder"][
            np.where(MAP_N_VALUES == 1000)[0][0],
            np.where(
                np.isclose(
                    MAP_H_VALUES,
                    0.15,
                )
            )[0][0],
        ]
    )
    high_h_map_probability = float(
        phase_map["probability_ladder"][
            np.where(MAP_N_VALUES == 1000)[0][0],
            np.where(
                np.isclose(
                    MAP_H_VALUES,
                    0.80,
                )
            )[0][0],
        ]
    )
    p4 = (
        original_map_probability < 0.15
        and large_map_probability > 0.50
        and high_h_map_probability
        < large_map_probability
    )

    log_dwell = np.log(
        dwell["dwell"]
    )
    dwell_r2 = r_squared(
        log_dwell,
        dwell["mean_exit"],
    )
    p5 = (
        dwell_r2 >= 0.95
        and np.all(
            np.diff(
                dwell["mean_exit"]
            ) > 0.0
        )
    )

    predicted_interaction = float(
        interaction["predicted_mean"]
    )
    dynamic_interaction = float(
        interaction["dynamic_mean"]
    )
    interaction_residual = (
        dynamic_interaction
        - predicted_interaction
    )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    print("=" * 108)
    print("THE COST OF RETURNING — CYCLE 4")
    print("THE ESCAPE LADDER: TAIL NUCLEATION, CASCADE PROPAGATION, AND INSTITUTIONAL TIME")
    print("=" * 108)
    print(f"master seed                           : {MASTER_SEED}")
    print(f"original regime                       : N=20, h=0.15")
    print(f"large-N separating regime             : N=1000, h=0.15")
    print(f"high-h return-to-tail regime          : N=1000, h=0.80")
    print()

    print("ANALYTIC N x h SEPARATION MAP")
    print(
        f"P(k*>1), original N=20,h=.15         : "
        f"{original_map_probability:.3f}"
    )
    print(
        f"P(k*>1), large N=1000,h=.15          : "
        f"{large_map_probability:.3f}"
    )
    print(
        f"P(k*>1), high h N=1000,h=.80         : "
        f"{high_h_map_probability:.3f}"
    )
    print()

    print("DYNAMIC CONFIRMATION ARMS")
    print(
        f"{'arm':<30}"
        f"{'exit obs':>10}"
        f"{'mean MAE':>11}"
        f"{'tail MAE':>11}"
        f"{'ladder MAE':>12}"
        f"{'median k*':>11}"
        f"{'gap':>9}"
        f"{'plateau':>10}"
    )
    print("-" * 106)

    for batch, arm in zip(
        confirmation,
        CONFIRMATION_ARMS,
    ):
        label = arm[2]
        plateau = (
            batch.first_defection
            - batch.half_exit
        )
        print(
            f"{label:<30}"
            f"{np.nanmedian(batch.half_exit):10.3f}"
            f"{mae(batch.half_exit, batch.theory.mean_exit):11.4f}"
            f"{mae(batch.half_exit, batch.realized_tail_exit):11.4f}"
            f"{mae(batch.half_exit, batch.realized_ladder_exit):12.4f}"
            f"{np.median(batch.realized_binding_rank):11.1f}"
            f"{np.median(batch.realized_ladder_gap):9.4f}"
            f"{np.nanmedian(plateau):10.4f}"
        )

    print()
    print(
        f"large-arm plateau/gap correlation     : "
        f"{plateau_correlation:.3f}"
    )

    print()
    print("LARGE-N HETEROGENEITY SWEEP")
    print(
        f"{'h':>7}"
        f"{'exit obs':>11}"
        f"{'mean':>11}"
        f"{'tail':>11}"
        f"{'ladder':>11}"
        f"{'k*':>8}"
        f"{'gap':>9}"
    )
    for batch in h_sweep:
        print(
            f"{batch.h:7.2f}"
            f"{np.nanmedian(batch.half_exit):11.3f}"
            f"{batch.theory.mean_exit:11.3f}"
            f"{np.median(batch.realized_tail_exit):11.3f}"
            f"{np.median(batch.realized_ladder_exit):11.3f}"
            f"{np.median(batch.realized_binding_rank):8.1f}"
            f"{np.median(batch.realized_ladder_gap):9.4f}"
        )

    print()
    print("N SWEEP AT h=0.15")
    print(
        f"{'N':>7}"
        f"{'exit obs':>11}"
        f"{'mean':>11}"
        f"{'tail':>11}"
        f"{'ladder':>11}"
        f"{'k*':>8}"
        f"{'gap':>9}"
    )
    for batch in n_sweep:
        print(
            f"{batch.n_agents:7d}"
            f"{np.nanmedian(batch.half_exit):11.3f}"
            f"{batch.theory.mean_exit:11.3f}"
            f"{np.median(batch.realized_tail_exit):11.3f}"
            f"{np.median(batch.realized_ladder_exit):11.3f}"
            f"{np.median(batch.realized_binding_rank):8.1f}"
            f"{np.median(batch.realized_ladder_gap):9.4f}"
        )

    print()
    print("DWELL LAW")
    print(
        f"{'dwell':>8}"
        f"{'median exit':>14}"
        f"{'mean exit':>12}"
        f"{'95% half-CI':>14}"
    )
    for index, dwell_value in enumerate(
        dwell["dwell"]
    ):
        print(
            f"{int(dwell_value):8d}"
            f"{dwell['median_exit'][index]:14.4f}"
            f"{dwell['mean_exit'][index]:12.4f}"
            f"{dwell['half_ci'][index]:14.4f}"
        )
    dwell_coefficients = np.polyfit(
        log_dwell,
        dwell["mean_exit"],
        1,
    )
    print(
        f"theta_exit ~= a + s ln(T), slope s  : "
        f"{dwell_coefficients[0]:.5f}"
    )
    print(
        f"log-dwell R^2                        : "
        f"{dwell_r2:.4f}"
    )

    print()
    print("OPEN LIABILITY x DECAY INTERACTION")
    print(
        f"ladder-predicted I_H                 : "
        f"{interaction['predicted_mean']:+.6f} "
        f"+/- {interaction['predicted_ci']:.6f}"
    )
    print(
        f"dynamic I_H                          : "
        f"{interaction['dynamic_mean']:+.6f} "
        f"+/- {interaction['dynamic_ci']:.6f}"
    )
    print(
        f"dynamic minus ladder residual        : "
        f"{interaction_residual:+.6f}"
    )

    print()
    print("DESIGN-FROZEN CHECKS")
    print(
        f"P1 ladder wins in separating regime  : "
        f"{'PASS' if p1 else 'FAIL'}"
    )
    print(
        f"P2 regime specificity                : "
        f"{'PASS' if p2 else 'FAIL'}"
    )
    print(
        f"P3 predicted partial plateau         : "
        f"{'PASS' if p3 else 'FAIL'}"
    )
    print(
        f"P4 N x h phase structure             : "
        f"{'PASS' if p4 else 'FAIL'}"
    )
    print(
        f"P5 log-dwell law                     : "
        f"{'PASS' if p5 else 'FAIL'}"
    )
    print(
        f"OPEN P6 ladder explains interaction  : "
        f"{'CLOSE' if abs(interaction_residual) <= 0.001 else 'RESIDUAL REMAINS'}"
    )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_phase_map(
        phase_map["probability_ladder"],
        "Where does propagation, not just the best channel, govern return?",
        "ladder-phase-map.png",
        "P(binding rank k*>1)",
    )

    # Prediction errors by confirmation arm.
    labels = [
        arm[2]
        for arm in CONFIRMATION_ARMS
    ]
    mean_errors = [
        mae(
            batch.half_exit,
            batch.theory.mean_exit,
        )
        for batch in confirmation
    ]
    tail_errors = [
        mae(
            batch.half_exit,
            batch.realized_tail_exit,
        )
        for batch in confirmation
    ]
    ladder_errors = [
        mae(
            batch.half_exit,
            batch.realized_ladder_exit,
        )
        for batch in confirmation
    ]

    positions = np.arange(
        len(labels)
    )
    width = 0.25
    fig, ax = plt.subplots(
        figsize=(10.5, 5.6)
    )
    ax.bar(
        positions - width,
        mean_errors,
        width,
        label="population mean",
    )
    ax.bar(
        positions,
        tail_errors,
        width,
        label="pure favorable tail",
    )
    ax.bar(
        positions + width,
        ladder_errors,
        width,
        label="escape ladder",
    )
    ax.set_xticks(
        positions,
        labels,
        rotation=12,
    )
    ax.set_ylabel(
        "half-exit threshold MAE"
    )
    ax.set_title(
        "The ladder matters only where the first defector cannot recruit alone"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR
        / "ladder-prediction-errors.png",
        dpi=170,
    )
    plt.close(fig)

    # h sweep.
    fig, ax = plt.subplots(
        figsize=(9.5, 5.5)
    )
    h_values = np.asarray(
        [batch.h for batch in h_sweep]
    )
    observed = np.asarray(
        [
            np.nanmedian(
                batch.half_exit
            )
            for batch in h_sweep
        ]
    )
    mean_pred = np.asarray(
        [
            batch.theory.mean_exit
            for batch in h_sweep
        ]
    )
    tail_pred = np.asarray(
        [
            np.median(
                batch.realized_tail_exit
            )
            for batch in h_sweep
        ]
    )
    ladder_pred = np.asarray(
        [
            np.median(
                batch.realized_ladder_exit
            )
            for batch in h_sweep
        ]
    )
    ax.plot(
        h_values,
        observed,
        marker="o",
        label="observed half-exit",
    )
    ax.plot(
        h_values,
        mean_pred,
        marker="s",
        linestyle="--",
        label="population mean",
    )
    ax.plot(
        h_values,
        tail_pred,
        marker="^",
        linestyle=":",
        label="pure tail",
    )
    ax.plot(
        h_values,
        ladder_pred,
        marker="x",
        label="ladder",
    )
    ax.set_xlabel(
        "lognormal heterogeneity h"
    )
    ax.set_ylabel(
        "exit threshold theta"
    )
    ax.set_title(
        "Intermediate heterogeneity separates tail nucleation from cascade propagation"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "ladder-h-sweep.png",
        dpi=170,
    )
    plt.close(fig)

    # N sweep.
    fig, ax = plt.subplots(
        figsize=(9.5, 5.5)
    )
    n_values = np.asarray(
        [
            batch.n_agents
            for batch in n_sweep
        ]
    )
    observed = np.asarray(
        [
            np.nanmedian(
                batch.half_exit
            )
            for batch in n_sweep
        ]
    )
    mean_pred = np.asarray(
        [
            batch.theory.mean_exit
            for batch in n_sweep
        ]
    )
    tail_pred = np.asarray(
        [
            np.median(
                batch.realized_tail_exit
            )
            for batch in n_sweep
        ]
    )
    ladder_pred = np.asarray(
        [
            np.median(
                batch.realized_ladder_exit
            )
            for batch in n_sweep
        ]
    )
    ax.plot(
        n_values,
        observed,
        marker="o",
        label="observed half-exit",
    )
    ax.plot(
        n_values,
        mean_pred,
        marker="s",
        linestyle="--",
        label="population mean",
    )
    ax.plot(
        n_values,
        tail_pred,
        marker="^",
        linestyle=":",
        label="pure tail",
    )
    ax.plot(
        n_values,
        ladder_pred,
        marker="x",
        label="ladder",
    )
    ax.set_xscale("log")
    ax.set_xlabel(
        "observer population N"
    )
    ax.set_ylabel(
        "exit threshold theta"
    )
    ax.set_title(
        "More possible nuclei, but weaker recruitment per defection"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "ladder-N-sweep.png",
        dpi=170,
    )
    plt.close(fig)

    # Representative staircase: seed nearest median ladder gap in large arm.
    representative_index = int(
        np.argsort(
            large_ladder_arm.realized_ladder_gap
        )[
            len(
                large_ladder_arm.realized_ladder_gap
            ) // 2
        ]
    )
    fig, ax = plt.subplots(
        figsize=(9.2, 5.3)
    )
    ax.step(
        large_ladder_arm.theta_descending,
        large_ladder_arm.branch_descending[
            representative_index
        ],
        where="post",
        label="shared fraction",
    )
    ax.axhline(
        0.5,
        linestyle=":",
        label="half-exit",
    )
    ax.axvline(
        large_ladder_arm.realized_tail_exit[
            representative_index
        ],
        linestyle="--",
        label="pure-tail threshold",
    )
    ax.axvline(
        large_ladder_arm.realized_ladder_exit[
            representative_index
        ],
        linestyle=":",
        label="ladder threshold",
    )
    ax.invert_xaxis()
    ax.set_xlabel(
        "theta, moving from shared-favorable to independence-favorable"
    )
    ax.set_ylabel(
        "shared-system fraction"
    )
    ax.set_title(
        "A strong nucleus can defect before enough followers can complete the escape"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "ladder-staircase.png",
        dpi=170,
    )
    plt.close(fig)

    # Dwell law.
    fig, ax = plt.subplots(
        figsize=(8.8, 5.2)
    )
    ax.errorbar(
        log_dwell,
        dwell["mean_exit"],
        yerr=dwell["half_ci"],
        marker="o",
        capsize=4,
        label="observed mean exit",
    )
    fitted = np.polyval(
        dwell_coefficients,
        log_dwell,
    )
    ax.plot(
        log_dwell,
        fitted,
        linestyle="--",
        label=f"linear in ln T, R²={dwell_r2:.3f}",
    )
    ax.set_xlabel(
        "ln(dwell evaluations)"
    )
    ax.set_ylabel(
        "exit threshold theta"
    )
    ax.set_title(
        "Institutional time shifts the measured escape threshold"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "ladder-dwell-law.png",
        dpi=170,
    )
    plt.close(fig)

    # Interaction.
    fig, ax = plt.subplots(
        figsize=(7.8, 5.0)
    )
    ax.bar(
        ["ladder prediction", "dynamic branch"],
        [
            interaction["predicted_mean"],
            interaction["dynamic_mean"],
        ],
        yerr=[
            interaction["predicted_ci"],
            interaction["dynamic_ci"],
        ],
        capsize=5,
    )
    ax.axhline(
        0.0,
        linestyle=":",
    )
    ax.set_ylabel(
        "liability x decay interaction I_H"
    )
    ax.set_title(
        "Does changing the binding rank explain the surviving subadditivity?"
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "ladder-interaction.png",
        dpi=170,
    )
    plt.close(fig)

    print()
    print("Saved:")
    for filename in (
        "ladder-phase-map.png",
        "ladder-prediction-errors.png",
        "ladder-h-sweep.png",
        "ladder-N-sweep.png",
        "ladder-staircase.png",
        "ladder-dwell-law.png",
        "ladder-interaction.png",
    ):
        print(
            f"  {OUTPUT_DIR / filename}"
        )

    if not all((p1, p2, p3, p4, p5)):
        print(
            "\nOne or more registered predictions failed. "
            "This is a scientific result, not a software error."
        )


if __name__ == "__main__":
    main()
