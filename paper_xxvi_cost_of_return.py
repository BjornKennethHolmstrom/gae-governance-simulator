#!/usr/bin/env python3
"""
Governance as Engineering — The Cost of Returning
==================================================

A deterministic Paper X extension prototype:
competence decay, hysteresis, and endogenous loss of rebuilding capacity.

Purpose
-------
Paper X already models epistemic consolidation as an attractor under
consensus-relative evaluation and a liability ratchet. It encodes difficult
reversion through a small fixed switch-back probability.

This experiment asks a narrower residual question:

    Does explicitly modelled decay of independent-channel competence turn
    consolidation into a wider hysteresis loop, and can second-order decay of
    the training ecology make internal recovery unreachable?

This is a PROTOTYPE / PAPER-GATE simulation, not a historical preregistration.

What is held from Paper X in reduced mean-field form
----------------------------------------------------
Observers are evaluated relative to the ensemble consensus, not against truth.

Let the shared system have a common hidden bias b. Independent systems have
decorrelated error variance sigma_I^2 / c^2, where c is their current competence.

With shared fraction f, the consensus is approximately f*b. Therefore:

    perceived shared error      = (1-f)^2 b^2
    perceived independent error = f^2 b^2 + sigma_I^2 / c^2

As f rises, adopters increasingly constitute the consensus against which they
are judged. The shared system appears more accurate even though its common bias
against truth is unchanged.

Importantly:
    - there is NO direct q_shared = q0 + gamma*f performance bonus;
    - there is NO restart-cost term in the baseline;
    - re-entry difficulty is generated only by low retained competence.

State
-----
    f   : fraction using the shared epistemic infrastructure
    c_I : mean independent competence among current independent observers
    c_S : mean latent independent competence among shared-system adopters

Control parameter
-----------------
    theta : exogenous shared-system advantage, swept upward and downward.
            It may represent cost subsidy, coordination convenience, or
            procurement preference. It is not the liability ratchet.

Factorial
---------
                              no liability ratchet    liability ratchet
    no competence decay               A                     C
    competence decay                  B                     D

The consensus-relative evaluation mechanism is active in every cell.

Hysteresis protocol
-------------------
The model is deterministic. At each theta value, it is iterated to a fixed
point before theta changes. The upward branch begins diverse; the downward
branch begins from the consolidated endpoint of the upward branch.

This avoids the ill-defined claim of strict equilibrium hysteresis in a finite
stochastic system with eventual barrier crossing.

Recovery protocol
-----------------
After a high-theta lock-in period, theta is reduced and a constant internal
training allocation m is offered for a fixed recovery horizon.

Two rebuilding laws are compared:

1. FIXED EFFICIENCY
       training effectiveness is independent of surviving trainer capacity.
   Prediction: recovery remains finite; cost rises and then saturates as latent
   competence approaches its passive floor.

2. TRAINER-DEPENDENT EFFICIENCY
       training effectiveness depends on surviving independent trainer mass.
       Below one institution in an N-equivalent population, internal training
       effectiveness is exactly zero.
   This is an explicit second-order mechanism: competent independent channels
   are required to train new independent channels.
   Prediction: sufficiently long consolidation can cross an internal
   irreversibility boundary.

The trainer threshold is a treatment assumption, not a derived universal law.
"Unreachable" means unreachable by the model's admissible INTERNAL rebuilding
actions; external transplantation of competence is outside the model.

Registered paper-gate checks [R within model]
----------------------------------------------
H1 Competence decay widens the quasi-static hysteresis loop relative to the
   corresponding no-decay cell.

H2 Entry/exit asymmetry is generated without an explicit restart cost:
       theta_exit < theta_entry.
   Competence decay should shift the EXIT threshold more than the entry
   threshold because competence is lost after consolidation.

H3 Under fixed rebuilding efficiency:
       - recovery is finite for every tested lock duration;
       - minimum tested constant-training expenditure rises with lock duration;
       - its late slope is lower than its early slope (concave/saturating).

H4 Under trainer-dependent rebuilding:
       - short lock durations remain recoverable;
       - a non-empty range of longer lock durations becomes internally
         unrecoverable after the trainer ecology falls below critical mass.

OPEN H5 Liability x competence-decay interaction:
   Report the difference-in-differences in loop area. No direction is forced.
   A near-zero result means the mechanisms are approximately additive.

Paper gate
----------
This deserves an independent paper only if:
    - competence decay adds persistent quasi-static loop width;
    - recovery cost or reachability depends materially on time spent locked in;
    - and preferably the second-order arm produces a meaningful internal
      irreversibility boundary.

Otherwise it remains a section/appendix in Paper X's orbit.

Dependencies
------------
Python 3.9+, NumPy, Matplotlib

Run
---
    python3 gae-sim-cost-of-return.py

Outputs
-------
    outputs_cost_of_return/cost-of-return-hysteresis.png
    outputs_cost_of_return/cost-of-return-loop-areas.png
    outputs_cost_of_return/cost-of-return-recovery-cost.png
    outputs_cost_of_return/cost-of-return-trainer-mass.png
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Frozen prototype parameters
# ---------------------------------------------------------------------------

SEED = 20260718
np.random.seed(SEED)

# Consensus-relative evaluation
COMMON_SHARED_BIAS = 0.45
INDEPENDENT_ERROR_STD = 0.12
COST_SHARED = 0.10
COST_INDEPENDENT = 0.20

# Strategy adjustment
SELECTION_GAIN = 8.0
SWITCH_RATE = 0.12

# Capability dynamics
ETA_INDEPENDENT = 0.05
PASSIVE_SHARED_RETENTION = 0.003
DECAY_OFF = 0.0
DECAY_ON = 0.005

# Liability ratchet
RATCHET_OFF = 0.0
RATCHET_ON = 0.15

# Quasi-static theta sweep
THETA_VALUES = np.linspace(-0.70, 0.35, 121)
EQUILIBRIUM_TOL = 1e-10
MAX_EQUILIBRIUM_STEPS = 20_000
SHARED_THRESHOLD = 0.50

# Recovery experiment
THETA_LOCK = 0.25
THETA_RECOVERY = -0.50
LOCK_DURATIONS = np.arange(0, 501, 20)
RECOVERY_HORIZON = 800
RECOVERY_SHARED_TARGET = 0.20
RECOVERY_COMPETENCE_TARGET = 0.70

TRAINING_LEVELS = np.linspace(0.0, 0.50, 101)
ETA_TRAINING = 0.25

# Trainer-dependent rebuilding law
N_EQUIVALENT_INSTITUTIONS = 200
MIN_TRAINER_MASS = 1.0 / N_EQUIVALENT_INSTITUTIONS
TRAINER_REFERENCE_MASS = 0.10

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_cost_of_return"


@dataclass(frozen=True)
class State:
    shared_fraction: float
    independent_competence: float
    shared_latent_competence: float


@dataclass(frozen=True)
class Cell:
    code: str
    label: str
    competence_decay: float
    liability_ratchet: float


CELLS = (
    Cell("A", "no decay / no ratchet", DECAY_OFF, RATCHET_OFF),
    Cell("B", "decay / no ratchet", DECAY_ON, RATCHET_OFF),
    Cell("C", "no decay / ratchet", DECAY_OFF, RATCHET_ON),
    Cell("D", "decay / ratchet", DECAY_ON, RATCHET_ON),
)


@dataclass
class BranchResult:
    cell: Cell
    upward: np.ndarray
    downward: np.ndarray
    entry_threshold: float
    exit_threshold: float
    width: float
    loop_area: float


@dataclass
class RecoveryResult:
    lock_duration: int
    trainer_mass_at_release: float
    fixed_recoverable: bool
    fixed_cost: float
    fixed_training: float
    fixed_time: int
    trainer_recoverable: bool
    trainer_cost: float
    trainer_training: float
    trainer_time: int


def positive_tanh(value: float) -> float:
    """Bounded deterministic best-response flow, exactly zero if disadvantageous."""
    return max(0.0, math.tanh(value))


def perceived_errors(shared_fraction: float, competence: float) -> tuple[float, float]:
    """
    Expected consensus-relative errors.

    Truth is normalized to zero. The shared infrastructure has common hidden
    bias b; independent observers are unbiased in aggregate but dispersed.
    """
    competence_safe = max(competence, 1e-4)
    shared_error = ((1.0 - shared_fraction) * COMMON_SHARED_BIAS) ** 2
    independent_error = (
        (shared_fraction * COMMON_SHARED_BIAS) ** 2
        + INDEPENDENT_ERROR_STD**2 / (competence_safe**2)
    )
    return shared_error, independent_error


def utilities(
    shared_fraction: float,
    competence_for_independent: float,
    theta: float,
    liability_ratchet: float,
) -> tuple[float, float]:
    """
    Shared and independent perceived utilities.

    No direct network-performance bonus and no restart-cost term are included.
    """
    shared_error, independent_error = perceived_errors(
        shared_fraction,
        competence_for_independent,
    )

    utility_shared = (
        -shared_error
        - COST_SHARED
        + theta
    )
    utility_independent = (
        -independent_error
        - COST_INDEPENDENT
        - liability_ratchet * shared_fraction
    )
    return utility_shared, utility_independent


def trainer_multiplier(state: State, law: Literal["fixed", "trainer"]) -> float:
    if law == "fixed":
        return 1.0

    trainer_mass = (
        (1.0 - state.shared_fraction)
        * state.independent_competence
    )
    if trainer_mass < MIN_TRAINER_MASS:
        return 0.0

    return min(
        1.0,
        trainer_mass / TRAINER_REFERENCE_MASS,
    )


def transition(
    state: State,
    theta: float,
    competence_decay: float,
    liability_ratchet: float,
    training_allocation: float = 0.0,
    rebuild_law: Literal["fixed", "trainer"] = "fixed",
) -> State:
    """
    One deterministic population update.

    Competence is transferred with switchers. Current independents rebuild
    competence; shared adopters lose latent independent competence.
    """
    f = state.shared_fraction
    c_i = state.independent_competence
    c_s = state.shared_latent_competence

    # Independent organizations considering adoption use their current c_i.
    utility_s_for_i, utility_i_current = utilities(
        f,
        c_i,
        theta,
        liability_ratchet,
    )
    advantage_i_to_s = utility_s_for_i - utility_i_current

    # Shared adopters considering independence use their retained latent c_s.
    utility_s_current, utility_i_for_s = utilities(
        f,
        c_s,
        theta,
        liability_ratchet,
    )
    advantage_s_to_i = utility_i_for_s - utility_s_current

    flow_i_to_s = (
        SWITCH_RATE
        * (1.0 - f)
        * positive_tanh(SELECTION_GAIN * advantage_i_to_s)
    )
    flow_s_to_i = (
        SWITCH_RATE
        * f
        * positive_tanh(SELECTION_GAIN * advantage_s_to_i)
    )

    # Transfer competence mass with switchers.
    mass_i = (1.0 - f) * c_i
    mass_s = f * c_s

    f_next = float(
        np.clip(
            f + flow_i_to_s - flow_s_to_i,
            0.0,
            1.0,
        )
    )

    mass_i_next = (
        mass_i
        - flow_i_to_s * c_i
        + flow_s_to_i * c_s
    )
    mass_s_next = (
        mass_s
        + flow_i_to_s * c_i
        - flow_s_to_i * c_s
    )

    if 1.0 - f_next > 1e-12:
        c_i_next = mass_i_next / (1.0 - f_next)
    else:
        c_i_next = c_i

    if f_next > 1e-12:
        c_s_next = mass_s_next / f_next
    else:
        c_s_next = c_s

    provisional = State(
        f_next,
        float(np.clip(c_i_next, 0.0, 1.0)),
        float(np.clip(c_s_next, 0.0, 1.0)),
    )
    multiplier = trainer_multiplier(provisional, rebuild_law)

    # Current independent institutions exercise and rebuild their competence.
    independent_rebuild_rate = (
        ETA_INDEPENDENT
        + ETA_TRAINING * training_allocation
    ) * multiplier
    c_i_next += independent_rebuild_rate * (1.0 - c_i_next)

    # Shared adopters retain a small passive floor; active retraining depends
    # on the rebuilding law.
    c_s_next += (
        PASSIVE_SHARED_RETENTION * (1.0 - c_s_next)
        - competence_decay * c_s_next
        + ETA_TRAINING
        * training_allocation
        * multiplier
        * (1.0 - c_s_next)
    )

    return State(
        shared_fraction=f_next,
        independent_competence=float(np.clip(c_i_next, 0.0, 1.0)),
        shared_latent_competence=float(np.clip(c_s_next, 0.0, 1.0)),
    )


def equilibrate(
    initial: State,
    theta: float,
    cell: Cell,
) -> tuple[State, int, bool]:
    state = initial
    for step_index in range(MAX_EQUILIBRIUM_STEPS):
        next_state = transition(
            state,
            theta,
            cell.competence_decay,
            cell.liability_ratchet,
        )
        difference = max(
            abs(next_state.shared_fraction - state.shared_fraction),
            abs(next_state.independent_competence - state.independent_competence),
            abs(next_state.shared_latent_competence - state.shared_latent_competence),
        )
        state = next_state
        if difference < EQUILIBRIUM_TOL:
            return state, step_index + 1, True

    return state, MAX_EQUILIBRIUM_STEPS, False


def threshold_from_branch(
    branch: np.ndarray,
    theta_values: np.ndarray,
) -> float:
    for theta, shared_fraction in zip(theta_values, branch[:, 0]):
        if shared_fraction >= SHARED_THRESHOLD:
            return float(theta)
    return float("nan")


def run_hysteresis_cell(cell: Cell) -> BranchResult:
    state = State(
        shared_fraction=0.01,
        independent_competence=1.0,
        shared_latent_competence=1.0,
    )

    upward_states = []
    for theta in THETA_VALUES:
        state, _steps, converged = equilibrate(state, float(theta), cell)
        if not converged:
            raise RuntimeError(
                f"Upward branch failed to converge for cell {cell.code}, theta={theta:.3f}"
            )
        upward_states.append(
            (
                state.shared_fraction,
                state.independent_competence,
                state.shared_latent_competence,
            )
        )

    downward_states_reversed = []
    for theta in THETA_VALUES[::-1]:
        state, _steps, converged = equilibrate(state, float(theta), cell)
        if not converged:
            raise RuntimeError(
                f"Downward branch failed to converge for cell {cell.code}, theta={theta:.3f}"
            )
        downward_states_reversed.append(
            (
                state.shared_fraction,
                state.independent_competence,
                state.shared_latent_competence,
            )
        )

    upward = np.asarray(upward_states, dtype=float)
    downward = np.asarray(
        downward_states_reversed[::-1],
        dtype=float,
    )

    entry = threshold_from_branch(upward, THETA_VALUES)
    exit_ = threshold_from_branch(downward, THETA_VALUES)
    width = entry - exit_
    loop_area = float(
        np.trapezoid(
            np.abs(upward[:, 0] - downward[:, 0]),
            THETA_VALUES,
        )
    )

    return BranchResult(
        cell=cell,
        upward=upward,
        downward=downward,
        entry_threshold=entry,
        exit_threshold=exit_,
        width=width,
        loop_area=loop_area,
    )


def state_after_lock(lock_duration: int) -> State:
    """
    Generate one common consolidated history using cell D dynamics.
    Rebuilding-law differences are introduced only after release.
    """
    state = State(0.01, 1.0, 1.0)
    cell_d = CELLS[3]

    for _ in range(lock_duration):
        state = transition(
            state,
            THETA_LOCK,
            cell_d.competence_decay,
            cell_d.liability_ratchet,
            training_allocation=0.0,
            rebuild_law="fixed",
        )
    return state


def recovery_success(state: State) -> bool:
    return (
        state.shared_fraction <= RECOVERY_SHARED_TARGET
        and state.independent_competence >= RECOVERY_COMPETENCE_TARGET
    )


def attempt_recovery(
    initial: State,
    training_allocation: float,
    rebuild_law: Literal["fixed", "trainer"],
) -> tuple[bool, int, State]:
    state = initial
    cell_d = CELLS[3]

    if recovery_success(state):
        return True, 0, state

    for step_index in range(1, RECOVERY_HORIZON + 1):
        state = transition(
            state,
            THETA_RECOVERY,
            cell_d.competence_decay,
            cell_d.liability_ratchet,
            training_allocation=training_allocation,
            rebuild_law=rebuild_law,
        )
        if recovery_success(state):
            return True, step_index, state

    return False, RECOVERY_HORIZON, state


def best_constant_recovery(
    initial: State,
    rebuild_law: Literal["fixed", "trainer"],
) -> tuple[bool, float, float, int]:
    """
    Search constant internal training levels.

    The returned cost is the minimum among tested CONSTANT policies:
        cost = training allocation * recovery time.
    It is not claimed to be the globally optimal dynamic-control cost.
    """
    best: tuple[float, float, int] | None = None

    for training in TRAINING_LEVELS:
        recovered, time_used, _state = attempt_recovery(
            initial,
            float(training),
            rebuild_law,
        )
        if not recovered:
            continue

        cost = float(training) * time_used
        candidate = (cost, float(training), time_used)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        return False, float("inf"), float("nan"), RECOVERY_HORIZON

    return True, best[0], best[1], best[2]


def run_recovery_cycle() -> list[RecoveryResult]:
    results = []

    for lock_duration in LOCK_DURATIONS:
        release_state = state_after_lock(int(lock_duration))
        trainer_mass = (
            (1.0 - release_state.shared_fraction)
            * release_state.independent_competence
        )

        fixed = best_constant_recovery(
            release_state,
            "fixed",
        )
        trainer = best_constant_recovery(
            release_state,
            "trainer",
        )

        results.append(
            RecoveryResult(
                lock_duration=int(lock_duration),
                trainer_mass_at_release=float(trainer_mass),
                fixed_recoverable=fixed[0],
                fixed_cost=fixed[1],
                fixed_training=fixed[2],
                fixed_time=fixed[3],
                trainer_recoverable=trainer[0],
                trainer_cost=trainer[1],
                trainer_training=trainer[2],
                trainer_time=trainer[3],
            )
        )

    return results


def first_infeasible_duration(
    results: list[RecoveryResult],
) -> int | None:
    for result in results:
        if not result.trainer_recoverable:
            return result.lock_duration
    return None


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    branch_results = {
        result.cell.code: result
        for result in (
            run_hysteresis_cell(cell)
            for cell in CELLS
        )
    }
    recovery_results = run_recovery_cycle()

    # ------------------------------------------------------------------
    # Registered checks
    # ------------------------------------------------------------------

    h1_no_ratchet = (
        branch_results["B"].loop_area
        > branch_results["A"].loop_area
    )
    h1_ratchet = (
        branch_results["D"].loop_area
        > branch_results["C"].loop_area
    )
    h1 = h1_no_ratchet and h1_ratchet

    h2 = all(
        result.exit_threshold < result.entry_threshold
        for result in branch_results.values()
    )

    entry_shift_no_ratchet = (
        branch_results["B"].entry_threshold
        - branch_results["A"].entry_threshold
    )
    exit_shift_no_ratchet = (
        branch_results["B"].exit_threshold
        - branch_results["A"].exit_threshold
    )
    entry_shift_ratchet = (
        branch_results["D"].entry_threshold
        - branch_results["C"].entry_threshold
    )
    exit_shift_ratchet = (
        branch_results["D"].exit_threshold
        - branch_results["C"].exit_threshold
    )

    exit_moves_more = (
        abs(exit_shift_no_ratchet) > abs(entry_shift_no_ratchet) + 1e-9
        and abs(exit_shift_ratchet) > abs(entry_shift_ratchet) + 1e-9
    )
    h2 = h2 and exit_moves_more

    fixed_finite = all(
        result.fixed_recoverable
        for result in recovery_results
    )
    fixed_positive = np.asarray(
        [
            (result.lock_duration, result.fixed_cost)
            for result in recovery_results
            if result.fixed_cost > 0.0 and math.isfinite(result.fixed_cost)
        ],
        dtype=float,
    )

    if len(fixed_positive) >= 6:
        midpoint = len(fixed_positive) // 2
        early_slope = linear_slope(
            fixed_positive[:midpoint, 0],
            fixed_positive[:midpoint, 1],
        )
        late_slope = linear_slope(
            fixed_positive[midpoint:, 0],
            fixed_positive[midpoint:, 1],
        )
    else:
        early_slope = float("nan")
        late_slope = float("nan")

    fixed_costs = np.asarray(
        [
            result.fixed_cost
            for result in recovery_results
            if math.isfinite(result.fixed_cost)
        ]
    )
    nondecreasing_fraction = float(
        np.mean(np.diff(fixed_costs) >= -1e-9)
    ) if len(fixed_costs) > 1 else 1.0

    h3 = (
        fixed_finite
        and nondecreasing_fraction >= 0.90
        and late_slope < early_slope
    )

    trainer_recoverability = np.asarray(
        [
            result.trainer_recoverable
            for result in recovery_results
        ],
        dtype=bool,
    )
    h4 = (
        np.any(trainer_recoverability)
        and np.any(~trainer_recoverability)
    )

    interaction_loop_area = (
        branch_results["D"].loop_area
        - branch_results["B"].loop_area
        - branch_results["C"].loop_area
        + branch_results["A"].loop_area
    )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    print("=" * 92)
    print("THE COST OF RETURNING")
    print("Paper X extension prototype: competence decay, hysteresis, rebuilding ecology")
    print("=" * 92)
    print(f"seed                               : {SEED}")
    print(f"theta sweep                        : {THETA_VALUES[0]:.2f} to {THETA_VALUES[-1]:.2f}")
    print(f"quasi-static points                : {len(THETA_VALUES)}")
    print(f"restart-cost term                  : 0.000 (excluded from baseline)")
    print(f"shared direct network bonus gamma  : 0.000 (consensus-relative evaluation only)")
    print()

    print("2 x 2 HYSTERESIS FACTORIAL")
    print(
        f"{'cell':<6}"
        f"{'condition':<27}"
        f"{'entry':>9}"
        f"{'exit':>9}"
        f"{'width':>10}"
        f"{'loop area':>12}"
    )
    print("-" * 73)
    for cell in CELLS:
        result = branch_results[cell.code]
        print(
            f"{cell.code:<6}"
            f"{cell.label:<27}"
            f"{result.entry_threshold:9.3f}"
            f"{result.exit_threshold:9.3f}"
            f"{result.width:10.3f}"
            f"{result.loop_area:12.3f}"
        )

    print()
    print("FACTORIAL CONTRASTS")
    print(
        f"decay effect without ratchet       : "
        f"{branch_results['B'].loop_area - branch_results['A'].loop_area:+.3f}"
    )
    print(
        f"decay effect with ratchet          : "
        f"{branch_results['D'].loop_area - branch_results['C'].loop_area:+.3f}"
    )
    print(
        f"ratchet effect without decay       : "
        f"{branch_results['C'].loop_area - branch_results['A'].loop_area:+.3f}"
    )
    print(
        f"ratchet effect with decay          : "
        f"{branch_results['D'].loop_area - branch_results['B'].loop_area:+.3f}"
    )
    print(
        f"OPEN H5 interaction I_H            : "
        f"{interaction_loop_area:+.6f}"
    )
    print()

    print("RECOVERY BY LOCK DURATION")
    print(
        f"{'T_lock':>7}"
        f"{'trainer mass':>14}"
        f"{'fixed cost':>12}"
        f"{'fixed m':>10}"
        f"{'fixed t':>10}"
        f"{'trainer cost':>15}"
        f"{'trainer t':>11}"
    )
    print("-" * 82)
    for result in recovery_results:
        fixed_cost_text = (
            f"{result.fixed_cost:.3f}"
            if result.fixed_recoverable
            else "INF"
        )
        trainer_cost_text = (
            f"{result.trainer_cost:.3f}"
            if result.trainer_recoverable
            else "UNREACH"
        )
        fixed_training_text = (
            f"{result.fixed_training:.3f}"
            if result.fixed_recoverable
            else "-"
        )
        trainer_time_text = (
            str(result.trainer_time)
            if result.trainer_recoverable
            else "-"
        )
        print(
            f"{result.lock_duration:7d}"
            f"{result.trainer_mass_at_release:14.6f}"
            f"{fixed_cost_text:>12}"
            f"{fixed_training_text:>10}"
            f"{result.fixed_time:10d}"
            f"{trainer_cost_text:>15}"
            f"{trainer_time_text:>11}"
        )

    print()
    print("REGISTERED PAPER-GATE CHECKS")
    print(
        f"H1 competence decay widens loop    : "
        f"{'PASS' if h1 else 'FAIL'}"
    )
    print(
        f"H2 no-cost entry/exit asymmetry    : "
        f"{'PASS' if h2 else 'FAIL'}"
    )
    print(
        f"   entry shifts (B-A, D-C)         : "
        f"{entry_shift_no_ratchet:+.3f}, {entry_shift_ratchet:+.3f}"
    )
    print(
        f"   exit shifts  (B-A, D-C)         : "
        f"{exit_shift_no_ratchet:+.3f}, {exit_shift_ratchet:+.3f}"
    )
    print(
        f"H3 fixed rebuilding finite/saturating: "
        f"{'PASS' if h3 else 'FAIL'}"
    )
    print(
        f"   nondecreasing-cost fraction      : "
        f"{nondecreasing_fraction:.3f}"
    )
    print(
        f"   early vs late cost slope         : "
        f"{early_slope:.5f} vs {late_slope:.5f}"
    )
    print(
        f"H4 trainer-dependent boundary       : "
        f"{'PASS' if h4 else 'FAIL'}"
    )
    print(
        f"   first internally infeasible lock : "
        f"{first_infeasible_duration(recovery_results)}"
    )
    print(
        f"OPEN H5 liability x decay I_H       : "
        f"{interaction_loop_area:+.6f}"
    )

    # ------------------------------------------------------------------
    # Figures — separate charts, no subplots
    # ------------------------------------------------------------------

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: all quasi-static branches
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for cell in CELLS:
        result = branch_results[cell.code]
        ax.plot(
            THETA_VALUES,
            result.upward[:, 0],
            label=f"{cell.code} upward",
        )
        ax.plot(
            THETA_VALUES,
            result.downward[:, 0],
            linestyle="--",
            label=f"{cell.code} downward",
        )
    ax.axhline(SHARED_THRESHOLD, linestyle=":")
    ax.set_xlabel("shared-system advantage theta")
    ax.set_ylabel("equilibrium shared-system fraction f")
    ax.set_title(
        "Quasi-static hysteresis: competence decay moves the return threshold"
    )
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "cost-of-return-hysteresis.png",
        dpi=170,
    )
    plt.close(fig)

    # Figure 2: loop areas
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(
        [cell.code for cell in CELLS],
        [branch_results[cell.code].loop_area for cell in CELLS],
    )
    ax.set_xlabel("factorial cell")
    ax.set_ylabel("quasi-static loop area")
    ax.set_title(
        f"Loop area by cell; liability x decay interaction I_H={interaction_loop_area:+.4f}"
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "cost-of-return-loop-areas.png",
        dpi=170,
    )
    plt.close(fig)

    # Figure 3: recovery cost
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fixed_x = [
        result.lock_duration
        for result in recovery_results
        if result.fixed_recoverable
    ]
    fixed_y = [
        result.fixed_cost
        for result in recovery_results
        if result.fixed_recoverable
    ]
    trainer_x = [
        result.lock_duration
        for result in recovery_results
        if result.trainer_recoverable
    ]
    trainer_y = [
        result.trainer_cost
        for result in recovery_results
        if result.trainer_recoverable
    ]
    infeasible_x = [
        result.lock_duration
        for result in recovery_results
        if not result.trainer_recoverable
    ]

    ax.plot(
        fixed_x,
        fixed_y,
        marker="o",
        label="fixed rebuilding efficiency",
    )
    ax.plot(
        trainer_x,
        trainer_y,
        marker="s",
        label="trainer-dependent: recoverable",
    )
    if infeasible_x:
        marker_height = max(fixed_y) * 1.08 if fixed_y else 1.0
        ax.scatter(
            infeasible_x,
            [marker_height] * len(infeasible_x),
            marker="x",
            label="trainer-dependent: internally unreachable",
        )
    ax.set_xlabel("time held in consolidated regime")
    ax.set_ylabel("minimum tested constant-training expenditure")
    ax.set_title(
        "The cost of returning: finite saturation versus loss of the training ecology"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "cost-of-return-recovery-cost.png",
        dpi=170,
    )
    plt.close(fig)

    # Figure 4: trainer mass at release
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(
        [result.lock_duration for result in recovery_results],
        [result.trainer_mass_at_release for result in recovery_results],
        marker="o",
    )
    ax.axhline(
        MIN_TRAINER_MASS,
        linestyle=":",
        label="minimum viable internal trainer mass",
    )
    ax.set_xlabel("time held in consolidated regime")
    ax.set_ylabel("surviving independent trainer mass")
    ax.set_title(
        "Second-order decay: the capacity to rebuild disappears after the channels do"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "cost-of-return-trainer-mass.png",
        dpi=170,
    )
    plt.close(fig)

    print()
    print("Saved:")
    for filename in (
        "cost-of-return-hysteresis.png",
        "cost-of-return-loop-areas.png",
        "cost-of-return-recovery-cost.png",
        "cost-of-return-trainer-mass.png",
    ):
        print(f"  {OUTPUT_DIR / filename}")

    if not all((h1, h2, h3, h4)):
        raise SystemExit(
            "\nOne or more registered paper-gate checks failed. "
            "Treat this as a scientific result, not a software error."
        )


if __name__ == "__main__":
    main()
