from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from .phase_e_sim_backend import theta_periods, wrap_theta


PHASE_E_OPTIMIZER_VERSION = "phase_e_optimizers_v0_2"


@dataclass(frozen=True)
class SPSAConfig:
    max_evaluations: int = 100
    a: float = 0.15
    c: float = 0.1
    alpha: float = 0.602
    gamma: float = 0.101
    stability_constant: float = 10.0
    blocking: bool = False


@dataclass(frozen=True)
class COBYLAConfig:
    max_evaluations: int = 500
    rhobeg: float = 0.5
    tol: float = 1e-4
    catol: float = 1e-6


@dataclass(frozen=True)
class OptimizerResult:
    status: str
    initial_theta: tuple[float, ...]
    final_theta: tuple[float, ...]
    best_theta: tuple[float, ...]
    final_value: float
    best_value: float
    num_evaluations: int
    optimizer: str
    runner_version: str


@dataclass(frozen=True)
class OptimizerCheckpoint:
    requested_max_evaluations: int
    num_evaluations: int
    iteration: int
    current_theta: tuple[float, ...]
    best_theta: tuple[float, ...]
    current_value: float
    best_value: float


@dataclass(frozen=True)
class CheckpointedOptimizerResult:
    status: str
    initial_theta: tuple[float, ...]
    final_theta: tuple[float, ...]
    best_theta: tuple[float, ...]
    final_value: float
    best_value: float
    num_evaluations: int
    checkpoints: tuple[OptimizerCheckpoint, ...]
    optimizer: str
    runner_version: str


def make_paired_initial_thetas(depth: int, starts: int, seed: int) -> list[tuple[float, ...]]:
    rng = np.random.default_rng(int(seed))
    periods = theta_periods(int(depth))
    out: list[tuple[float, ...]] = []
    for _ in range(int(starts)):
        out.append(tuple(float(rng.uniform(0.0, period)) for period in periods))
    return out


def spsa_initialize_state(
    objective: Callable[[tuple[float, ...]], float],
    initial_theta: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    theta = np.asarray(wrap_theta(initial_theta), dtype=float)
    rng = np.random.default_rng(int(seed))
    current_value = float(objective(tuple(theta)))
    return {
        "status": "initialized",
        "optimizer": "spsa_resume_state",
        "runner_version": PHASE_E_OPTIMIZER_VERSION,
        "seed": int(seed),
        "initial_theta": [float(x) for x in wrap_theta(initial_theta)],
        "current_theta": [float(x) for x in theta],
        "best_theta": [float(x) for x in theta],
        "current_value": current_value,
        "best_value": current_value,
        "num_evaluations": 1,
        "iteration": 0,
        "rng_state": rng.bit_generator.state,
    }


def spsa_run_state(
    objective: Callable[[tuple[float, ...]], float],
    state: dict[str, Any],
    config: SPSAConfig,
    stop_at_evaluations: int | None = None,
) -> dict[str, Any]:
    stop = int(config.max_evaluations if stop_at_evaluations is None else stop_at_evaluations)
    if stop > int(config.max_evaluations):
        raise ValueError("stop_at_evaluations cannot exceed config.max_evaluations.")
    if stop < int(state["num_evaluations"]):
        raise ValueError("stop_at_evaluations cannot be lower than the stored evaluation count.")

    theta = np.asarray(state["current_theta"], dtype=float)
    periods = np.asarray(theta_periods(theta.size // 2), dtype=float)
    best_theta = np.asarray(state["best_theta"], dtype=float)
    current_value = float(state["current_value"])
    best_value = float(state["best_value"])
    evaluations = int(state["num_evaluations"])
    iteration = int(state["iteration"])
    rng = np.random.default_rng(int(state["seed"]))
    rng.bit_generator.state = state["rng_state"]

    while evaluations + 3 <= stop and evaluations < int(config.max_evaluations):
        ak = float(config.a / ((iteration + 1 + config.stability_constant) ** config.alpha))
        ck = float(config.c / ((iteration + 1) ** config.gamma))
        delta = rng.choice(np.asarray([-1.0, 1.0]), size=theta.size)
        theta_plus = np.mod(theta + ck * delta, periods)
        theta_minus = np.mod(theta - ck * delta, periods)
        value_plus = float(objective(tuple(theta_plus)))
        value_minus = float(objective(tuple(theta_minus)))
        evaluations += 2

        gradient = ((value_plus - value_minus) / (2.0 * ck)) * delta
        candidate = np.mod(theta - ak * gradient, periods)

        if config.blocking and evaluations < config.max_evaluations:
            candidate_value = float(objective(tuple(candidate)))
            evaluations += 1
            if candidate_value <= current_value:
                theta = candidate
                current_value = candidate_value
            if candidate_value < best_value:
                best_theta = candidate.copy()
                best_value = candidate_value
        else:
            theta = candidate
            current_value = float(objective(tuple(theta)))
            evaluations += 1
            if current_value < best_value:
                best_theta = theta.copy()
                best_value = current_value

        iteration += 1

    return {
        "status": "complete" if evaluations + 3 > int(config.max_evaluations) or evaluations >= int(config.max_evaluations) else "partial",
        "optimizer": "spsa_resume_state",
        "runner_version": PHASE_E_OPTIMIZER_VERSION,
        "seed": int(state["seed"]),
        "initial_theta": [float(x) for x in state["initial_theta"]],
        "current_theta": [float(x) for x in theta],
        "best_theta": [float(x) for x in best_theta],
        "current_value": float(current_value),
        "best_value": float(best_value),
        "num_evaluations": int(evaluations),
        "iteration": int(iteration),
        "rng_state": rng.bit_generator.state,
    }


def spsa_result_from_state(state: dict[str, Any], optimizer: str = "spsa_resumed") -> OptimizerResult:
    return OptimizerResult(
        status="complete",
        initial_theta=tuple(float(x) for x in state["initial_theta"]),
        final_theta=tuple(float(x) for x in state["current_theta"]),
        best_theta=tuple(float(x) for x in state["best_theta"]),
        final_value=float(state["current_value"]),
        best_value=float(state["best_value"]),
        num_evaluations=int(state["num_evaluations"]),
        optimizer=optimizer,
        runner_version=PHASE_E_OPTIMIZER_VERSION,
    )


def spsa_minimize(
    objective: Callable[[tuple[float, ...]], float],
    initial_theta: Sequence[float],
    config: SPSAConfig,
    seed: int,
) -> OptimizerResult:
    theta = np.asarray(wrap_theta(initial_theta), dtype=float)
    periods = np.asarray(theta_periods(theta.size // 2), dtype=float)
    rng = np.random.default_rng(int(seed))

    current_value = float(objective(tuple(theta)))
    evaluations = 1
    best_theta = theta.copy()
    best_value = current_value

    # This implementation counts the two SPSA perturbation evaluations plus
    # the post-update objective check. Keeping the count inside the frozen
    # budget matters because Phase E compares objectives under equal budgets.
    iterations = max(0, (int(config.max_evaluations) - evaluations) // 3)
    for k in range(iterations):
        ak = float(config.a / ((k + 1 + config.stability_constant) ** config.alpha))
        ck = float(config.c / ((k + 1) ** config.gamma))
        delta = rng.choice(np.asarray([-1.0, 1.0]), size=theta.size)
        theta_plus = np.mod(theta + ck * delta, periods)
        theta_minus = np.mod(theta - ck * delta, periods)
        value_plus = float(objective(tuple(theta_plus)))
        value_minus = float(objective(tuple(theta_minus)))
        evaluations += 2

        gradient = ((value_plus - value_minus) / (2.0 * ck)) * delta
        candidate = np.mod(theta - ak * gradient, periods)

        if config.blocking and evaluations < config.max_evaluations:
            candidate_value = float(objective(tuple(candidate)))
            evaluations += 1
            if candidate_value <= current_value:
                theta = candidate
                current_value = candidate_value
            if candidate_value < best_value:
                best_theta = candidate.copy()
                best_value = candidate_value
        else:
            theta = candidate
            current_value = float(objective(tuple(theta)))
            evaluations += 1
            if current_value < best_value:
                best_theta = theta.copy()
                best_value = current_value

        if evaluations >= config.max_evaluations:
            break

    return OptimizerResult(
        status="complete",
        initial_theta=tuple(float(x) for x in wrap_theta(initial_theta)),
        final_theta=tuple(float(x) for x in theta),
        best_theta=tuple(float(x) for x in best_theta),
        final_value=float(current_value),
        best_value=float(best_value),
        num_evaluations=int(evaluations),
        optimizer="spsa",
        runner_version=PHASE_E_OPTIMIZER_VERSION,
    )


def cobyla_minimize(
    objective: Callable[[tuple[float, ...]], float],
    initial_theta: Sequence[float],
    config: COBYLAConfig,
) -> OptimizerResult:
    periods = np.asarray(theta_periods(len(initial_theta) // 2), dtype=float)
    initial = np.asarray(wrap_theta(initial_theta), dtype=float)
    evaluations = 0
    best_theta = initial.copy()
    best_value = float("inf")

    def wrapped_objective(raw_theta: np.ndarray) -> float:
        nonlocal evaluations, best_theta, best_value
        theta = np.mod(np.asarray(raw_theta, dtype=float), periods)
        value = float(objective(tuple(float(x) for x in theta)))
        evaluations += 1
        if value < best_value:
            best_value = value
            best_theta = theta.copy()
        return value

    constraints = []
    for index, period in enumerate(periods):
        constraints.append({"type": "ineq", "fun": lambda x, i=index: float(x[i])})
        constraints.append({"type": "ineq", "fun": lambda x, i=index, p=float(period): float(p - x[i])})

    try:
        result = minimize(
            wrapped_objective,
            initial,
            method="COBYLA",
            constraints=constraints,
            options={
                "maxiter": int(config.max_evaluations),
                "rhobeg": float(config.rhobeg),
                "tol": float(config.tol),
                "catol": float(config.catol),
                "disp": False,
            },
        )
        final_theta = np.mod(np.asarray(result.x, dtype=float), periods)
        final_value = float(result.fun)
        if final_value < best_value:
            best_value = final_value
            best_theta = final_theta.copy()
        status = "complete" if bool(result.success) else "budget_exhausted_or_not_converged"
    except Exception:
        final_theta = initial.copy()
        final_value = float("nan")
        status = "failed"
        if not np.isfinite(best_value):
            best_value = float("nan")

    return OptimizerResult(
        status=status,
        initial_theta=tuple(float(x) for x in initial),
        final_theta=tuple(float(x) for x in final_theta),
        best_theta=tuple(float(x) for x in best_theta),
        final_value=float(final_value),
        best_value=float(best_value),
        num_evaluations=int(evaluations),
        optimizer="cobyla",
        runner_version=PHASE_E_OPTIMIZER_VERSION,
    )


def spsa_minimize_checkpoints(
    objective: Callable[[tuple[float, ...]], float],
    initial_theta: Sequence[float],
    config: SPSAConfig,
    checkpoints: Sequence[int],
    seed: int,
) -> CheckpointedOptimizerResult:
    requested = tuple(sorted({int(value) for value in checkpoints}))
    if not requested:
        raise ValueError("At least one checkpoint is required.")
    if requested[-1] > int(config.max_evaluations):
        raise ValueError("All checkpoints must be <= config.max_evaluations.")
    if requested[0] < 1:
        raise ValueError("Checkpoint budgets must be positive.")

    theta = np.asarray(wrap_theta(initial_theta), dtype=float)
    periods = np.asarray(theta_periods(theta.size // 2), dtype=float)
    rng = np.random.default_rng(int(seed))

    current_value = float(objective(tuple(theta)))
    evaluations = 1
    best_theta = theta.copy()
    best_value = current_value
    states: list[tuple[int, int, np.ndarray, np.ndarray, float, float]] = [
        (0, evaluations, theta.copy(), best_theta.copy(), current_value, best_value)
    ]

    iterations = max(0, (int(config.max_evaluations) - evaluations) // 3)
    for k in range(iterations):
        ak = float(config.a / ((k + 1 + config.stability_constant) ** config.alpha))
        ck = float(config.c / ((k + 1) ** config.gamma))
        delta = rng.choice(np.asarray([-1.0, 1.0]), size=theta.size)
        theta_plus = np.mod(theta + ck * delta, periods)
        theta_minus = np.mod(theta - ck * delta, periods)
        value_plus = float(objective(tuple(theta_plus)))
        value_minus = float(objective(tuple(theta_minus)))
        evaluations += 2

        gradient = ((value_plus - value_minus) / (2.0 * ck)) * delta
        candidate = np.mod(theta - ak * gradient, periods)

        if config.blocking and evaluations < config.max_evaluations:
            candidate_value = float(objective(tuple(candidate)))
            evaluations += 1
            if candidate_value <= current_value:
                theta = candidate
                current_value = candidate_value
            if candidate_value < best_value:
                best_theta = candidate.copy()
                best_value = candidate_value
        else:
            theta = candidate
            current_value = float(objective(tuple(theta)))
            evaluations += 1
            if current_value < best_value:
                best_theta = theta.copy()
                best_value = current_value

        states.append((k + 1, evaluations, theta.copy(), best_theta.copy(), current_value, best_value))
        if evaluations >= config.max_evaluations:
            break

    checkpoint_records: list[OptimizerCheckpoint] = []
    for requested_budget in requested:
        eligible = [state for state in states if state[1] <= requested_budget]
        if not eligible:
            eligible = [states[0]]
        iteration, num_evaluations, current_theta, checkpoint_best_theta, checkpoint_current_value, checkpoint_best_value = eligible[-1]
        checkpoint_records.append(
            OptimizerCheckpoint(
                requested_max_evaluations=int(requested_budget),
                num_evaluations=int(num_evaluations),
                iteration=int(iteration),
                current_theta=tuple(float(x) for x in current_theta),
                best_theta=tuple(float(x) for x in checkpoint_best_theta),
                current_value=float(checkpoint_current_value),
                best_value=float(checkpoint_best_value),
            )
        )

    return CheckpointedOptimizerResult(
        status="complete",
        initial_theta=tuple(float(x) for x in wrap_theta(initial_theta)),
        final_theta=tuple(float(x) for x in theta),
        best_theta=tuple(float(x) for x in best_theta),
        final_value=float(current_value),
        best_value=float(best_value),
        num_evaluations=int(evaluations),
        checkpoints=tuple(checkpoint_records),
        optimizer="spsa_checkpointed",
        runner_version=PHASE_E_OPTIMIZER_VERSION,
    )
