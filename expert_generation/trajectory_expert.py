"""Find swing-up trajectories with CasADi and IPOPT."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import casadi as ca
from environment import PlantConfig, TorchFivePoleBatch


def symbolic_plant(config=None):
    """Make a symbolic copy of the PyTorch plant equations."""
    env = TorchFivePoleBatch(1, config=config)
    state = ca.SX.sym("state", 12)
    acceleration = ca.SX.sym("acceleration", 6)
    force = ca.SX.sym("force")

    angles = state[2:7]
    angular_velocity = state[7:]
    energy_weights = ca.DM(env.energy_weights.numpy())
    angular_coefficients = ca.DM(env.angular_coeff.numpy())

    mass_matrix = ca.SX.zeros(6, 6)
    mass_matrix[0, 0] = env.cart_mass + float(env.pole_masses.sum())
    cart_coupling = energy_weights * ca.cos(angles)
    mass_matrix[0, 1:] = ca.transpose(cart_coupling)
    mass_matrix[1:, 0] = cart_coupling

    angle_difference = ca.repmat(angles, 1, 5) - ca.repmat(angles.T, 5, 1)
    mass_matrix[1:, 1:] = angular_coefficients * ca.cos(angle_difference)

    cart_force = (
        force
        - env.cart_friction * state[1]
        + ca.sum1(energy_weights * ca.sin(angles) * angular_velocity**2)
    )
    coriolis = ca.sum2(
        angular_coefficients
        * ca.sin(angle_difference)
        * ca.repmat((angular_velocity**2).T, 5, 1)
    )
    joint_force = (
        env.gravity * energy_weights * ca.sin(angles)
        - coriolis
        - ca.DM(env.damping_matrix.numpy()) @ angular_velocity
    )
    right_side = ca.vertcat(cart_force, joint_force)

    residual = ca.Function(
        "mechanics", [state, acceleration, force], [mass_matrix @ acceleration - right_side]
    )
    solved_acceleration = ca.solve(mass_matrix, right_side)
    derivative = ca.vertcat(
        state[1], solved_acceleration[0], angular_velocity, solved_acceleration[1:]
    )
    dynamics = ca.Function("dynamics", [state, force], [derivative])
    return residual, dynamics


def solve_trajectory(initial, horizon=6., intervals=120, seed=0, guess=None,
                     assistance=0., max_iterations=1500, config=None, verbose=False):
    config = config or PlantConfig()
    control_step = config.physics_dt * config.action_repeat
    off_grid = abs(horizon / control_step - round(horizon / control_step)) > 1e-8
    if intervals < 12 or horizon <= 0 or horizon > config.duration - 2 or off_grid:
        raise ValueError(
            "Use at least 12 intervals and leave two seconds for balancing"
        )
    if np.shape(initial) != (12,) or not np.isfinite(initial).all():
        raise ValueError("Initial state must be a finite 12-vector")

    residual, dynamics = symbolic_plant(config)
    optimizer = ca.Opti()
    interval_count = intervals
    interval_time = horizon / interval_count
    states = optimizer.variable(12, interval_count + 1)
    accelerations = optimizer.variable(6, interval_count + 1)
    midpoint_accelerations = optimizer.variable(6, interval_count)
    normalized_forces = optimizer.variable(1, interval_count + 1)

    scaled_accelerations = accelerations * 30
    scaled_midpoint_accelerations = midpoint_accelerations * 30
    forces = normalized_forces * config.max_force
    assistance_forces = None
    if assistance > 0:
        assistance_forces = optimizer.variable(5, 2 * interval_count + 1)

    def state_derivative(state, acceleration):
        return ca.vertcat(state[1], acceleration[0], state[7:], acceleration[1:])

    derivatives = ca.horzcat(
        *[
            state_derivative(states[:, step], scaled_accelerations[:, step])
            for step in range(interval_count + 1)
        ]
    )
    for step in range(interval_count + 1):
        error = residual(
            states[:, step], scaled_accelerations[:, step], forces[step]
        )
        optimizer.subject_to(error[:1] == 0)
        joint_assistance = (
            assistance_forces[:, step] if assistance_forces is not None else 0
        )
        optimizer.subject_to(error[1:] == joint_assistance)

    for step in range(interval_count):
        midpoint = (states[:, step] + states[:, step + 1]) / 2
        midpoint += interval_time * (
            derivatives[:, step] - derivatives[:, step + 1]
        ) / 8
        midpoint_derivative = state_derivative(
            midpoint, scaled_midpoint_accelerations[:, step]
        )
        integrated_change = interval_time * (
            derivatives[:, step]
            + 4 * midpoint_derivative
            + derivatives[:, step + 1]
        ) / 6
        optimizer.subject_to(states[:, step + 1] - states[:, step] == integrated_change)

        midpoint_force = (forces[step] + forces[step + 1]) / 2
        error = residual(
            midpoint, scaled_midpoint_accelerations[:, step], midpoint_force
        )
        optimizer.subject_to(error[:1] == 0)
        joint_assistance = (
            assistance_forces[:, interval_count + 1 + step]
            if assistance_forces is not None
            else 0
        )
        optimizer.subject_to(error[1:] == joint_assistance)
        optimizer.subject_to(
            optimizer.bounded(
                -config.track_limit + 0.05,
                midpoint[0],
                config.track_limit - 0.05,
            )
        )
    initial = np.asarray(initial, dtype=float).copy()
    initial[2:7] = np.pi + (initial[2:7] - np.pi + np.pi) % (2 * np.pi) - np.pi
    optimizer.subject_to(states[:, 0] == initial)
    optimizer.subject_to(states[:, -1] == 0)
    optimizer.subject_to(optimizer.bounded(-1, normalized_forces, 1))
    optimizer.subject_to(
        optimizer.bounded(
            -config.track_limit + 0.05,
            states[0, :],
            config.track_limit - 0.05,
        )
    )
    optimizer.subject_to(optimizer.bounded(-8, states[1, :], 8))
    optimizer.subject_to(optimizer.bounded(-40, states[7:, :], 40))
    optimizer.subject_to(optimizer.bounded(-6 * np.pi, states[2:7, :], 6 * np.pi))

    cost = interval_time * (
        0.01 * ca.sumsqr(normalized_forces)
        + 0.002 * ca.sumsqr(states[0, :])
        + 0.0001 * ca.sumsqr(states[7:, :])
    )
    cost += 0.002 * ca.sumsqr(normalized_forces[:, 1:] - normalized_forces[:, :-1])
    if assistance_forces is not None:
        cost += interval_time * assistance * ca.sumsqr(assistance_forces)
    optimizer.minimize(cost)
    if guess is None:
        phase = np.linspace(0, 1, interval_count + 1)
        blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
        initial_guess = initial[:, None] * (1 - blend)
        random = np.random.default_rng(seed)
        initial_guess[0] = 0.7 * np.sin(2 * np.pi * phase) * np.sin(np.pi * phase)
        initial_guess[2:7] += random.normal(0, 0.25, (5, 1)) * np.sin(
            2 * np.pi * phase
        )
        initial_guess[1] = np.gradient(initial_guess[0], interval_time)
        initial_guess[7:] = np.gradient(
            initial_guess[2:7], interval_time, axis=1
        )
        initial_guess[:, 0] = initial
        initial_guess[:, -1] = 0
        optimizer.set_initial(states, initial_guess)
        acceleration_guess = np.vstack(
            [
                np.gradient(initial_guess[1], interval_time),
                np.gradient(initial_guess[7:], interval_time, axis=1),
            ]
        )
        optimizer.set_initial(accelerations, acceleration_guess / 30)
    else:
        old_times = np.linspace(0, horizon, len(guess["states"]))
        new_times = np.linspace(0, horizon, interval_count + 1)
        initial_guess = np.array(
            [
                np.interp(new_times, old_times, guess["states"][:, index])
                for index in range(12)
            ]
        )
        force_guess = np.interp(
            new_times,
            np.linspace(0, horizon, len(guess["forces"])),
            guess["forces"],
        )
        optimizer.set_initial(states, initial_guess)
        optimizer.set_initial(normalized_forces, force_guess / config.max_force)
        acceleration_indices = [1, 7, 8, 9, 10, 11]
        acceleration_guess = np.array(
            [
                np.asarray(dynamics(initial_guess[:, step], force_guess[step])).ravel()[
                    acceleration_indices
                ]
                for step in range(interval_count + 1)
            ]
        ).T
        optimizer.set_initial(accelerations, acceleration_guess / 30)
        optimizer.set_initial(
            midpoint_accelerations,
            (acceleration_guess[:, :-1] + acceleration_guess[:, 1:]) / 60,
        )

    optimizer.solver(
        "ipopt",
        {"expand": True, "print_time": False},
        {
            "max_iter": max_iterations,
            "tol": 1e-7,
            "constr_viol_tol": 1e-7,
            "print_level": 5 if verbose else 0,
            "sb": "yes",
            "mu_strategy": "adaptive",
        },
    )
    try:
        solution = optimizer.solve()
        success = True
    except RuntimeError:
        solution = optimizer.debug
        success = False
    stats = optimizer.stats()

    try:
        result_states = np.asarray(solution.value(states)).T
        result_forces = np.asarray(solution.value(forces)).ravel()
        result_accelerations = np.asarray(solution.value(scaled_accelerations)).T
        upper_violation = np.maximum(
            np.asarray(solution.value(optimizer.g - optimizer.ubg)), 0
        )
        lower_violation = np.maximum(
            np.asarray(solution.value(optimizer.lbg - optimizer.g)), 0
        )
        violation = float(max(upper_violation.max(), lower_violation.max()))
        max_assistance = (
            float(abs(solution.value(assistance_forces)).max())
            if assistance_forces is not None
            else 0.0
        )
    except RuntimeError:
        result_states = np.zeros((interval_count + 1, 12))
        result_forces = np.zeros(interval_count + 1)
        result_accelerations = np.zeros((interval_count + 1, 6))
        violation = float("inf")
        max_assistance = float("inf")

    return {
        "states": result_states,
        "forces": result_forces,
        "accelerations": result_accelerations,
        "horizon": horizon,
        "initial_state": initial.tolist(),
        "solver_success": success,
        "status": stats["return_status"],
        "iterations": stats["iter_count"],
        "violation": violation,
        "max_assistance": max_assistance,
        "plant": asdict(config),
    }


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=float, default=6.0)
    parser.add_argument("--intervals", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--assistance", type=float, default=0.0)
    parser.add_argument("--guess")
    parser.add_argument("--output", default="experts/candidate.npz")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)

    initial_state = np.zeros(12)
    initial_state[2:7] = np.pi
    guess = dict(np.load(args.guess)) if args.guess else None
    start_time = time.time()
    result = solve_trajectory(
        initial_state,
        args.horizon,
        args.intervals,
        args.seed,
        guess,
        args.assistance,
        args.iterations,
        verbose=args.verbose,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        states=result["states"],
        forces=result["forces"],
        accelerations=result["accelerations"],
        horizon=result["horizon"],
    )
    metadata = {
        name: value
        for name, value in result.items()
        if name not in {"states", "forces", "accelerations"}
    }
    metadata["wall_s"] = time.time() - start_time
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
