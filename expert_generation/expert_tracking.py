import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import casadi as ca
import numpy as np
import torch
from scipy.interpolate import CubicHermiteSpline

from controllers import LocalLQR, design_lqr
from environment import PlantConfig, TorchFivePoleBatch
from expert_generation.trajectory_expert import symbolic_plant


def discrete_model(config):
    """Return one control step and its state/input Jacobians."""
    _, dynamics = symbolic_plant(config)
    state = ca.MX.sym("state", 12)
    force = ca.MX.sym("force")
    next_state = state
    time_step = config.physics_dt

    for _ in range(config.action_repeat):
        slope_1 = dynamics(next_state, force)
        slope_2 = dynamics(next_state + time_step * slope_1 / 2, force)
        slope_3 = dynamics(next_state + time_step * slope_2 / 2, force)
        slope_4 = dynamics(next_state + time_step * slope_3, force)
        next_state += time_step * (
            slope_1 + 2 * slope_2 + 2 * slope_3 + slope_4
        ) / 6

    step = ca.Function("step", [state, force], [next_state])
    jacobian = ca.Function(
        "step_jacobian",
        [state, force],
        [ca.jacobian(next_state, state), ca.jacobian(next_state, force)],
    )
    return step, jacobian


def prepare_reference(candidate, config=None):
    """Resample an optimized path and add feedback gains along it."""
    config = config or PlantConfig()
    states = np.asarray(candidate["states"])
    forces = np.asarray(candidate["forces"])
    horizon = float(candidate["horizon"])
    control_step = config.physics_dt * config.action_repeat

    source_times = np.linspace(0, horizon, len(states))
    state_derivatives = np.column_stack(
        [
            states[:, 1],
            candidate["accelerations"][:, 0],
            states[:, 7:],
            candidate["accelerations"][:, 1:],
        ]
    )
    control_times = np.arange(round(horizon / control_step) + 1) * control_step
    reference = CubicHermiteSpline(
        source_times, states, state_derivatives
    )(control_times)
    inputs = np.interp(
        control_times[:-1] + control_step / 2, source_times, forces
    )

    _, jacobian = discrete_model(config)
    _, terminal_cost, _, _ = design_lqr(config)
    future_cost = terminal_cost * control_step
    state_limits = np.array(
        [1.2, 2.0, *([np.deg2rad(10)] * 5), *([0.5] * 5)]
    )
    state_cost = np.diag(1 / state_limits**2) * control_step
    input_cost = np.array([[control_step / config.max_force**2]])
    gains = np.zeros((len(inputs), 12))

    for step in reversed(range(len(inputs))):
        state_matrix, input_matrix = map(
            np.asarray, jacobian(reference[step], inputs[step])
        )
        force_cost = input_cost + input_matrix.T @ future_cost @ input_matrix
        gain = np.linalg.solve(
            force_cost, input_matrix.T @ future_cost @ state_matrix
        )
        gains[step] = gain
        future_cost = (
            state_cost
            + state_matrix.T @ future_cost @ state_matrix
            - state_matrix.T @ future_cost @ input_matrix @ gain
        )
        future_cost = (future_cost + future_cost.T) / 2

    return {
        "reference": reference,
        "inputs": inputs,
        "gains": gains,
        "horizon": horizon,
        "plant": asdict(config),
    }


class TrajectoryTracker:
    def __init__(self, env, reference):
        self.env = env
        self.local = LocalLQR(env)
        self.reference = env.tensor(reference["reference"])
        self.inputs = env.tensor(reference["inputs"])
        self.gains = env.tensor(reference["gains"])
        self.local_mode = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )

    def action(self, step=None):
        if step is None:
            phase = self.env.steps
        else:
            phase = torch.full_like(self.env.steps, step)
        index = phase.clamp(max=len(self.inputs) - 1)

        error = self.env.state - self.reference[index]
        error[:, 2:7] = self.env.wrap(error[:, 2:7])
        force = self.inputs[index] - (self.gains[index] * error).sum(dim=1)

        staying_local = self.local_mode & self.local.eligible(
            self.env.state, exit=True
        )
        entering_local = self.local.eligible(self.env.state)
        trajectory_finished = phase >= len(self.inputs)
        self.local_mode = staying_local | entering_local | trajectory_finished
        force = torch.where(
            self.local_mode, self.local.force(self.env.state), force
        )
        return (force / self.env.max_force).clamp(-1, 1)


@torch.no_grad()
def verify_reference(reference, config=None, initial=None, seed=0, record=False):
    """Replay a reference in the simulator and apply the real success rule."""
    config = config or PlantConfig(**reference["plant"])
    env = TorchFivePoleBatch(1, seed=seed, config=config)
    start = reference["reference"][0] if initial is None else initial
    env.reset(states=np.asarray(start)[None])
    tracker = TrajectoryTracker(env, reference)

    recorded_states = [env.state[0].numpy().copy()]
    recorded_actions = []
    recorded_times = []
    max_tracking_error = 0.0
    max_force = 0.0
    saturated_steps = 0

    for step in range(env.max_steps):
        action = tracker.action()
        recorded_actions.append(float(action[0]))
        recorded_times.append(step * env.dt)
        _, _, done, info = env.step(action)
        recorded_states.append(env.state[0].numpy().copy())

        max_force = max(max_force, float(abs(action[0]) * env.max_force))
        saturated_steps += int(abs(float(action[0])) > 0.999)
        if step + 1 < len(reference["reference"]):
            error = env.state[0].numpy() - reference["reference"][step + 1]
            error[2:7] = (error[2:7] + np.pi) % (2 * np.pi) - np.pi
            max_tracking_error = max(
                max_tracking_error, float(np.max(abs(error)))
            )
        if done[0]:
            break

    elapsed = float(env.elapsed_ticks[0] * env.physics_dt)
    result = {
        "success": bool(env.success_achieved[0]),
        "hold_s": float(env.max_hold_ticks[0] * env.physics_dt),
        "failed_track": bool(info["failed_track"][0]),
        "failed_numeric": bool(info["failed_numeric"][0]),
        "elapsed_s": elapsed,
        "max_force": max_force,
        "max_tracking_error": max_tracking_error,
        "saturated_fraction": saturated_steps / len(recorded_actions),
        "terminal_state": env.state[0].numpy().tolist(),
    }
    result["accepted"] = (
        result["success"]
        and not result["failed_track"]
        and not result["failed_numeric"]
        and elapsed >= config.duration - 1e-8
    )
    if record:
        result.update(
            states=np.asarray(recorded_states),
            actions=np.asarray(recorded_actions),
            times=np.asarray(recorded_times),
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--output", default="experts/reference.npz")
    args = parser.parse_args()
    torch.set_num_threads(1)
    start_time = time.time()

    candidate_path = Path(args.candidate)
    metadata = json.loads(candidate_path.with_suffix(".json").read_text())
    invalid_candidate = (
        not metadata.get("solver_success")
        or metadata.get("max_assistance", 1) != 0
        or metadata.get("violation", 1) > 1e-5
    )
    if invalid_candidate:
        raise ValueError("Candidate is not an unassisted feasible solution")

    config = PlantConfig(**metadata["plant"])
    candidate = dict(np.load(candidate_path))
    reference = prepare_reference(candidate, config)
    result = verify_reference(
        reference, config, initial=metadata.get("initial_state"), record=True
    )

    output = Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        output,
        **{name: value for name, value in reference.items() if name != "plant"},
    )
    np.savez_compressed(
        output.with_name(output.stem + "_replay.npz"),
        states=result.pop("states"),
        actions=result.pop("actions"),
        times=result.pop("times"),
    )
    result["wall_s"] = time.time() - start_time
    result["plant"] = reference["plant"]
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
