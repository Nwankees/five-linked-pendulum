from functools import lru_cache

import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from environment import PlantConfig, TorchFivePoleBatch


@lru_cache(maxsize=8)
def design_lqr(config):
    # Stabilize the poles when they are near upright
    env = TorchFivePoleBatch(1, config=config)
    epsilon = 1e-5

    def advance(states, forces):
        for _ in range(config.action_repeat):
            states = env.integrate(states, forces)
        return states

    state_changes = torch.eye(12, dtype=torch.float64) * epsilon
    zero_forces = torch.zeros(12, dtype=torch.float64)
    upright = torch.zeros((1, 12), dtype=torch.float64)

    state_matrix = (advance(state_changes, zero_forces) - advance(-state_changes, zero_forces)).numpy().T / (2 * epsilon)

    positive = advance(upright, env.tensor([epsilon]))
    negative = advance(upright, env.tensor([-epsilon]))
    input_matrix = ((positive - negative) / (2 * epsilon)).numpy().T

    state_limits = np.array([1.2, 2.0, *([np.deg2rad(10)] * 5), *([0.5] * 5)])
    state_cost = np.diag(1 / state_limits**2)
    input_cost = np.array([[1 / config.max_force**2]])

    cost_to_go = solve_discrete_are(state_matrix, input_matrix, state_cost, input_cost)
    cost_to_go = (cost_to_go + cost_to_go.T) / 2
    gain = np.linalg.solve(
        input_cost + input_matrix.T @ cost_to_go @ input_matrix,
        input_matrix.T @ cost_to_go @ state_matrix,
    )

    inverse_cost = np.linalg.inv(cost_to_go)
    capture_limits = np.array([1.2, 2.0, *([0.15] * 5), *([1.0] * 5)])
    state_radius = np.min(capture_limits**2 / np.diag(inverse_cost))
    force_radius = (0.8 * config.max_force) ** 2 / (
        gain @ inverse_cost @ gain.T
    ).item()
    capture_radius = 0.2 * min(state_radius, force_radius)

    continuous_state = (
        env.derivative(state_changes, zero_forces)
        - env.derivative(-state_changes, zero_forces)
    ).numpy().T / (2 * epsilon)
    continuous_input = (
        env.derivative(upright, env.tensor([epsilon]))
        - env.derivative(upright, env.tensor([-epsilon]))
    ).numpy().T / (2 * epsilon)
    open_loop_rates = np.linalg.eigvals(continuous_state)
    controllability = min(
        np.linalg.svd(
            np.c_[rate * np.eye(12) - continuous_state, continuous_input],
            compute_uv=False,
        )[-1]
        for rate in open_loop_rates
    )

    report = {
        "open_loop_rates": sorted(open_loop_rates.real.tolist()),
        "pbh_min_singular": float(controllability),
        "closed_loop_radius": float(
            max(abs(np.linalg.eigvals(state_matrix - input_matrix @ gain)))
        ),
        "max_gain": float(abs(gain).max()),
        "rho": float(capture_radius),
        "p_condition": float(np.linalg.cond(cost_to_go)),
    }
    return gain, cost_to_go, capture_radius, report


class LocalLQR:
    def __init__(self, env):
        gain, cost, self.rho, self.report = design_lqr(env.config)
        self.gain = env.tensor(gain[0])
        self.cost_matrix = env.tensor(cost)
        self.max_force = env.max_force

        # Keep these names for the training code
        self.K = self.gain
        self.P = self.cost_matrix

    def cost(self, state):
        value = torch.einsum("bi,ij,bj->b", state, self.cost_matrix, state).clamp_min(0)
        return value / self.rho

    def force(self, state):
        return -(state * self.gain).sum(dim=1)

    def eligible(self, state, exit=False):
        cost_limit = 4.0 if exit else 1.0
        angles_are_close = state[:, 2:7].abs().max(dim=1).values < 0.3
        cart_is_safe = state[:, 0].abs() < 1.8
        return (self.cost(state) <= cost_limit) & angles_are_close & cart_is_safe

    def action(self, state):
        return (self.force(state) / self.max_force).clamp(-1, 1)


def load_bundle(path, device="cpu", config_override=None):
    # Load an expert or imitation model
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    checkpoint_type = checkpoint.get("format")

    if checkpoint_type not in {"expert-library-v1", "imitation-v1"}:
        raise ValueError("This project only supports expert-library-v1 and imitation-v1 files")

    config = PlantConfig(**checkpoint["plant"])
    if config_override is not None and config_override != config:
        raise ValueError("The checkpoint must use its saved plant configuration")

    from imitation import ExpertPolicy, ImitationPolicy

    if checkpoint_type == "expert-library-v1":
        policy = ExpertPolicy(checkpoint, device)
    else:
        policy = ImitationPolicy(checkpoint["policy"], device)

    return policy.eval(), None, config, checkpoint


def make_controller(env, policy, catch=None, mode="auto"):
    # Set up the controller
    del catch
    from imitation import ImitationController, ImitationPolicy

    if mode == "auto":
        mode = "policy" if isinstance(policy, ImitationPolicy) else "hybrid"
    return ImitationController(env, policy, mode)
