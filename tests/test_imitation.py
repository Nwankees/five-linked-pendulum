import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from controllers import LocalLQR, load_bundle, make_controller
from environment import PlantConfig, TorchFivePoleBatch
from expert_generation.expert_tracking import discrete_model, verify_reference
from expert_generation.trajectory_expert import symbolic_plant
from imitation import (
    ImitationPolicy,
    fit_balance_imitation,
    fit_behavioral_cloning,
    state_error,
)


torch.set_num_threads(1)


class ImitationTests(unittest.TestCase):
    def test_symbolic_model_matches_pytorch_model(self):
        env = TorchFivePoleBatch(1)
        _, symbolic_dynamics = symbolic_plant()
        symbolic_step, symbolic_jacobian = discrete_model(env.config)

        for _ in range(8):
            state = env.randn(1, 12)
            force = env.randn(1) * 10

            expected_derivative = env.derivative(state, force)[0].numpy()
            symbolic_derivative = np.asarray(
                symbolic_dynamics(state[0].numpy(), float(force[0]))
            ).ravel()
            np.testing.assert_allclose(
                symbolic_derivative, expected_derivative, atol=1e-9, rtol=1e-9
            )

            expected_next_state = state.clone()
            for _ in range(env.action_repeat):
                expected_next_state = env.integrate(expected_next_state, force)
            actual_next_state = np.asarray(
                symbolic_step(state[0].numpy(), float(force[0]))
            ).ravel()
            np.testing.assert_allclose(
                actual_next_state,
                expected_next_state[0].numpy(),
                atol=1e-9,
                rtol=1e-9,
            )

        state_vector = state[0].numpy()
        force_value = float(force[0])
        symbolic_a, _ = map(
            np.asarray, symbolic_jacobian(state_vector, force_value)
        )
        epsilon = 1e-6
        finite_difference_columns = []
        for direction in np.eye(12):
            positive = np.asarray(
                symbolic_step(state_vector + epsilon * direction, force_value)
            )
            negative = np.asarray(
                symbolic_step(state_vector - epsilon * direction, force_value)
            )
            finite_difference_columns.append(
                ((positive - negative) / (2 * epsilon)).ravel()
            )
        finite_difference = np.column_stack(finite_difference_columns)
        np.testing.assert_allclose(
            symbolic_a, finite_difference, atol=1e-6, rtol=1e-6
        )

    def test_behavioral_cloning_learns_force_labels(self):
        generator = torch.Generator().manual_seed(81)
        states = torch.randn(
            (128, 9, 12), generator=generator, dtype=torch.float64
        ) * 0.02
        gains = torch.randn((9, 12), generator=generator, dtype=torch.float64) * 100
        bias = torch.randn(9, generator=generator, dtype=torch.float64)
        forces = (states * gains).sum(dim=-1) + bias

        fitted, metrics = fit_behavioral_cloning(
            [{"states": states, "forces": forces}],
            torch.zeros((1, 12)),
            ridge=1e-9,
        )
        model = ImitationPolicy(fitted)

        held_out = torch.randn((9, 12), generator=generator, dtype=torch.float64) * 0.02
        routes = torch.zeros(9, dtype=torch.long)
        predicted = model.raw_force(held_out, routes, torch.arange(9))
        expected = (held_out * gains).sum(dim=-1) + bias
        torch.testing.assert_close(predicted, expected, atol=1e-6, rtol=1e-6)
        self.assertLess(metrics[0]["force_rmse_N"], 1e-6)

    def test_angle_error_wraps_across_pi(self):
        first = torch.zeros((1, 12), dtype=torch.float64)
        second = first.clone()
        first[:, 2:7] = np.pi - 0.01
        second[:, 2:7] = -np.pi + 0.01
        expected = torch.full((1, 5), -0.02, dtype=torch.float64)
        torch.testing.assert_close(state_error(first, second)[:, 2:7], expected)

    def test_checkpoint_loads_and_returns_safe_actions(self):
        states = torch.randn(32, 5, 12, dtype=torch.float64) * 0.01
        samples = [{"states": states, "forces": torch.zeros(32, 5)}]
        fitted, _ = fit_behavioral_cloning(samples, torch.zeros((1, 12)))

        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "student.pt"
            torch.save(
                {
                    "format": "imitation-v1",
                    "policy": fitted,
                    "plant": asdict(PlantConfig()),
                },
                path,
            )
            model, _, config, _ = load_bundle(path)
            env = TorchFivePoleBatch(2, config=config)
            action = make_controller(env, model).action()

        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue((action.abs() <= 1).all())

    def test_balance_controller_is_learned_from_examples(self):
        env = TorchFivePoleBatch(1)
        local = LocalLQR(env)
        states = env.randn(256, 12) * 1e-4
        forces = local.force(states)
        balance_data, rmse = fit_balance_imitation(states, forces)

        policy_data = {
            "centers": torch.zeros(1, 3, 12, dtype=torch.float64),
            "scales": torch.ones(1, 3, 12, dtype=torch.float64),
            "weights": torch.zeros(1, 3, 13, dtype=torch.float64),
            "initials": states[:1],
            **balance_data,
        }
        model = ImitationPolicy(policy_data)
        held_out = env.randn(64, 12) * 1e-4
        torch.testing.assert_close(
            model.balance_force(held_out),
            local.force(held_out),
            atol=1e-8,
            rtol=1e-8,
        )
        self.assertLess(rmse, 1e-8)

    def test_failed_replay_is_rejected(self):
        states = np.zeros((11, 12))
        states[:, 2:7] = np.pi
        reference = {
            "reference": states,
            "inputs": np.zeros(10),
            "gains": np.zeros((10, 12)),
            "plant": asdict(PlantConfig()),
        }
        result = verify_reference(reference)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
