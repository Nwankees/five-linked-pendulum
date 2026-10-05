import math
import unittest

import numpy as np
import torch

from controllers import LocalLQR
from environment import FivePoleEnv, PlantConfig, TorchFivePoleBatch


torch.set_num_threads(1)


class EnvironmentTests(unittest.TestCase):
    def test_reset_starts_with_all_links_hanging(self):
        env = TorchFivePoleBatch(1000, seed=123)
        self.assertTrue((env.state[:, 2:7].cos() < -0.9).all())

    def test_mass_matrix_is_positive(self):
        env = TorchFivePoleBatch(256)
        angles = env.randn(256, 5) * 3
        eigenvalues = torch.linalg.eigvalsh(env.mass_matrix(angles))
        self.assertGreater(float(eigenvalues.min()), 0)

    def test_energy_changes_by_applied_and_friction_power(self):
        env = TorchFivePoleBatch(32)
        state = env.randn(32, 12) * 0.8
        force = env.randn(32) * 5
        direction = env.derivative(state, force)
        epsilon = 1e-6

        def total_energy(value):
            kinetic, potential, _ = env.energy(value)
            return kinetic + potential

        measured_rate = (
            total_energy(state + epsilon * direction)
            - total_energy(state - epsilon * direction)
        ) / (2 * epsilon)
        damping_loss = torch.einsum(
            "bi,ij,bj->b", state[:, 7:], env.damping_matrix, state[:, 7:]
        )
        expected_rate = (
            force * state[:, 1]
            - env.cart_friction * state[:, 1].square()
            - damping_loss
        )
        torch.testing.assert_close(measured_rate, expected_rate, atol=1e-7, rtol=1e-7)

    def test_single_and_batch_interfaces_match(self):
        single = FivePoleEnv(seed=17)
        batch = TorchFivePoleBatch(1, seed=17)
        np.testing.assert_array_equal(single.state, batch.state[0].numpy())

        for action in [0.1, -0.1, 0.3, 0.0] * 5:
            single_observation, _, _, _ = single.step(action)
            batch_observation, _, _, _ = batch.step(batch.tensor([action]))
            np.testing.assert_array_equal(
                single_observation, batch_observation[0].numpy()
            )

    def test_success_requires_two_continuous_seconds(self):
        env = TorchFivePoleBatch(1)
        env.reset(states=torch.zeros((1, 12)))

        steps_before_success = math.ceil(2 / env.dt) - 1
        for _ in range(steps_before_success):
            env.step(env.tensor([0]))
        self.assertFalse(bool(env.success_achieved[0]))

        env.step(env.tensor([0]))
        self.assertTrue(bool(env.success_achieved[0]))

    def test_local_controller_balances_nearby_states(self):
        env = TorchFivePoleBatch(32, seed=56)
        controller = LocalLQR(env)
        directions = env.randn(32, 12)
        directions /= directions.norm(dim=1, keepdim=True)
        factor = torch.linalg.cholesky(controller.P)
        states = torch.linalg.solve_triangular(
            factor.T, directions.T, upper=True
        ).T * math.sqrt(controller.rho)
        env.reset(states=states)

        for _ in range(env.max_steps):
            env.step(controller.action(env.state))

        self.assertTrue(env.success_achieved.all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
