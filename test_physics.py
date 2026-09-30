"""Run with: python -m unittest test_physics"""

import math
import unittest
from unittest.mock import patch
import numpy as np
from physics import accelerations, integrate
from environment import PendulumEnvironment


class PhysicsTests(unittest.TestCase):
    def test_equilibria(self):
        for angle in [0, math.pi]:
            state = np.zeros((1, 12))
            state[:, 2:7] = angle
            np.testing.assert_allclose(accelerations(state, np.zeros(1)), 0, atol=1e-12)

    def test_cart_force_moves_links(self):
        state = np.zeros((1, 12))
        acceleration = accelerations(state, np.array([1.0]))
        self.assertGreater(acceleration[0, 0], 0)
        self.assertLess(acceleration[0, 1], 0)

    def test_finer_integration_converges(self):
        state = np.zeros((1, 12))
        state[0, 2:7] = [0.1, -0.15, 0.2, -0.1, 0.05]
        def advance(step):
            result = state.copy()
            for index in range(round(0.02 / step)):
                result = integrate(result, np.array([0.7]), step)
            return result
        coarse = advance(0.005)
        fine = advance(0.0025)
        reference = advance(0.00125)
        self.assertLess(np.linalg.norm(fine - reference), np.linalg.norm(coarse - reference))

    def test_left_right_symmetry(self):
        state = np.array([[0.1, 0.2, 0.1, -0.2, 0.3, -0.1, 0.05, 0.4, 0.2, -0.1, 0.3, -0.2]])
        positive = integrate(state, np.array([0.8]), 0.005)
        negative = integrate(-state, np.array([-0.8]), 0.005)
        np.testing.assert_allclose(positive, -negative, atol=1e-12)

    def test_energy_conservation_without_friction(self):
        from physics import LENGTHS, MASSES, CART_MASS, GRAVITY
        def energy(state):
            horizontal_speed = state[1]
            vertical_speed = 0.0
            height = 0.0
            total = 0.5 * CART_MASS * state[1]**2
            for link in range(5):
                angle = state[2 + link]
                speed = state[7 + link]
                horizontal_speed += LENGTHS[link] * math.cos(angle) * speed
                vertical_speed += LENGTHS[link] * math.sin(angle) * speed
                height += LENGTHS[link] * math.cos(angle)
                total += 0.5 * MASSES[link] * (horizontal_speed**2 + vertical_speed**2)
                total += MASSES[link] * GRAVITY * height
            return total
        state = np.zeros((1, 12))
        state[0, 2:7] = [0.1, -0.2, 0.15, -0.1, 0.05]
        initial_energy = energy(state[0])
        with patch('physics.CART_FRICTION', 0), patch('physics.JOINT_FRICTION', 0):
            for step in range(100):
                state = integrate(state, np.array([0.0]), 0.001)
        self.assertAlmostEqual(energy(state[0]), initial_energy, places=7)

    def test_balance_requires_two_seconds(self):
        environment = PendulumEnvironment()
        environment.states[:] = 0
        for step in range(99):
            environment.step([0])
        self.assertFalse(environment.success[0])
        environment.step([0])
        self.assertTrue(environment.success[0])

    def test_invalid_physics_step(self):
        for step in [0, -0.005, 0.003]:
            with self.assertRaises(ValueError):
                PendulumEnvironment(physics_step=step)

    def test_every_link_must_balance(self):
        environment = PendulumEnvironment()
        environment.states[:] = 0
        environment.states[0, 6] = math.pi
        environment.step([0])
        self.assertEqual(environment.hold_seconds[0], 0)

    def test_track_failure_overrides_previous_success(self):
        environment = PendulumEnvironment()
        environment.success[:] = True
        environment.states[0, 0] = 2.5
        environment.step([0])
        self.assertTrue(environment.failed[0])


if __name__ == '__main__':
    unittest.main()
