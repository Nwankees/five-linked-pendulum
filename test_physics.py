"""Run with: python -m unittest test_physics"""

import math
import unittest
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
