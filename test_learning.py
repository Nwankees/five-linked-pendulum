"""Check supervised fitting, serialization and wrapped route selection."""

import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from model import StageNetwork, ScheduledController
from train import fit_network


class LearningTests(unittest.TestCase):
    def test_fit_learns_forces_from_examples(self):
        generator = torch.Generator().manual_seed(37)
        inputs = torch.randn(256, 12, dtype=torch.float64, generator=generator)
        targets = 2 * inputs[:, 0] - 3 * inputs[:, 6] + 0.7
        weights, error = fit_network(inputs, targets)
        network = StageNetwork()
        network.load_state_dict(weights)
        held_out = torch.randn(64, 12, dtype=torch.float64, generator=generator)
        expected = 2 * held_out[:, 0] - 3 * held_out[:, 6] + 0.7
        torch.testing.assert_close(network(held_out), expected, atol=1e-6, rtol=1e-6)
        self.assertLess(error, 1e-6)

    def test_saved_weights_preserve_predictions(self):
        network = StageNetwork()
        inputs = torch.randn(8, 12, dtype=torch.float64)
        expected = network(inputs).detach()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.pt'
            torch.save(network.state_dict(), path)
            restored = StageNetwork()
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(restored(inputs), expected)

    def test_route_distance_wraps_angles(self):
        controller = object.__new__(ScheduledController)
        controller.initial_states = np.zeros((2, 12))
        controller.initial_states[0, 2:7] = np.pi - 0.01
        state = np.zeros((1, 12))
        state[0, 2:7] = -np.pi + 0.01
        controller.reset(state)
        self.assertEqual(controller.routes[0], 0)


if __name__ == '__main__':
    unittest.main()
