"""Evaluate actual rollouts, including every physics substep's track check."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from environment import PendulumEnvironment
from model import PolicyNetwork, features


class NeuralController:
    def __init__(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        self.network = PolicyNetwork().eval()
        self.network.load_state_dict(saved['weights'])

    @torch.no_grad()
    def action(self, states, step):
        inputs = features(torch.tensor(states, dtype=torch.float32), step)
        return self.network(inputs).numpy()


def evaluate(model, episodes=512, seed=91000, physics_step=0.005):
    torch.set_num_threads(1)
    environment = PendulumEnvironment(episodes, seed, physics_step)
    controller = NeuralController(model)
    while not environment.finished.all():
        action = controller.action(environment.states, environment.step_number)
        environment.step(action)
    passed = environment.success & ~environment.failed
    trials = []
    for episode in range(episodes):
        trials.append({'episode': episode, 'success': bool(passed[episode]),
                       'failed': bool(environment.failed[episode]),
                       'best_hold_seconds': float(environment.best_hold[episode]),
                       'maximum_cart_position': float(environment.maximum_position[episode])})
    return {'model': str(model), 'episodes': episodes, 'successes': int(passed.sum()),
            'seed': seed, 'physics_step': physics_step, 'control_step': 0.02,
            'definition': 'All links within 10 degrees and 0.5 rad/s for 2 seconds; no track exit in 15 seconds.',
            'trials': trials}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/imitation.pt')
    parser.add_argument('--episodes', type=int, default=512)
    parser.add_argument('--seed', type=int, default=91000)
    parser.add_argument('--physics-step', type=float, default=0.005)
    parser.add_argument('--output', default='results/evaluation.json')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('Episodes must be positive.')
    report = evaluate(args.model, args.episodes, args.seed, args.physics_step)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"Success: {report['successes']}/{args.episodes}")
