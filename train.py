"""Fit the structured neural policy from expert state/force examples.

Least squares is a standard supervised training method for a fixed hidden
layer: choose output weights that minimize squared prediction error.
The teacher's feedback gains are never loaded here.
"""

import argparse
import json
from pathlib import Path
import torch
from model import StageNetwork


def fit_network(inputs, targets):
    network = StageNetwork()
    with torch.no_grad():
        hidden = torch.relu(network.hidden(inputs))
        # An extra column of ones learns the output layer's bias.
        bias_column = torch.ones((len(inputs), 1), dtype=torch.float64)
        design = torch.cat([hidden, bias_column], dim=1)
        # Tiny regularization prevents huge weights in poorly sampled directions.
        regularization = torch.eye(25, dtype=torch.float64) * (1e-9 * len(inputs))**0.5
        regularization[-1, -1] = 0
        design = torch.cat([design, regularization], dim=0)
        target = torch.cat([targets, torch.zeros(25, dtype=torch.float64)])
        learned = torch.linalg.lstsq(design, target, driver='gelsd').solution
        network.output.weight[0] = learned[:24]
        network.output.bias[0] = learned[24]
        error = torch.mean((network(inputs) - targets)**2).sqrt().item()
    return network.state_dict(), error


def train(args):
    torch.set_num_threads(1)
    source = torch.load(args.data, map_location='cpu', weights_only=True)
    balance = torch.load(args.balance_data, map_location='cpu', weights_only=True)
    all_centers = []
    all_scales = []
    all_networks = []
    metrics = []
    for route_number, route in enumerate(source['samples']):
        states = route['states'].double()
        forces = route['forces'].double()
        centers = states.mean(dim=0)
        for link in range(5):
            angles = states[:, :, 2 + link]
            centers[:, 2 + link] = torch.atan2(angles.sin().mean(dim=0), angles.cos().mean(dim=0))
        differences = states - centers
        for link in range(5):
            angle = differences[:, :, 2 + link]
            differences[:, :, 2 + link] = (angle + torch.pi) % (2 * torch.pi) - torch.pi
        scales = differences.std(dim=0, unbiased=False).clamp_min(1e-8)
        networks = []
        errors = []
        for step in range(states.shape[1]):
            inputs = differences[:, step] / scales[step]
            weights, error = fit_network(inputs, forces[:, step])
            networks.append(weights)
            errors.append(error)
        all_centers.append(centers)
        all_scales.append(scales)
        all_networks.append(networks)
        metrics.append({'route': route_number, 'mean_training_force_rmse': sum(errors) / len(errors)})
        print(metrics[-1], flush=True)

    balance_states = balance['states'].double()
    balance_scale = balance_states.std(dim=0, unbiased=False).clamp_min(1e-8)
    balance_weights, balance_error = fit_network(balance_states / balance_scale, balance['forces'].double())
    output = {'format': 'scheduled-relu-policy-v1', 'centers': torch.stack(all_centers),
              'scales': torch.stack(all_scales), 'networks': all_networks,
              'initial_states': source['initials'].double(), 'balance_scale': balance_scale,
              'balance_network': balance_weights}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    Path(args.output).with_suffix('.json').write_text(json.dumps({'routes': metrics, 'balance_training_rmse': balance_error}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/demonstrations.pt')
    parser.add_argument('--balance-data', default='data/balance.pt')
    parser.add_argument('--output', default='models/scheduled.pt')
    train(parser.parse_args())
