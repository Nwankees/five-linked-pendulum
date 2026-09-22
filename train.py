"""Behavioural cloning: predict the force chosen in expert demonstrations.

The explicit training loop mirrors a normal supervised-learning project.
No teacher is called by the saved policy at evaluation time.
"""

import argparse
import json
from pathlib import Path
import torch
from model import PolicyNetwork, features


def load_examples(path):
    demonstrations = torch.load(path, map_location='cpu', weights_only=True)
    inputs = []
    targets = []
    for route in demonstrations['samples']:
        for step in range(route['states'].shape[1]):
            states = route['states'][:, step].float()
            inputs.append(features(states, step))
            targets.append(route['forces'][:, step].float() / 45)
    return torch.cat(inputs), torch.cat(targets)


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs, targets = load_examples(args.data)
    # Split by episode: adjacent steps of one episode stay in the same split.
    # The source groups every route as [episode, time, state].
    # load_examples puts episodes together within each step, so use those
    # episode IDs when selecting the held-out fifth of each route.
    source = torch.load(args.data, map_location='cpu', weights_only=True)
    validation_masks = []
    for route in source['samples']:
        episode_ids = torch.arange(len(route['states']))
        held_out = episode_ids % 5 == 0
        for step in range(route['states'].shape[1]):
            validation_masks.append(held_out)
    validation_mask = torch.cat(validation_masks)
    validation_inputs = inputs[validation_mask].to(device)
    validation_targets = targets[validation_mask].to(device)
    inputs = inputs[~validation_mask].to(device)
    targets = targets[~validation_mask].to(device)

    network = PolicyNetwork().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    best_loss = float('inf')
    history = []
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        network.train()
        order = torch.randperm(len(inputs), device=device)
        total_loss = 0.0
        for start in range(0, len(inputs), args.batch_size):
            indices = order[start:start + args.batch_size]
            prediction = network(inputs[indices])
            loss = torch.mean((prediction - targets[indices])**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)

        network.eval()
        with torch.no_grad():
            prediction = network(validation_inputs)
            validation_loss = torch.mean((prediction - validation_targets)**2).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({'weights': network.cpu().state_dict(), 'architecture': 'mlp-18-256-256-1',
                        'seed': args.seed, 'validation_mse': best_loss}, args.output)
            network.to(device)
        row = {'epoch': epoch + 1, 'training_mse': total_loss / len(inputs),
               'validation_mse': validation_loss}
        history.append(row)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: train {row['training_mse']:.6f}, validation {validation_loss:.6f}", flush=True)

    Path(args.output).with_suffix('.json').write_text(json.dumps(history, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/demonstrations.pt')
    parser.add_argument('--output', default='models/imitation.pt')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    train(parser.parse_args())
