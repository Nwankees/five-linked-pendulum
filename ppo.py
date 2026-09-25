"""A small PPO experiment, with the collection and update loops visible.

PPO compares the probability of an action before and after an update.
Clipping that ratio prevents one update from changing the policy too far.
This experiment is kept separate from supervised imitation learning.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from environment import PendulumEnvironment


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(17, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(17, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.log_standard_deviation = nn.Parameter(torch.tensor(-0.5))

    def distribution(self, observations):
        mean = self.actor(observations).squeeze(-1)
        return Normal(mean, self.log_standard_deviation.exp())

    def value(self, observations):
        return self.critic(observations).squeeze(-1)


def experiment(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    environment = PendulumEnvironment(count=32, seed=args.seed)
    network = ActorCritic()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0003)
    history = []
    episodes = 0
    successes = 0
    transitions = 0

    for update in range(args.updates):
        observations = []
        actions = []
        log_probabilities = []
        rewards = []
        done_flags = []
        values = []

        # Start independent truncated rollouts. The last state is bootstrapped.
        environment.reset()
        for step in range(128):
            observation = torch.tensor(environment.observation(), dtype=torch.float32)
            with torch.no_grad():
                distribution = network.distribution(observation)
                action = distribution.sample()
                value = network.value(observation)
            already_finished = environment.finished.copy()
            _, reward, done = environment.step(action.numpy())
            valid = ~already_finished
            observations.append(observation)
            actions.append(action)
            log_probabilities.append(distribution.log_prob(action))
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            done_flags.append(torch.tensor(done))
            values.append(value)
            # Completed carts contribute no later loss in this rollout.
            if step == 0:
                active_masks = []
            active_masks.append(torch.tensor(valid))
            newly_done = done & ~already_finished
            episodes += int(newly_done.sum())
            successes += int((newly_done & environment.success & ~environment.failed).sum())
            transitions += int(valid.sum())

        with torch.no_grad():
            final_observation = torch.tensor(environment.observation(), dtype=torch.float32)
            next_value = network.value(final_observation)
        advantages = torch.zeros((128, environment.count))
        next_advantage = torch.zeros(environment.count)
        for step in reversed(range(128)):
            continuing = (~done_flags[step]).float()
            prediction_error = rewards[step] + 0.99 * next_value * continuing - values[step]
            advantages[step] = prediction_error + 0.99 * 0.95 * continuing * next_advantage
            next_advantage = advantages[step]
            next_value = values[step]

        mask = torch.stack(active_masks).flatten()
        observations = torch.stack(observations).reshape(-1, 17)[mask]
        actions = torch.stack(actions).flatten()[mask]
        old_log_probability = torch.stack(log_probabilities).flatten()[mask].detach()
        returns = (advantages + torch.stack(values)).flatten()[mask]
        advantages = advantages.flatten()[mask]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(4):
            order = torch.randperm(len(observations))
            for start in range(0, len(order), 256):
                indices = order[start:start + 256]
                distribution = network.distribution(observations[indices])
                new_log_probability = distribution.log_prob(actions[indices])
                ratio = torch.exp(new_log_probability - old_log_probability[indices])
                ordinary_objective = ratio * advantages[indices]
                clipped_objective = torch.clamp(ratio, 0.8, 1.2) * advantages[indices]
                policy_loss = -torch.minimum(ordinary_objective, clipped_objective).mean()
                value_loss = ((network.value(observations[indices]) - returns[indices])**2).mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * distribution.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                optimizer.step()

        row = {'update': update + 1, 'transitions': transitions, 'completed_episodes': episodes,
               'successes': successes, 'mean_rollout_reward': float(torch.stack(rewards).mean())}
        history.append(row)
        print(row, flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'weights': network.state_dict(), 'seed': args.seed}, args.output)
    Path(args.output).with_suffix('.json').write_text(json.dumps(history, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='models/ppo_experiment.pt')
    experiment(parser.parse_args())
