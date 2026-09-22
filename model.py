"""A small, ordinary PyTorch network: inputs -> two hidden layers -> force."""

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    def __init__(self, input_size=18, hidden_size=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observation):
        return self.layers(observation).squeeze(-1)


def features(states, step_number):
    """The network sees the measured state and elapsed time, not expert forces."""
    values = [states[:, 0] / 2.4, states[:, 1] / 5]
    for link in range(5):
        angle = states[:, 2 + link]
        values.append(torch.sin(angle))
        values.append(torch.cos(angle))
        values.append(states[:, 7 + link] / 10)
    # The demonstrations are five-second swing-ups followed by balancing.
    time = torch.as_tensor(step_number, dtype=states.dtype, device=states.device)
    values.append(torch.ones_like(states[:, 0]) * time / 250)
    return torch.stack(values, dim=1)
