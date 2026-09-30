"""Small neural networks scheduled along demonstrated swing-up trajectories.

This is a structured policy, not one universal state-to-force network.
Each stage has a 12 -> 24 ReLU -> 1 network. Its inputs are differences
from the typical demonstrated state at that stage. A final network balances.

The hidden layer splits positive and negative input components. Keeping
that layer fixed lets us fit the output layer with supervised least squares.
These networks can represent a linear correction exactly; this deliberate
restriction matches the local expert and avoids a much harder learning task.
No expert gains or expert controller are used at inference time.
"""

import numpy as np
import torch
from torch import nn
from physics import wrap_angle


class StageNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(12, 24, bias=False, dtype=torch.float64)
        self.output = nn.Linear(24, 1, dtype=torch.float64)
        with torch.no_grad():
            self.hidden.weight[:12] = torch.eye(12)
            self.hidden.weight[12:] = -torch.eye(12)
        self.hidden.weight.requires_grad_(False)

    def forward(self, inputs):
        hidden_values = torch.relu(self.hidden(inputs))
        return self.output(hidden_values).squeeze(-1)


class ScheduledController:
    def __init__(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        self.centers = saved['centers'].numpy()
        self.scales = saved['scales'].numpy()
        self.initial_states = saved['initial_states'].numpy()
        self.routes = None
        self.networks = []
        for route_weights in saved['networks']:
            route_networks = []
            for weights in route_weights:
                network = StageNetwork().eval()
                network.load_state_dict(weights)
                route_networks.append(network)
            self.networks.append(route_networks)
        self.balance_network = StageNetwork().eval()
        self.balance_network.load_state_dict(saved['balance_network'])
        self.balance_scale = saved['balance_scale'].numpy()

    def reset(self, states):
        self.routes = []
        for state in states:
            best_route = 0
            best_distance = float('inf')
            for route, initial_state in enumerate(self.initial_states):
                difference = state - initial_state
                difference[2:7] = wrap_angle(difference[2:7])
                distance = 0.0
                for variable in range(12):
                    scale = 0.1 if variable < 7 else 0.3
                    distance += (difference[variable] / scale)**2
                if distance < best_distance:
                    best_distance = distance
                    best_route = route
            self.routes.append(best_route)
        self.routes = np.array(self.routes)

    @torch.no_grad()
    def action(self, states, step):
        if self.routes is None or step == 0:
            self.reset(states)
        if step >= len(self.networks[0]):
            inputs = torch.tensor(states / self.balance_scale, dtype=torch.float64)
            return self.balance_network(inputs).numpy() / 45

        forces = np.zeros(len(states))
        # One call per route, rather than one call per cart.
        for route in range(len(self.networks)):
            selected = self.routes == route
            if not selected.any():
                continue
            difference = states[selected] - self.centers[route, step]
            for link in range(5):
                difference[:, 2 + link] = wrap_angle(difference[:, 2 + link])
            inputs = torch.tensor(difference / self.scales[route, step], dtype=torch.float64)
            forces[selected] = self.networks[route][step](inputs).numpy()
        return forces / 45
