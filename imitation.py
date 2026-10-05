import torch
from torch import nn

from controllers import LocalLQR


def state_error(state, center):
    # Find the difference between two states
    error = (state - center).clone()
    angles = error[..., 2:7]
    error[..., 2:7] = torch.remainder(angles + torch.pi, 2 * torch.pi) - torch.pi
    return error


def choose_route(state, initial_states):
    # Find the closest starting route
    error = state_error(state[:, None, :], initial_states[None, :, :])
    scales = state.new_tensor([0.1] * 7 + [0.3] * 5)
    distance = (error / scales).square().sum(dim=-1)
    return distance.argmin(dim=1)


class ExpertPolicy(nn.Module):
    label = "TRAJECTORY EXPERT"

    def __init__(self, data, device="cpu"):
        super().__init__()
        self.register_buffer("reference", data["reference"].to(device))
        self.register_buffer("inputs", data["inputs"].to(device))
        self.register_buffer("gains", data["gains"].to(device))
        self.horizon_steps = self.inputs.shape[1]

    @property
    def initials(self):
        return self.reference[:, 0]

    def raw_force(self, state, route, phase):
        phase = phase.clamp(max=self.horizon_steps - 1)
        error = state_error(state, self.reference[route, phase])
        correction = (self.gains[route, phase] * error).sum(dim=1)
        return self.inputs[route, phase] - correction


class ImitationPolicy(nn.Module):
    label = "IMITATION POLICY"

    def __init__(self, data, device="cpu"):
        super().__init__()
        self.register_buffer("centers", data["centers"].to(device))
        self.register_buffer("scales", data["scales"].to(device))
        self.register_buffer("weights", data["weights"].to(device))
        self.register_buffer("initials", data["initials"].to(device))

        self.horizon_steps = self.weights.shape[1]
        self.has_balance = "balance_weights" in data
        if self.has_balance:
            self.register_buffer("balance_weights", data["balance_weights"].to(device))
            self.register_buffer("balance_scales", data["balance_scales"].to(device))

    def raw_force(self, state, route, phase):
        balancing = phase >= self.horizon_steps
        phase = phase.clamp(max=self.horizon_steps - 1)

        error = state_error(state, self.centers[route, phase])
        normalized_error = error / self.scales[route, phase]
        bias = torch.ones_like(normalized_error[:, :1])
        features = torch.cat([bias, normalized_error], dim=1)
        force = (features * self.weights[route, phase]).sum(dim=1)

        if self.has_balance:
            force = torch.where(balancing, self.balance_force(state), force)
        return force

    def balance_force(self, state):
        normalized_state = state / self.balance_scales
        return (normalized_state * self.balance_weights).sum(dim=1)


class ImitationController:
    def __init__(self, env, policy, mode="hybrid", routes=None):
        self.env = env
        self.policy = policy
        self.kind = mode
        self.local = LocalLQR(env)

        self.mode = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.entries = torch.zeros_like(self.mode)
        self.switches = torch.zeros_like(self.mode)

        if routes is None:
            self.route = choose_route(env.state, policy.initials)
        else:
            self.route = routes.clone()

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones_like(self.mode, dtype=torch.bool)

        self.mode[mask] = 0
        self.entries[mask] = 0
        self.switches[mask] = 0
        self.route[mask] = choose_route(self.env.state[mask], self.policy.initials)

    @torch.no_grad()
    def raw_force(self):
        old_mode = self.mode.clone()
        entering_balance = self.local.eligible(self.env.state)
        was_balanced = old_mode == 2
        can_stay_balanced = self.local.eligible(self.env.state, exit=True)
        staying_balanced = was_balanced & can_stay_balanced
        use_balance = entering_balance | staying_balanced

        # The optimized trajectory ends at this point, so balancing takes over.
        use_balance |= self.env.steps >= self.policy.horizon_steps

        if self.kind == "policy":
            use_balance.zero_()
        elif self.kind == "lqr":
            use_balance.fill_(True)

        self.mode = torch.where(use_balance, 2, 0)
        self.entries += ((old_mode != 2) & use_balance).long()
        self.switches += (old_mode != self.mode).long()

        trajectory_force = self.policy.raw_force(
            self.env.state, self.route, self.env.steps
        )
        policy_has_balance = isinstance(self.policy, ImitationPolicy)
        if policy_has_balance:
            policy_has_balance = self.policy.has_balance
        if policy_has_balance and self.kind != "lqr":
            balance_force = self.policy.balance_force(self.env.state)
        else:
            balance_force = self.local.force(self.env.state)

        return torch.where(use_balance, balance_force, trajectory_force)

    def action(self):
        return (self.raw_force() / self.env.max_force).clamp(-1, 1)


@torch.no_grad()
def fit_behavioral_cloning(samples, initials, ridge=1e-7):
    # Fit a controller for each route and step
    all_centers = []
    all_scales = []
    all_weights = []
    metrics = []

    for route_samples in samples:
        states = route_samples["states"].double().cpu()
        target_forces = route_samples["forces"].double().cpu()

        center = states.mean(dim=0)
        angles = states[:, :, 2:7]
        mean_sin = angles.sin().mean(dim=0)
        mean_cos = angles.cos().mean(dim=0)
        center[:, 2:7] = torch.atan2(mean_sin, mean_cos)

        error = state_error(states, center)
        scale = error.std(dim=0, unbiased=False).clamp_min(1e-8)
        normalized_error = error / scale
        bias = torch.ones((*states.shape[:2], 1), dtype=torch.float64)
        features = torch.cat([bias, normalized_error], dim=2).transpose(0, 1)
        targets = target_forces.T.unsqueeze(-1)

        penalty = torch.eye(13, dtype=torch.float64) * ridge * len(states)
        penalty[0, 0] = 0
        normal_matrix = features.transpose(1, 2) @ features + penalty
        right_side = features.transpose(1, 2) @ targets
        weights = torch.linalg.solve(normal_matrix, right_side).squeeze(-1)

        prediction = (features * weights[:, None, :]).sum(dim=-1)
        rmse = (prediction - targets.squeeze(-1)).square().mean().sqrt()

        all_centers.append(center)
        all_scales.append(scale)
        all_weights.append(weights)
        route_metrics = {"episodes": len(states), "force_rmse_N": float(rmse)}
        metrics.append(route_metrics)

    fitted = {
        "centers": torch.stack(all_centers),
        "scales": torch.stack(all_scales),
        "weights": torch.stack(all_weights),
        "initials": initials.cpu().double(),
    }
    return fitted, metrics


@torch.no_grad()
def fit_balance_imitation(states, forces):
    # Learn the balance controller
    states = states.double().cpu()
    forces = forces.double().cpu()
    scales = states.std(dim=0, unbiased=False).clamp_min(1e-12)
    features = states / scales

    fit = torch.linalg.lstsq(features, forces, rcond=1e-13, driver="gelsd")
    if int(fit.rank) != 12:
        raise ValueError("Balance examples do not cover every state direction")

    prediction = features @ fit.solution
    rmse = (prediction - forces).square().mean().sqrt()
    result = {"balance_scales": scales, "balance_weights": fit.solution}
    return result, float(rmse)
