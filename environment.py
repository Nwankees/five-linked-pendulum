# Five pole physics environment
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch


@dataclass(frozen=True)
class PlantConfig:
    physics_dt: float = 0.005
    action_repeat: int = 4
    duration: float = 15.0
    max_force: float = 45.0
    track_limit: float = 2.4
    cart_mass: float = 1.5
    gravity: float = 9.81
    cart_friction: float = 0.10
    joint_damping: float = 0.003
    integrator: str = 'rk4'
    damping: str = 'relative'

    def __post_init__(self):
        if min(self.physics_dt, self.duration, self.max_force, self.track_limit) <= 0 or self.action_repeat < 1:
            raise ValueError('Time, force and track parameters must be positive')
        if self.integrator not in ('rk4', 'euler') or self.damping not in ('relative', 'absolute'):
            raise ValueError('Unknown integration or damping model')


class TorchFivePoleBatch:
    def __init__(self, num_envs, device='cpu', seed=0, config=None):
        if config is None:
            self.config = PlantConfig()
        else:
            self.config = config
        for key, value in asdict(self.config).items():
            setattr(self, key, value)
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = torch.float64  # Large local gains need precise state feedback.
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.num_poles = 5
        self.observation_size = 17
        self.action_size = 1
        self.dt = self.physics_dt*self.action_repeat
        self.max_steps = math.ceil(self.duration/self.dt)
        self.balance_angle = math.radians(10)
        self.balance_velocity = 0.5
        self.success_ticks = math.ceil(2.0/self.physics_dt - 1e-10)
        self.pole_masses = self.tensor([.24, .22, .20, .18, .16])
        self.pole_lengths = self.tensor([.48, .44, .40, .36, .32])
        self.mass_above = self.pole_masses.flip(0).cumsum(0).flip(0)
        idx = torch.arange(5, device=self.device)
        self.angular_coeff = (self.mass_above[torch.maximum(idx[:, None], idx[None, :])]
                              * self.pole_lengths[:, None]*self.pole_lengths[None, :])
        self.energy_weights = self.mass_above*self.pole_lengths
        self.target_energy = self.gravity*self.energy_weights.sum()
        # Joint damping
        D = torch.eye(5, dtype=self.dtype, device=self.device)
        D[idx[1:], idx[:-1]] = -1
        if self.damping == 'relative':
            self.damping_matrix = self.joint_damping * (D.T @ D)
        else:
            self.damping_matrix = self.joint_damping * torch.eye(5, device=self.device, dtype=self.dtype)
        self.state = self.tensor(np.zeros((num_envs, 12)))
        self.steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.hold_ticks = torch.zeros_like(self.steps)
        self.max_hold_ticks = torch.zeros_like(self.steps)
        self.elapsed_ticks = torch.zeros_like(self.steps)
        self.success_achieved = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.finished = self.success_achieved.clone()
        self.episode_returns = self.tensor(np.zeros(num_envs))
        self.reset()

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def randn(self, *shape):
        return torch.randn(*shape, device=self.device, dtype=self.dtype, generator=self.generator)

    @staticmethod
    def wrap(angles):
        return torch.remainder(angles+math.pi, 2*math.pi)-math.pi

    def reset(self, mask=None, states=None):
        if mask is None:
            mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        count = int(mask.sum())
        if states is None:
            relative = self.randn(count, 5)*.035
            relative[:, 0] = math.pi+.01*self.randn(count)
            position = .01 * self.randn(count, 1)
            velocity = .02 * self.randn(count, 1)
            angles = relative.cumsum(1)
            angular_velocity = .03 * self.randn(count, 5)
            states = torch.cat([position, velocity, angles, angular_velocity], 1)
        states = self.tensor(states).clone()
        states[:, 2:7] = self.wrap(states[:, 2:7])
        self.state[mask] = states
        for value in (self.steps, self.hold_ticks, self.max_hold_ticks, self.elapsed_ticks,
                      self.success_achieved, self.finished, self.episode_returns):
            value[mask] = 0
        return self.observation()

    def observation(self):
        return self.observe(self.state)

    def observe(self, state):
        poles = torch.stack([state[:, 2:7].sin(), state[:, 2:7].cos(), state[:, 7:]/10], 2)
        return torch.cat([state[:, :1]/self.track_limit, state[:, 1:2]/5, poles.flatten(1)], 1).float()

    def mass_matrix(self, angles):
        M = torch.zeros((len(angles), 6, 6), device=self.device, dtype=self.dtype)
        M[:, 0, 0] = self.cart_mass+self.pole_masses.sum()
        M[:, 0, 1:] = self.energy_weights*angles.cos()
        M[:, 1:, 0] = M[:, 0, 1:]
        M[:, 1:, 1:] = self.angular_coeff*(angles[:, :, None]-angles[:, None, :]).cos()
        return M

    def accelerations(self, force, state=None):
        if state is None:
            s = self.state
        else:
            s = state
        angles = s[:, 2:7]
        w = s[:, 7:]
        rhs = torch.zeros((len(s), 6), device=self.device, dtype=self.dtype)
        rhs[:, 0] = force-self.cart_friction*s[:, 1]+(self.energy_weights*angles.sin()*w.square()).sum(1)
        coriolis = (self.angular_coeff*(angles[:, :, None]-angles[:, None, :]).sin()*w[:, None, :].square()).sum(2)
        rhs[:, 1:] = self.gravity*self.energy_weights*angles.sin()-coriolis-w@self.damping_matrix.T
        return torch.linalg.solve(self.mass_matrix(angles), rhs.unsqueeze(-1)).squeeze(-1)

    def derivative(self, state, force):
        a = self.accelerations(force, state)
        return torch.cat([state[:, 1:2], a[:, :1], state[:, 7:], a[:, 1:]], 1)

    def integrate(self, state, force, dt=None):
        if dt is None:
            h = self.physics_dt
        else:
            h = dt
        if self.integrator == 'euler':
            a = self.accelerations(force, state)
            v = torch.cat([state[:, 1:2], state[:, 7:]], 1)+h*a
            q = torch.cat([state[:, :1], state[:, 2:7]], 1)+h*v
            return torch.cat([q[:, :1], v[:, :1], q[:, 1:], v[:, 1:]], 1)
        k1 = self.derivative(state, force)
        k2 = self.derivative(state+h*k1/2, force)
        k3 = self.derivative(state+h*k2/2, force)
        k4 = self.derivative(state+h*k3, force)
        return state+h*(k1+2*k2+2*k3+k4)/6

    def energy(self, state=None):
        if state is None:
            s = self.state
        else:
            s = state
        M = self.mass_matrix(s[:, 2:7])
        v = torch.cat([s[:, 1:2], s[:, 7:]], 1)
        kinetic = .5*torch.einsum('bi,bij,bj->b', v, M, v)
        potential = self.gravity*(self.energy_weights*s[:, 2:7].cos()).sum(1)
        relative_kinetic = .5*torch.einsum('bi,bij,bj->b', s[:, 7:], M[:, 1:, 1:], s[:, 7:])
        return kinetic, potential, relative_kinetic

    @torch.no_grad()
    def step(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError('Controller produced a non-finite action')
        force = self.tensor(actions).reshape(-1).clamp(-1, 1)*self.max_force
        active = ~self.finished
        was_active = active.clone()
        failed_track = torch.zeros_like(active)
        failed_numeric = torch.zeros_like(active)
        had_success = self.success_achieved.clone()
        for _ in range(self.action_repeat):
            candidate = self.integrate(self.state, torch.where(active, force, 0))
            finite = torch.isfinite(candidate).all(1) & torch.isfinite(self.observe(candidate)).all(1)
            numeric = active & ~finite
            track = active & finite & (candidate[:, 0].abs() > self.track_limit)
            self.state = torch.where((active & finite)[:, None], candidate, self.state)
            self.state[:, 2:7] = self.wrap(self.state[:, 2:7])
            self.elapsed_ticks += active.long()
            failed_numeric |= numeric
            failed_track |= track
            angles_ok = (self.state[:, 2:7].abs() <= self.balance_angle).all(1)
            velocity_ok = (self.state[:, 7:].abs() <= self.balance_velocity).all(1)
            balanced = active & ~numeric & ~track & angles_ok & velocity_ok
            new_hold = torch.where(balanced, self.hold_ticks + 1, 0)
            self.hold_ticks = torch.where(active, new_hold, self.hold_ticks)
            self.max_hold_ticks = torch.maximum(self.max_hold_ticks, self.hold_ticks)
            self.success_achieved |= self.hold_ticks >= self.success_ticks
            active &= ~(numeric | track)
        self.steps += (~self.finished).long()
        terminated = failed_track | failed_numeric
        timed_out = (self.steps >= self.max_steps) & ~terminated
        self.finished |= terminated | timed_out
        each = (self.state[:, 2:7].cos()+1)/2
        kinetic, potential, relative_kinetic = self.energy()
        # Reward is not used to check success
        reward = each.mean(1)*each.min(1).values
        reward = torch.where(terminated, -torch.ones_like(reward), reward)
        reward = torch.where(was_active, reward, 0)
        self.episode_returns += reward
        info = dict(success=self.success_achieved.clone(), newly_successful=self.success_achieved & ~had_success,
                    all_balanced=self.hold_ticks > 0, balanced_time=self.hold_ticks*self.physics_dt,
                    max_hold_time=self.max_hold_ticks*self.physics_dt,
                    failed_track=failed_track, failed_numeric=failed_numeric,
                    terminated=terminated, timed_out=timed_out, upright=each.mean(1),
                    weakest_upright=each.min(1).values, kinetic_energy=kinetic,
                    potential_energy=potential, relative_kinetic_energy=relative_kinetic,
                    episode_return=self.episode_returns.clone(), elapsed_time=self.elapsed_ticks*self.physics_dt)
        return self.observation(), reward.float(), self.finished.clone(), info


class FivePoleEnv:
    # Single environment version
    def __init__(self, seed=None, config=None):
        self.batch = TorchFivePoleBatch(1, 'cpu', seed=0 if seed is None else seed, config=config)
        self.rng = np.random.default_rng(seed)

    def __getattr__(self, name):
        value = getattr(self.batch, name)
        if isinstance(value, torch.Tensor):
            a = value.cpu().numpy()
            if a.size == 1:
                return a.item()
            return a
        return value

    @property
    def state(self):
        return self.batch.state[0].numpy()

    @state.setter
    def state(self, value):
        self.batch.state[0] = self.batch.tensor(value)

    def reset(self, state=None):
        if state is None:
            states = None
        else:
            states = np.asarray(state)[None]
        result = self.batch.reset(states=states)
        return result[0].numpy()

    def get_observation(self):
        return self.batch.observation()[0].numpy()

    def step(self, action):
        obs, reward, done, info = self.batch.step(self.batch.tensor([float(np.asarray(action).reshape(-1)[0])]))
        new_info = {}
        for key, value in info.items():
            new_info[key] = value[0].item()
        info = new_info
        info.update(x=float(self.state[0]), max_angle_deg=float(np.max(np.abs(np.degrees(self.state[2:7])))))
        return obs[0].numpy(), reward.item(), done.item(), info

    def _mass_matrix(self, angles):
        return self.batch.mass_matrix(self.batch.tensor(angles)[None])[0].numpy()

    def _accelerations(self, state, force):
        a = self.batch.accelerations(self.batch.tensor([force]), self.batch.tensor(state)[None])[0].numpy()
        return a[0], a[1:]

    def disturb(self, strength=.8):
        i = int(self.rng.integers(5))
        direction = float(self.rng.choice([-1, 1]))
        self.state[7 + i] += direction * strength
        return i
