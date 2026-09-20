"""Episode rules. Learning code and drawing code both use this class."""

import math
import numpy as np
from physics import integrate, wrap_angle

CONTROL_STEP = 0.02
MAX_FORCE = 45.0
TRACK_LIMIT = 2.4
EPISODE_SECONDS = 15.0


class PendulumEnvironment:
    def __init__(self, count=1, seed=0, physics_step=0.005):
        substeps = round(CONTROL_STEP / physics_step)
        if physics_step <= 0 or substeps < 1 or not math.isclose(substeps * physics_step, CONTROL_STEP):
            raise ValueError('Physics step must divide the 0.02-second control interval.')
        self.count = count
        self.random = np.random.default_rng(seed)
        self.physics_step = physics_step
        self.substeps = substeps
        self.reset()

    def reset(self):
        self.states = np.zeros((self.count, 12))
        self.states[:, 0] = self.random.normal(0, 0.01, self.count)
        self.states[:, 1] = self.random.normal(0, 0.02, self.count)
        angle = math.pi + self.random.normal(0, 0.01, self.count)
        for link in range(5):
            if link > 0:
                angle = angle + self.random.normal(0, 0.035, self.count)
            self.states[:, 2 + link] = wrap_angle(angle)
            self.states[:, 7 + link] = self.random.normal(0, 0.03, self.count)
        self.hold_seconds = np.zeros(self.count)
        self.best_hold = np.zeros(self.count)
        self.maximum_position = np.abs(self.states[:, 0])
        self.success = np.zeros(self.count, dtype=bool)
        self.failed = np.zeros(self.count, dtype=bool)
        self.finished = np.zeros(self.count, dtype=bool)
        self.step_number = 0
        return self.observation()

    def observation(self):
        values = [self.states[:, 0] / TRACK_LIMIT, self.states[:, 1] / 5]
        for link in range(5):
            angle = self.states[:, 2 + link]
            values.append(np.sin(angle))
            values.append(np.cos(angle))
            values.append(self.states[:, 7 + link] / 10)
        return np.stack(values, axis=1)

    def step(self, actions):
        actions = np.asarray(actions, dtype=float).reshape(self.count)
        if not np.isfinite(actions).all():
            raise ValueError('Actions must be finite.')
        forces = np.clip(actions, -1, 1) * MAX_FORCE

        for substep in range(self.substeps):
            active = ~self.finished
            # Do not integrate a failed state again.
            candidate = self.states.copy()
            candidate[active] = integrate(self.states[active], forces[active], self.physics_step)
            finite = np.isfinite(candidate).all(axis=1)
            on_track = np.abs(candidate[:, 0]) <= TRACK_LIMIT
            failed_now = active & (~finite | ~on_track)
            self.failed |= failed_now
            self.finished |= failed_now
            for link in range(5):
                candidate[:, 2 + link] = wrap_angle(candidate[:, 2 + link])
            self.states[active & finite] = candidate[active & finite]
            self.maximum_position = np.maximum(self.maximum_position, np.abs(self.states[:, 0]))

            # Check every link explicitly. One bad link breaks the hold.
            balanced = active & ~failed_now
            for link in range(5):
                angle_ok = np.abs(self.states[:, 2 + link]) <= math.radians(10)
                speed_ok = np.abs(self.states[:, 7 + link]) <= 0.5
                balanced = balanced & angle_ok & speed_ok
            self.hold_seconds[balanced] += self.physics_step
            self.hold_seconds[active & ~balanced] = 0
            self.best_hold = np.maximum(self.best_hold, self.hold_seconds)
            self.success |= self.hold_seconds >= 2.0 - 1e-10

        self.step_number += 1
        if self.step_number * CONTROL_STEP >= EPISODE_SECONDS:
            self.finished[:] = True

        # Reward guides PPO; it never decides whether a trial succeeded.
        reward = np.ones(self.count)
        for link in range(5):
            reward *= (np.cos(self.states[:, 2 + link]) + 1) / 2
        reward[self.failed] = -1
        return self.observation(), reward, self.finished.copy()
