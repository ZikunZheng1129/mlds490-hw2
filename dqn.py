"""Small, shared DQN implementation for CartPole-v1 and ALE/MsPacman-v5.

Configuration and sampled records are immutable. Replay storage and optimizer
state are the explicitly stateful parts of the learning algorithm.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    task: str
    seed: int
    gamma: float
    learning_rate: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    train_frequency: int = 1
    target_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 30_000
    gradient_clip: float = 10.0
    reward_clip: bool = False
    frame_stack: int = 4

    def __post_init__(self):
        if self.task not in {"cartpole", "mspacman"}:
            raise ValueError("task must be cartpole or mspacman")
        positive = (self.batch_size, self.buffer_size, self.train_frequency,
                    self.target_interval, self.epsilon_decay_steps, self.frame_stack)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in positive):
            raise ValueError("sizes, frequencies and frame_stack must be positive integers")
        if self.buffer_size < self.batch_size or self.learning_starts < 0:
            raise ValueError("buffer must hold a batch; learning_starts must be nonnegative")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("gamma/epsilon must be in [0, 1], with epsilon_end <= start")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be finite and positive")


def defaults(task: str, seed: int = 7) -> Config:
    if task == "mspacman":
        return Config(task, seed, 0.99, learning_rate=1e-4, batch_size=32,
                      learning_starts=10_000, train_frequency=4, target_interval=2_000,
                      epsilon_end=0.05, epsilon_decay_steps=500_000, reward_clip=True)
    return Config(task, seed, 0.95)


def preprocess_observation(observation: np.ndarray) -> np.ndarray:
    """Assignment crop/grayscale/contrast, stored losslessly as unsigned bytes.

    Network normalization later subtracts 128, so this has the same centered
    pixel representation as the assignment without storing negative byte values.
    """
    if observation.shape != (210, 160, 3) or observation.dtype != np.uint8:
        raise ValueError("expected an RGB uint8 Atari image with shape (210, 160, 3)")
    summed = observation[1:176:2, ::2].sum(axis=2, dtype=np.uint16)
    contrasted = np.where(summed == 210 + 164 + 74, 0, summed)
    return (contrasted // 3).astype(np.uint8)


class ImageObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (88, 80), dtype=np.uint8)

    def observation(self, observation):
        return preprocess_observation(observation)


def make_env(task: str, seed: int | None = None, frame_stack: int = 4,
             repeat_action_probability: float = 0.25,
             max_episode_steps: int | None = None):
    """Explicit overrides are for labeled validation experiments, not main scores."""
    if task not in {"cartpole", "mspacman"}:
        raise ValueError("unknown task")
    if not isinstance(frame_stack, int) or frame_stack < 1:
        raise ValueError("frame_stack must be positive")
    if not 0 <= repeat_action_probability <= 1:
        raise ValueError("repeat_action_probability must be in [0, 1]")
    if max_episode_steps is not None and max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    options = {} if max_episode_steps is None else {"max_episode_steps": max_episode_steps}
    if task == "cartpole":
        env = gym.make("CartPole-v1", **options)
    else:
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make("ALE/MsPacman-v5", obs_type="rgb", frameskip=4,
                       repeat_action_probability=repeat_action_probability,
                       full_action_space=False, **options)
        env = gym.wrappers.FrameStackObservation(ImageObservation(env), frame_stack)
    if seed is not None:
        env.action_space.seed(seed)
    return env


class QNetwork(nn.Module):
    def __init__(self, observation_shape, n_actions: int):
        super().__init__()
        shape = tuple(observation_shape)
        if n_actions < 1 or len(shape) not in {1, 3} or any(v < 1 for v in shape):
            raise ValueError("QNetwork needs a vector or CHW image and positive action count")
        self.image_input = len(shape) == 3
        if self.image_input:
            encoder = nn.Sequential(
                nn.Conv2d(shape[0], 32, 8, 4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
            with torch.no_grad():
                features = encoder(torch.zeros(1, *shape)).shape[1]
            self.network = nn.Sequential(encoder, nn.Linear(features, 256),
                                         nn.ReLU(), nn.Linear(256, n_actions))
        else:
            self.network = nn.Sequential(nn.Linear(shape[0], 128), nn.ReLU(),
                                         nn.Linear(128, 128), nn.ReLU(),
                                         nn.Linear(128, n_actions))

    def forward(self, observations):
        values = observations.float()
        if self.image_input:
            values = (values - 128.0) / 128.0
        return self.network(values)


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    """Uniform replay with bounded O(1) insertion and immutable stored records."""
    def __init__(self, capacity: int, seed: int = 7):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._records: list[Transition | None] = [None] * capacity
        self._size = self._position = 0
        self._shape = None
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return self._size

    def add(self, state, action, reward, next_state, terminated, truncated=False):
        first, second = np.asarray(state), np.asarray(next_state)
        shape = self._shape if self._shape is not None else first.shape
        if first.shape != shape or second.shape != shape:
            raise ValueError("replay observation shapes must match")
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError("replay observations must be finite")
        if not isinstance(action, (int, np.integer)) or action < 0:
            raise ValueError("action must be a nonnegative integer")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        stored, following = np.array(first, copy=True), np.array(second, copy=True)
        stored.setflags(write=False)
        following.setflags(write=False)
        record = Transition(stored, int(action), float(reward), following,
                            bool(terminated), bool(truncated))
        self._records[self._position] = record
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._shape = shape

    def sample(self, batch_size: int, device="cpu") -> Batch:
        if not isinstance(batch_size, int) or not 1 <= batch_size <= self._size:
            raise ValueError("batch_size must be positive and no larger than stored replay")
        records = [self._records[i] for i in self._rng.choice(self._size, batch_size, replace=False)]
        def tensor(values, dtype=None):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)
        return Batch(tensor(np.stack([r.state for r in records])),
                     tensor([r.action for r in records], torch.long),
                     tensor([r.reward for r in records], torch.float32),
                     tensor(np.stack([r.next_state for r in records])),
                     tensor([r.terminated for r in records], torch.bool),
                     tensor([r.truncated for r in records], torch.bool))


def bellman_targets(rewards, terminated, next_values, gamma: float):
    """True terminals stop bootstrap; an external time limit does not."""
    with torch.no_grad():
        return rewards + gamma * (~terminated).float() * next_values


class DQNAgent:
    def __init__(self, observation_shape, n_actions: int, config: Config, device="cpu"):
        self.observation_shape = tuple(int(value) for value in observation_shape)
        self.n_actions = int(n_actions)
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(config.seed)
        self.online = QNetwork(self.observation_shape, n_actions).to(self.device)
        self.target = deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)

    def act(self, state, epsilon: float, rng: np.random.Generator) -> int:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        with torch.no_grad():
            batch = torch.as_tensor(np.asarray(state), device=self.device).unsqueeze(0)
            return int(self.online(batch).argmax(1).item())

    def update(self, batch: Batch) -> dict[str, float]:
        with torch.no_grad():
            next_values = self.target(batch.next_states).max(1).values
            targets = bellman_targets(batch.rewards, batch.terminated, next_values, self.config.gamma)
        estimates = self.online(batch.states).gather(1, batch.actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(estimates, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite DQN loss; inspect observations and targets")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip,
                                       error_if_nonfinite=True)
        self.optimizer.step()
        return {"loss": float(loss.detach()),
                "td_abs": float((estimates.detach() - targets).abs().mean()),
                "grad_norm": float(norm)}

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": asdict(self.config), "observation_shape": self.observation_shape,
                   "n_actions": self.n_actions, "online": self.online.state_dict(),
                   "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "metadata": json.loads(json.dumps(metadata or {}, allow_nan=False))}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=True)
        agent = cls(payload["observation_shape"], payload["n_actions"],
                    Config(**payload["config"]), device)
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        return agent, payload["metadata"]
