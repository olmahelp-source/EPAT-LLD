"""Reinforcement-learning agents: tabular Q-learning, a deep Q-network and a policy-gradient agent.

All share one small interface, so training and evaluation code does not care which is used:

    act(obs, greedy=False) -> action index
    observe(obs, action, reward, next_obs, done)     # learn from one transition
    end_episode()                                    # learn from the finished episode (policy gradient), decay exploration

A caution that belongs next to the code: an RL agent trained on one price history has seen that history thousands of
times. In-sample reward means nothing. Judge it only with :func:`evaluate` on data it never trained on, with costs.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.exceptions import ModelError, ValidationError
from ..dl.nn import Activation, Adam, Dense, Layer, Sequential
from ..utils.numeric import softmax
from .env import TradingEnv


class _Exploring:
    """Epsilon-greedy schedule shared by the value-based agents."""

    def _init_eps(self, epsilon: float, epsilon_min: float, epsilon_decay: float) -> None:
        if not 0 <= epsilon_min <= epsilon <= 1 or not 0 < epsilon_decay <= 1:
            raise ValidationError("need 0 <= epsilon_min <= epsilon <= 1 and 0 < epsilon_decay <= 1")
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class QLearningAgent(_Exploring):
    """Tabular Q-learning over quantile-binned observations. Call :meth:`prepare` with representative observations
    (training data) first: it fixes the bin edges. Transparent and hard to overfit with few bins; the table can be read."""

    def __init__(self, n_actions: int, bins: int = 4, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 1.0,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.97, seed: Optional[int] = 0) -> None:
        if n_actions < 2 or bins < 2 or not 0 < alpha <= 1 or not 0 <= gamma < 1:
            raise ValidationError("need n_actions >= 2, bins >= 2, 0 < alpha <= 1, 0 <= gamma < 1")
        self.n_actions, self.bins, self.alpha, self.gamma = n_actions, bins, alpha, gamma
        self._init_eps(epsilon, epsilon_min, epsilon_decay)
        self._rng = np.random.default_rng(seed)
        self.q: Dict[Tuple[int, ...], np.ndarray] = {}
        self._edges: Optional[List[np.ndarray]] = None

    def prepare(self, observations) -> "QLearningAgent":
        obs = np.asarray(observations, float)
        qs = np.linspace(0, 1, self.bins + 1)[1:-1]
        self._edges = [np.unique(np.quantile(obs[:, j], qs)) for j in range(obs.shape[1])]
        return self

    def _state(self, obs) -> Tuple[int, ...]:
        if self._edges is None:
            raise ModelError("call prepare(observations) before using the agent")
        return tuple(int(np.searchsorted(e, v)) for e, v in zip(self._edges, np.asarray(obs, float)))

    def q_values(self, obs) -> np.ndarray:
        return self.q.get(self._state(obs), np.zeros(self.n_actions))

    def act(self, obs, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        q = self.q_values(obs)
        return int(self._rng.choice(np.flatnonzero(q == q.max())))        # break ties at random, not always action 0

    def observe(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        s = self._state(obs)
        q = self.q.setdefault(s, np.zeros(self.n_actions))
        target = reward + (0.0 if done else self.gamma * float(self.q_values(next_obs).max()))
        q[action] += self.alpha * (target - q[action])


def _mlp(n_in: int, hidden: Sequence[int], n_out: int, rng: np.random.Generator, lr: float) -> Sequential:
    layers: List[Layer] = []
    prev = n_in
    for h in hidden:
        layers += [Dense(prev, h, rng), Activation("relu")]
        prev = h
    layers.append(Dense(prev, n_out, rng, init="glorot"))
    return Sequential(layers, "mse", Adam(lr, clip_norm=5.0))


class DQNAgent(_Exploring):
    """Deep Q-network: experience replay, a periodically synchronised target network, Huber-clipped TD errors and
    (by default) double Q-learning, which stops the max operator from systematically overestimating action values."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (32, 32), lr: float = 1e-3, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.95, buffer_size: int = 10_000,
                 batch_size: int = 32, warmup: int = 200, target_sync: int = 250, double: bool = True, seed: Optional[int] = 0) -> None:
        if obs_dim < 1 or n_actions < 2 or not 0 <= gamma < 1 or batch_size < 1 or warmup < batch_size:
            raise ValidationError("need obs_dim >= 1, n_actions >= 2, 0 <= gamma < 1, warmup >= batch_size >= 1")
        self.obs_dim, self.n_actions, self.gamma, self.batch_size, self.warmup = obs_dim, n_actions, gamma, batch_size, warmup
        self.target_sync, self.double = target_sync, double
        self._init_eps(epsilon, epsilon_min, epsilon_decay)
        self._rng = np.random.default_rng(seed)
        self.online = _mlp(obs_dim, hidden, n_actions, self._rng, lr)
        self.target = _mlp(obs_dim, hidden, n_actions, self._rng, lr)
        self.target.set_weights(self.online.get_weights())
        self._buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=buffer_size)
        self.steps = 0
        self.losses: List[float] = []

    def q_values(self, obs) -> np.ndarray:
        return self.online.forward(np.asarray(obs, float).reshape(1, -1))[0]

    def act(self, obs, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(obs)))

    def observe(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        self._buffer.append((np.asarray(obs, float), int(action), float(reward), np.asarray(next_obs, float), bool(done)))
        if len(self._buffer) >= self.warmup:
            self._learn()

    def _learn(self) -> None:
        idx = self._rng.integers(len(self._buffer), size=self.batch_size)
        batch = [self._buffer[i] for i in idx]
        S, A = np.array([b[0] for b in batch]), np.array([b[1] for b in batch])
        R, S2, D = np.array([b[2] for b in batch]), np.array([b[3] for b in batch]), np.array([b[4] for b in batch], float)
        q_next = self.target.forward(S2)
        rows = np.arange(self.batch_size)
        nxt = q_next[rows, np.argmax(self.online.forward(S2), axis=1)] if self.double else q_next.max(axis=1)
        y = R + self.gamma * (1.0 - D) * nxt
        q = self.online.forward(S, training=True)                       # must be the last forward pass before backward
        td = q[rows, A] - y
        if not np.isfinite(td).all():
            raise ModelError("DQN diverged (non-finite TD error); lower the learning rate or reward_scale")
        grad = np.zeros_like(q)
        grad[rows, A] = np.clip(td, -1.0, 1.0) / self.batch_size        # derivative of the Huber loss
        self.online.backward(grad)
        self.online.optimizer.step(self.online.params, self.online.grads)
        self.losses.append(float(np.mean(np.where(np.abs(td) <= 1, 0.5 * td ** 2, np.abs(td) - 0.5))))
        self.steps += 1
        if self.steps % self.target_sync == 0:
            self.target.set_weights(self.online.get_weights())


class ReinforceAgent:
    """Policy gradient (REINFORCE) with a return baseline and an entropy bonus. The policy is stochastic while training
    and takes the most probable action when ``greedy``. Learns once per episode, from the whole trajectory."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (32,), lr: float = 3e-3, gamma: float = 0.95,
                 entropy_coef: float = 0.01, seed: Optional[int] = 0) -> None:
        if obs_dim < 1 or n_actions < 2 or not 0 <= gamma <= 1 or entropy_coef < 0:
            raise ValidationError("need obs_dim >= 1, n_actions >= 2, 0 <= gamma <= 1, entropy_coef >= 0")
        self.obs_dim, self.n_actions, self.gamma, self.entropy_coef = obs_dim, n_actions, gamma, entropy_coef
        self._rng = np.random.default_rng(seed)
        self.policy = _mlp(obs_dim, hidden, n_actions, self._rng, lr)
        self._traj: List[Tuple[np.ndarray, int, float]] = []
        self.epsilon = 0.0

    def probabilities(self, obs) -> np.ndarray:
        return softmax(self.policy.forward(np.asarray(obs, float).reshape(1, -1)))[0]

    def act(self, obs, greedy: bool = False) -> int:
        p = self.probabilities(obs)
        return int(np.argmax(p)) if greedy else int(self._rng.choice(self.n_actions, p=p))

    def observe(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        self._traj.append((np.asarray(obs, float), int(action), float(reward)))

    def end_episode(self) -> None:
        if len(self._traj) < 2:
            self._traj = []
            return
        S, A = np.array([t[0] for t in self._traj]), np.array([t[1] for t in self._traj])
        G = np.zeros(len(self._traj))
        running = 0.0
        for i in range(len(self._traj) - 1, -1, -1):                    # discounted reward-to-go
            running = self._traj[i][2] + self.gamma * running
            G[i] = running
        G = (G - G.mean()) / (G.std() + 1e-8)                           # baseline + scale: lower-variance gradient
        logits = self.policy.forward(S, training=True)
        p = softmax(logits)
        grad = p.copy()
        grad[np.arange(len(A)), A] -= 1.0                               # d(-log pi(a|s)) / d logits
        grad *= G[:, None]
        logp = np.log(np.clip(p, 1e-12, None))
        entropy_grad = p * (logp - (p * logp).sum(axis=1, keepdims=True))           # d(-entropy) / d logits
        self.policy.backward((grad + self.entropy_coef * entropy_grad) / len(A))
        self.policy.optimizer.step(self.policy.params, self.policy.grads)
        self._traj = []


# ---- loops ----------------------------------------------------------------------------------------------------------------
def train(agent: Any, env: TradingEnv, episodes: int = 50, callback: Optional[Callable[[int, float], None]] = None) -> List[float]:
    """Run ``episodes`` training episodes; returns the total (scaled) reward of each."""
    if episodes < 1:
        raise ValidationError("episodes must be >= 1")
    totals: List[float] = []
    for ep in range(episodes):
        obs, done, total = env.reset(), False, 0.0
        while not done:
            action = agent.act(obs)
            nxt, reward, done, _ = env.step(action)
            agent.observe(obs, action, reward, nxt, done)
            obs, total = nxt, total + reward
        agent.end_episode()
        totals.append(total)
        if callback:
            callback(ep, total)
    return totals


def evaluate(agent: Any, env: TradingEnv, periods_per_year: float = 252.0) -> Dict[str, Any]:
    """One greedy pass over the whole environment (no exploration, no learning). Net of costs."""
    saved, env.episode_length = env.episode_length, None
    try:
        obs, done = env.reset(0), False
        pnl: List[float] = []
        positions: List[float] = []
        while not done:
            obs, _, done, info = env.step(agent.act(obs, greedy=True))
            pnl.append(info["pnl"])
            positions.append(info["position"])
    finally:
        env.episode_length = saved
    r = np.array(pnl)
    pos = np.array(positions)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return {"total_return": float(np.prod(1 + r) - 1), "sharpe": float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0,
            "equity": np.cumprod(1 + r), "positions": pos, "trades": int((np.diff(np.concatenate([[0.0], pos])) != 0).sum()),
            "exposure": float(np.mean(np.abs(pos))), "steps": len(r)}


def as_policy(agent: Any, feature_fn: Callable[[Dict[str, np.ndarray]], np.ndarray], actions: Sequence[float] = (-1.0, 0.0, 1.0),
              include_position: bool = True) -> Callable[[Dict[str, np.ndarray]], float]:
    """Wrap a trained agent as a ``strategy.PolicyStrategy`` policy for one symbol: ``feature_fn(window)`` must return
    the same (standardised) features the agent was trained on. The wrapper remembers the last target position because
    the agent's observation includes it."""
    state = {"position": 0.0}

    def policy(window: Dict[str, np.ndarray]) -> float:
        f = np.asarray(feature_fn(window), float).ravel()
        if not np.isfinite(f).all():
            return state["position"]
        obs = np.append(f, state["position"]) if include_position else f
        state["position"] = float(actions[agent.act(obs, greedy=True)])
        return state["position"]
    return policy
