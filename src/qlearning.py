import abc
from abc import abstractmethod

import numpy as np

from src.approx import FeatureMap


def egreedy(vals: np.ndarray, eps: float) -> int:
    """
    Egreedy action selection
    """
    if np.random.rand() < eps:
        return np.random.choice(vals.size)
    else:
        # randomize over all actions with maximum qval - this prevents issues when all qvals are equal
        max_q = np.max(vals)
        return np.random.choice(np.arange(vals.size)[vals == max_q])


class Learner(abc.ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        pass

    @abstractmethod
    def start(self, state: np.ndarray) -> int:
        pass

    @abstractmethod
    def update(self, nstate: np.ndarray, reward: float, done: bool) -> int:
        pass

    def __call__(self, state: np.ndarray) -> int:
        return self.select_action(state)


class Qlearner(Learner):
    """
    Epsilon greedy Qlearner with linear qvalue approximation
    """
    def __init__(self, feature_map: FeatureMap, n_actions: int, lr: float, discount: float, eps: float):
        self.phi = feature_map
        self.weights = np.zeros((feature_map.n_features, n_actions))
        self.lr = lr
        self.discount = discount
        self.eps = eps
        self.feat = None
        self.action = None

    def get_state_value(self, states):
        features = self.phi(states)
        return self.get_values(features)

    def get_values(self, features):
        return features @ self.weights

    def select_action(self, state):
        features = self.phi(state).flatten()
        values = self.get_values(features).flatten()
        return egreedy(values, self.eps)

    def start(self, state):
        self.feat = self.phi(state).flatten()
        self.action = self.select_action(state)
        return self.action

    def update(self, nstate, reward, done):
        qvals = self.get_values(self.feat)
        nphi = self.phi(nstate).flatten()
        nqvals = self.get_values(nphi)
        delta = reward - qvals[self.action]
        delta += self.discount * np.max(nqvals) * float(1. - done)
        self.weights[:, self.action] += self.lr * delta * self.feat
        self.feat = nphi
        self.action = self.select_action(nstate)
        return self.action
