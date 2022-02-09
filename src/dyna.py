import numpy as np

from src.models import ReplayData
from src.qlearning import Qlearner


class QlearnerWithReplay(Qlearner):
    def __init__(self, replay_size: int, n_samples: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_db = ReplayData(replay_size)
        self.n_samples = n_samples

    def replay_update(self):
        features, actions, rewards, nfeatures, dones = self.replay_db.sample(self.n_samples)
        qvals, nqvals = self.get_values(features), self.get_values(nfeatures)
        deltas = rewards - qvals[np.arange(actions.size), actions]
        deltas += self.discount * np.max(nqvals, -1) * (1. - dones).astype(float)
        self.weights[:, actions] += self.lr * (deltas.reshape(-1, 1) * features).T

    def update(self, nstate: np.ndarray, reward: float, done: bool):
        feat = self.feat.copy()
        act = self.action
        nact = super().update(nstate, reward, done)
        self.replay_db.append((feat, act, reward, self.feat, done))
        if len(self.replay_db) > self.n_samples:
            self.replay_update()
        return nact
