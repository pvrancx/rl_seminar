import numpy as np
from sklearn.linear_model import LinearRegression

from src.approx import FeatureMap


class ReplayData:
    """
    Replay data store. Stores transition samples and allows sampling batches of transitions.
    """
    def __init__(self, max_size: int):
        self.db = []
        self.max_size = max_size
        self.idx = 0

    def append(self, sample):
        if len(self.db) < self.max_size:
            self.db.append(sample)
        else:
            self.db[self.idx % self.max_size] = sample
        self.idx += 1

    def sample(self, n: int):
        idx = np.random.choice(len(self), size=n, replace=False)
        samples = [self.db[i] for i in idx]
        data_arrays = map(np.array, zip(*samples))
        return tuple(data_arrays)

    def __len__(self):
        return len(self.db)

    def clear(self):
        self.db = []
        self.idx = 0


class EnvModel:
    """
    Environment Model. Trains an expected linear model to predict nstates and rewards.
    """
    def __init__(self, feature_map: FeatureMap):
        self.reward_models = {}
        self.transition_models = {}
        self.phi = feature_map

    def train(self, states, actions, rewards, nstates):
        actions = np.atleast_1d(actions)
        states = np.atleast_2d(states)
        features = self.phi(states)
        for a in np.unique(actions):
            mask = (actions == a)
            feat = features[mask, :]
            self.reward_models[a] = LinearRegression().fit(feat, rewards[mask])
            self.transition_models[a] = LinearRegression().fit(feat, nstates[mask, :])

    def predict(self, states, actions):
        actions = np.atleast_1d(actions)
        states = np.atleast_2d(states)
        rewards = np.zeros_like(actions)
        nstates = np.zeros_like(states)

        features = self.phi(states)
        for a in np.unique(actions):
            assert a in self.reward_models, 'train first'
            assert a in self.transition_models, 'train first'
            mask = ( actions == a)
            rewards[mask] = self.reward_models[a].predict(features[mask, :])
            nstates[mask, :] = self.transition_models[a].predict(features[mask, :])
        return nstates, rewards


class ModelWrapper:
    """
    Wrapper to provide (partial) gym Environment interface for model
    """
    def __init__(self, env_model: EnvModel):
        self.model = env_model
        self.state = None
        self.nsteps = 0

    def reset(self):
        self.state = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])
        self.nsteps = 0
        return np.array(self.state, dtype=np.float32)

    def is_goal(self, state):
        return bool(state[0] >= 0.5 or self.nsteps >= 5000)

    def step(self, action):
        nstate, reward = self.model.predict(self.state, action)
        self.state = nstate.flatten()
        self.nsteps += 1
        return self.state, reward[-1], self.is_goal(self.state), {}
