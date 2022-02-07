from abc import ABC, abstractmethod

import gym
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


class FeatureMap(ABC):
    @abstractmethod
    def __call__(self, state: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def n_features(self) -> int:
        pass


class PolyFeatures(FeatureMap):
    """
    Features for polynomial regression
    """
    def __init__(self, observation_space: gym.spaces.Box, degree: int):
        self.phi = Pipeline(
            [('scaler', MinMaxScaler((-1., 1.))), ('poly', PolynomialFeatures(degree=degree, include_bias=True))])
        self.phi.fit(np.array([observation_space.low, observation_space.high]))

    def __call__(self, state):
        feat = self.phi.transform(np.atleast_2d(state))
        return feat / np.abs(feat).sum(axis=-1, keepdims=True)

    @property
    def n_features(self):
        return self.phi['poly'].n_output_features_
