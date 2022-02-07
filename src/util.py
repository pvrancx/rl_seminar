from collections import Callable

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import Env

from src.approx import FeatureMap
from src.qlearning import Learner


def evaluate(env: Env, learner: Learner, n_eps: int):
    """
    Evaluate learner on given environment
    :param env: gym evaluation environment
    :param learner: reinforcement learning agent
    :param n_eps: number evluation episodes
    :return:
    """
    steps = []
    rewards = []
    for ep in range(n_eps):
        state = env.reset()
        done = False
        total_reward = 0.
        n_steps = 0
        while not done:
            action = learner.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            n_steps += 1
        steps.append(n_steps)
        rewards.append(total_reward)
    return np.mean(steps), np.mean(rewards)


def run_learner(env: Env, learner: Learner, n_eps: int, eval_env: Env, eval_eps: int = 10, log_steps: int = 2000):
    """
    Train rl agent
    """
    exp_log = {'steps': [], 'rewards': [], 'episode_length': []}
    n_steps = 0
    for ep in range(n_eps):
        state = env.reset()
        action = learner.start(state)
        done = False
        total_reward = 0.
        while not done:
            nstate, reward, done, _ = env.step(action)
            action = learner.update(nstate, reward, done)
            total_reward += reward
            n_steps += 1
            if log_steps > 0 and n_steps % log_steps == 0:
                mean_steps, mean_rewards = evaluate(eval_env, learner, eval_eps)
                exp_log['steps'].append(n_steps)
                exp_log['rewards'].append(mean_rewards)
                exp_log['episode_length'].append(mean_steps)

        print(f"Episode {ep} - total reward {total_reward}")

    return exp_log


def generate_data(env: Env, n_samples: int, policy: Callable = lambda s: np.random.randint(3)):
    """
    Generate a fixed number of samples from a gym environment using fixed action selection policy
    """
    samples = []
    state = env.reset()
    while len(samples) < n_samples:
        action = policy(state)
        nstate, reward, done, _ = env.step(action)
        samples.append((state, action, reward, nstate, done))
        if done:
            state = env.reset()
        else:
            state = nstate.copy()
    return samples


def plot_mc_value(observation_space: gym.spaces.Box, feature_map: FeatureMap, weights: np.ndarray, fig=None):
    """
    Plot MountainCar value function
    """
    xs = np.linspace(observation_space.low[0], observation_space.high[0])
    ys = np.linspace(observation_space.low[1], observation_space.high[1])
    gridx, gridy = np.meshgrid(xs, ys)

    features = feature_map(np.hstack([gridx.reshape(-1,1), gridy.reshape(-1,1)]))
    vals = np.max(features @ weights, axis=-1)
    fig = fig or plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(gridx, gridy, np.reshape(vals, gridx.shape))
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax.set_zlabel('value')
    return ax


def smooth(vector: np.ndarray, window_size: int):
    """Time series smoothing"""
    return np.convolve(vector, np.ones(window_size), mode='valid') / window_size

