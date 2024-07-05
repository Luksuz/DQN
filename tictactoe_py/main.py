from TicTacToe import TicTacToe
env = TicTacToe()

from collections import defaultdict
import numpy as np
from typing import Tuple

class Agent:
    def __init__(self, lr, initial_epsilon, epsilon_decay, min_epsilon, discout=0.95):
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.shape))
        self.lr = lr
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.discount_factor = discout

        self.training_error = []

    def get_action(self, obs, env):
        if np.random.random() < self.epsilon:
            return [1, np.random.choice(env.available_indices)]

        else:
            q_values = self.q_values[obs]
            valid_actions = [(i, q_values[i]) for i in env.available_indices]
            action = max(valid_actions, key=lambda x: x[1])[0]
            return [1, action]

    def update(self, obs: Tuple[int], action, reward, terminated, next_obs: Tuple[int]):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]

        self.q_values[obs][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
        

obs, info = env.reset(player2_advantage=True, human_opponent=True)

action = agent.get_action(obs, env)
next_obs, reward, terminated, truncated, info = env.step(action, human_opponent=True)

done = terminated

agent.update(obs, action, reward, terminated, next_obs)
obs = next_obs

env.display_interface()