import numpy as np
from typing import Tuple

class Agent:
    def __init__(self, lr, initial_epsilon, epsilon_decay, min_epsilon, q_values, discout=0.95):
        self.q_values = q_values
        self.lr = lr
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.discount_factor = discout

        self.training_error = []

    def get_action(self, obs, available_indices: Tuple[int]):
        """if len(available_indices) >= 8 and available_indices[4] == 0:
            return [1, 4]"""
        if np.random.random() < self.epsilon:
            return [1, np.random.choice(available_indices)]
        else:
            q_values = self.q_values[obs]
            valid_actions = [(i, q_values[i]) for i in available_indices]
            action = max(valid_actions, key=lambda x: x[1])[0]
            return [1, action]

    def update(self, obs: Tuple[int], action, reward, terminated, next_obs: Tuple[int]):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]

        self.q_values[obs][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)