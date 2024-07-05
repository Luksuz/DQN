from collections import defaultdict
import gymnasium as gym

env = gym.make("")

class TicTacToeAgent:
    def __init__(self, lr, initial_epsilon, epsilon_decay, min_epsilon, discount=0.95):
        self.q_values = defaultdict(lambda: env.action_space.n)
        self.lr = lr
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.discount = discount