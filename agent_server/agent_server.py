from flask import Flask, request
import pickle
import numpy as np
from typing import Tuple

app = Flask(__name__)

class Agent:
    def __init__(self, lr, initial_epsilon, epsilon_decay, min_epsilon, q_values, discount=0.95):
        self.q_values = q_values
        self.lr = lr
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.discount_factor = discount

        self.training_error = []

    def get_action(self, obs):
        available_indices = [i for i, x in enumerate(obs) if x == 0]
        if np.random.random() < self.epsilon:
            return [1, np.random.choice(available_indices)]
        else:
            obs_tuple = tuple(obs)  # Ensure obs is a tuple for dictionary key
            if obs_tuple in self.q_values:
                q_values = self.q_values[obs_tuple]
                valid_actions = [(i, q_values[i]) for i in available_indices]
                action = max(valid_actions, key=lambda x: x[1])[0]
                return [1, action]
            else:
                print("random")
                return [1, np.random.choice(available_indices)]

    def update(self, obs: Tuple[int], action, reward, terminated, next_obs: Tuple[int]):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]

        self.q_values[obs][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)


with open("q_table.pkl", "rb") as f:
    q_values = pickle.load(f)
wins = 0

agent = Agent(lr=0.1, initial_epsilon=0, epsilon_decay=5e-6, min_epsilon=0.01, q_values=q_values)

@app.route("/make-move", methods=["POST"])
def make_move():
    data = request.get_json()
    obs = data["obs"]
    _, action = agent.get_action(obs)
    return str(action)

if __name__ == "__main__":
    app.run(debug=True, port=5000)