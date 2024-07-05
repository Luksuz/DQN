import gymnasium as gym
from lunarlanderDQN.DQNN import Agent
import numpy as np

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, min_eps=0.01, input_dims=8, lr=0.003)
    scores, eps_history = [], []
    n_games = 700

    for i in range(n_games):
        score = 0
        done = False
        obs, info = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, _, info = env.step(action)
            score += reward
            agent.store_transition(obs, action, reward, obs_, done)
            agent.learn()
            obs = obs_

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])  # Corrected to consider the last 100 episodes

        print(f"Episode: {i + 1}\nScore: {score}\nAverage Score (Last 100 Episodes): {avg_score}\nEpsilon: {agent.epsilon}")

    # Optional: save the results for further analysis
    np.save('scores.npy', scores)
    np.save('eps_history.npy', eps_history)