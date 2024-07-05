
import torch
from torch.nn import Module, Linear, MSELoss
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np


class DQNN(Module):
    def __init__(self, lr, input_dims, fc1_neurons, fc2_neurons, epsilon, min_epsilon, n_actions):
        super(DQNN, self).__init__()
        self.input_dims = input_dims
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.n_action = n_actions

        self.fc1 = Linear(input_dims, fc1_neurons)
        self.fc2 = Linear(fc1_neurons, fc2_neurons)
        self.fc3 = Linear(fc2_neurons, n_actions)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss_fn = MSELoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)

        return action
    

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100_000, min_eps=0.01, eps_decay=5e-5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_action = n_actions
        self.min_eps = min_eps
        self.eps_decay = eps_decay

        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DQNN(self.lr, input_dims=input_dims, n_actions=n_actions, fc1_neurons=256, fc2_neurons=256, epsilon=epsilon, min_epsilon=min_eps)

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size), dtype=np.int32)
        self.reward_memory = np.zeros((self.mem_size), dtype=np.float32)
        self.terminal_memory = np.zeros((self.mem_size), dtype=np.bool_)


    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, obesrvation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([obesrvation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return 
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_size, self.mem_cntr)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = torch.tensor(self.action_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss_fn(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_decay if self.epsilon > self.min_eps else self.min_eps

        

    

