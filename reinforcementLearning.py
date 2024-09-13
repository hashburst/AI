import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Neural Network for Q-Learning
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the RL agent
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        target = reward + (1 - done) * self.gamma * torch.max(self.q_network(next_state)).item()
        current_q = self.q_network(state)[0, action]
        loss = self.loss_fn(current_q, torch.tensor(target))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if done:
            self.epsilon *= 0.995  # Reduce exploration over time
