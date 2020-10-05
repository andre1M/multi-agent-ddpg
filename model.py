from torch.nn import functional as F
from torch import nn
import torch

from utilities import hidden_layer_init


class Actor(nn.Module):
    def __init__(self, observation_size, action_size, seed, fc1_units=256):
        super().__init__()

        # Random seed
        torch.manual_seed(seed)

        # Layers
        self.fc1 = nn.Linear(observation_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

        # Initialize
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and biases
        self.fc1.weight.data.uniform_(*hidden_layer_init(self.fc1))

        # Initialize output layer weights and biases
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(self.bn1(x))
        return torch.tanh(self.fc2(x))


class Critic(nn.Module):
    def __init__(self, observation_size, action_size, seed, fc1_units=256, fc2_units=256, fc3_units=128):
        super().__init__()

        # Random seed
        torch.manual_seed(seed)

        # Layers
        self.fc1 = nn.Linear(observation_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)

        self.dropout = nn.Dropout(p=0.5)

        # Initialize
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and biases
        self.fc1.weight.data.uniform_(*hidden_layer_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_layer_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_layer_init(self.fc3))

        # Initialize output layer weights and biases
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = F.leaky_relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        x = F.leaky_relu(self.bn2(x))
        x = F.leaky_relu(self.fc3(x))
        self.dropout(x)
        return self.fc4(x)
