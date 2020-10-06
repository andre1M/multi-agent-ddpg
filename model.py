from torch.nn import functional as F
from torch import nn
import torch

from utilities import hidden_layer_init


class Actor(nn.Module):
    def __init__(self, observation_size, action_size, seed, fc1_units=400, fc2_units=300):
        super().__init__()

        # Random seed
        torch.manual_seed(seed)

        # Layers
        self.fc1 = nn.Linear(observation_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        # Normalization
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

        # Initialize
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and biases
        self.fc1.weight.data.uniform_(*hidden_layer_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_layer_init(self.fc2))

        # Initialize output layer weights and biases
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return torch.tanh(x)


class Critic(nn.Module):
    def __init__(self, observation_size, action_size, seed, fc1_units=400, fc2_units=300):
        super().__init__()

        # Random seed
        torch.manual_seed(seed)

        # Layers
        self.fc1 = nn.Linear(observation_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        # Normalization
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Initialize
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and biases
        self.fc1.weight.data.uniform_(*hidden_layer_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_layer_init(self.fc2))

        # Initialize output layer weights and biases
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = torch.cat((x, action), dim=1)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        x = self.dropout(x)

        return self.fc3(x)
