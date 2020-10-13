from model import Actor, Critic
from utilities import ReplayBuffer, OrnsteinUhlenbeckActionNoise

from torch.nn import functional as F
from torch.optim import Adam
import numpy as np
import torch

import random


# # # HYPERPARAMETERS # # #
BUFFER_SIZE     = int(1e6)      # Replay memory buffer size
BATCH_SIZE      = 512           # Number of experience tuples to be sampled for learning
DISCOUNT_FACTOR = 0.98          # Discounting factor for rewards
SOFT_UPDATE     = 1e-2          # Soft update ratio for target network
UPDATE_EVERY    = 1             # Parameters update frequency
NUM_UPDATES     = 1             # Number of updates per step
ACTOR_LR        = 1e-5          # Actor learn rate
CRITIC_LR       = 1e-5          # Critic learn rate
CRITIC_WD       = 0             # Critic weight decay
EPS_START       = 1
EPS_DECAY       = 0.99995
EPS_MIN         = 0.01
# # # HYPERPARAMETERS # # #

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiAgent:
    """
    Multi-Agent Deep Deterministic Policy Gradient algorithm.
    """

    def __init__(self, observation_dim: int, action_dim: int, num_agents: int, seed: int):
        """
        :param observation_dim: observation dimension per agent;
        :param action_dim: action dimension per agent;
        :param num_agents: number of agents;
        :param seed: random seed.
        """

        random.seed(seed)

        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.state_dim = num_agents * observation_dim
        self.full_action_dim = num_agents * action_dim

        # Initialize shared experience buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize agents
        self.agents = [
            DeepDeterministicPolicyGradient(observation_dim, action_dim, num_agents, idx, seed)
            for idx in range(num_agents)
        ]

        # Update step counter
        self.t_step = 0

    def target_actions(self, observations):
        actions = [0] * self.num_agents
        for i, agent in enumerate(self.agents):
            actions[i] = agent.actor_target(observations[i::self.num_agents])
        return torch.cat(actions, dim=1)

    def local_actions(self, idx, observations):
        actions = [0] * self.num_agents
        for i, agent in enumerate(self.agents):
            if idx == i:
                actions[i] = agent.actor_local(observations[i::self.num_agents])
            else:
                actions[i] = agent.actor_local(observations[i::self.num_agents]).detach()
        return torch.cat(actions, dim=1)

    def act(self, observations, explore: bool = False):
        """
        Take an action for the given observation as per current policy per agent.

        :param observations: (array_like) current observation per agent;
        :param explore: explore or exploit flag.
        """

        actions = np.zeros((self.num_agents, self.action_dim))
        for i, agent in enumerate(self.agents):
            actions[i] = agent.act(observations[i], explore)
        return actions

    def step(self, observations, actions, rewards, next_observations, dones):
        """
        Save experiences in the replay memory and learn if possible.

        :param observations: (array_like) current observation per agent;
        :param actions: (array_like) taken action per agent;
        :param rewards: (array_like) received reward per agent;
        :param next_observations: (array_like) next observation per agent;
        :param dones: (array_like) termination state flag per agent;
        """

        self.memory.push(observations, actions, rewards, next_observations, dones)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if len(self.memory) >= BATCH_SIZE and self.t_step == 0:
            for _ in range(NUM_UPDATES):
                self.learn()

    def learn(self):
        """
        Update model parameters with experiences sampled from memory.
        """

        for idx, agent in enumerate(self.agents):
            observations, actions, rewards, next_observations, dones = self.memory.sample()
            states = observations.reshape(BATCH_SIZE, self.state_dim)
            actions = actions.reshape(BATCH_SIZE, self.full_action_dim)
            rewards = rewards[:, idx].reshape(BATCH_SIZE, 1)
            next_states = next_observations.reshape(BATCH_SIZE, self.state_dim)
            dones = dones[:, idx].reshape(BATCH_SIZE, 1)
            next_actions = self.target_actions(next_observations)
            agent.update_critic(states, actions, rewards, dones, next_states, next_actions)
        # for idx, agent in enumerate(self.agents):
        #     observations, actions, rewards, next_observations, dones = self.memory.sample()
        #     states = observations.reshape(BATCH_SIZE, self.state_dim)
            action_predictions = self.local_actions(idx, observations)
            agent.update_actor(states, action_predictions)
            agent.soft_update()

    def reset(self):
        for agent in self.agents:
            agent.reset()


class DeepDeterministicPolicyGradient:
    """
    Interacts with and learns from the environment.
    Deep Deterministic Policy Gradient algorithm.
    """

    def __init__(self, observation_dim: int, action_dim: int, num_agents: int, idx: int, seed: int):
        """
        Initialize an Agent object.

        :param observation_dim: observation dimension per agent;
        :param action_dim: action dimension per agent;
        :param num_agents: number of agents;
        :param idx: agent's index;
        :param seed: random seed.
        """

        random.seed(seed)

        self.idx = idx
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.state_dim = num_agents * observation_dim
        self.full_action_dim = num_agents * action_dim

        self.eps = EPS_START

        # Initialize networks and optimizers
        self.actor_local = Actor(self.observation_dim, self.action_dim, seed=seed).to(DEVICE)
        self.actor_target = Actor(self.observation_dim, self.action_dim, seed=seed).to(DEVICE)
        self.hard_update(self.actor_local, self.actor_target)
        self.actor_optim = Adam(self.actor_local.parameters(), lr=ACTOR_LR)

        self.critic_local = Critic(self.state_dim, self.full_action_dim, seed=seed).to(DEVICE)
        self.critic_target = Critic(self.state_dim, self.full_action_dim, seed=seed).to(DEVICE)
        self.hard_update(self.critic_local, self.critic_target)
        self.critic_optim = Adam(self.critic_local.parameters(), lr=CRITIC_LR, weight_decay=CRITIC_WD)

        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim, seed, theta=0.15, sigma=0.2)

    def act(self, observation, explore=True):
        """
        Returns actions for given state as per current policy.

        :param observation: (array_like) current observation;
        :param explore: (bool) explore or exploit flag.
        """

        observation = torch.from_numpy(observation).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(observation.unsqueeze(0)).cpu().data.numpy()
        self.actor_local.train()

        # Add noise for exploration
        if explore:
            action += self.eps * self.noise()
            self.eps = max(EPS_MIN, self.eps * EPS_DECAY)

        return np.clip(action, -1, 1)

    def update_critic(self, states, actions, rewards, dones, next_states, next_actions):
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (DISCOUNT_FACTOR * Q_targets_next * (1 - dones)).detach()
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()

    def update_actor(self, states, action_predictions):
        # Update Actor
        actor_loss = -self.critic_local(states, action_predictions).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def soft_update(self):
        self._soft_update(self.critic_local, self.critic_target, SOFT_UPDATE)
        self._soft_update(self.actor_local, self.actor_target, SOFT_UPDATE)

    # def learn(self, states, actions, rewards, next_states, dones, next_actions, action_predictions):
    #     # Update Critic
    #     Q_targets_next = self.critic_target(next_states, next_actions)
    #     Q_targets = rewards + (DISCOUNT_FACTOR * Q_targets_next * (1 - dones)).detach()
    #     Q_expected = self.critic_local(states, actions)
    #     critic_loss = F.mse_loss(Q_expected, Q_targets)
    #     self.critic_optim.zero_grad()
    #     critic_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
    #     self.critic_optim.step()
    #
    #     # Update Actor
    #     actor_loss = -self.critic_local(states, action_predictions).mean()
    #     self.actor_optim.zero_grad()
    #     actor_loss.backward()
    #     self.actor_optim.step()
    #
    #     # Target network soft update
    #     self.soft_update(self.critic_local, self.critic_target, SOFT_UPDATE)
    #     self.soft_update(self.actor_local, self.actor_target, SOFT_UPDATE)

    def reset(self):
        self.noise.reset()

    def make_checkpoint(self):
        torch.save(self.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(self.critic_local.state_dict(), 'checkpoint_critic.pth')

    @staticmethod
    def _soft_update(local_model, target_model, tau: float):
        """
        Soft update model parameters:
        θ_target = τ * θ_local + (1 - τ) * θ_target.

        :param local_model: (PyTorch model) weights will be copied from;
        :param target_model: (PyTorch model) weights will be copied to;
        :param tau: interpolation parameter.
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def hard_update(local_model, target_model):
        """
        Hard update model parameters.

        :param local_model: (PyTorch model) weights will be copied from;
        :param target_model: (PyTorch model) weights will be copied to;
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
