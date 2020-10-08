from model import Actor, Critic
from utilities import hard_update, soft_update, OrnsteinUhlenbeckActionNoise, ReplayBuffer, DEVICE

from torch.nn import functional as F
from torch.optim import Adam
from torchviz import make_dot
import numpy as np
import torch

import random


# # # HYPERPARAMETERS # # #
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
DISCOUNT_FACTOR = 0.99
BUFFER_SIZE = int(1e6)
MINIBATCH_SIZE = 1024
UPDATE_EVERY = 1
NUM_UPDATES = 1
TAU = 1e-3
# # # HYPERPARAMETERS # # #


class MultiAgentDeepDeterministicPolicyGradient:
    """
    Interacts with and learns from the environment.
    Deep Deterministic Policy Gradient for multiple collaborating-competing agents.
    """

    def __init__(self, observation_size: int, action_size: int, num_agents: int, seed: int):
        """
        Initialize an Agent object.

        :param observation_size: dimension of each state;
        :param action_size: dimension of each action;
        :param num_agents: number of agents in the environment;
        :param seed: random seed.
        """

        random.seed(seed)

        self.observation_size = observation_size
        self.action_size = action_size
        self.num_agents = num_agents

        self.action_low = -1.0
        self.action_high = 1.0

        # Initialize networks and optimizers
        # # # SHARED ACTOR # # #
        self.actor_local = Actor(self.observation_size, self.action_size, seed=seed).to(DEVICE)
        self.actor_target = Actor(self.observation_size, self.action_size, seed=seed).to(DEVICE)
        hard_update(self.actor_local, self.actor_target)
        self.actor_optim = Adam(self.actor_local.parameters(), lr=ACTOR_LR)

        # # # INDIVIDUAL CRITICS # # #
        self.critics_local = [
            Critic(self.num_agents * self.observation_size, self.num_agents * self.action_size, seed=seed).to(DEVICE)
            for seed in range(self.num_agents * 2, self.num_agents * 3)
        ]
        self.critics_target = [
            Critic(self.num_agents * self.observation_size, self.num_agents * self.action_size, seed=seed).to(DEVICE)
            for seed in range(self.num_agents * 3, self.num_agents * 4)
        ]
        for i in range(self.num_agents):
            hard_update(self.critics_local[i], self.critics_target[i])

        self.critic_optims = [
            Adam(critic.parameters(), lr=CRITIC_LR) for critic in self.critics_local
        ]

        # Random process
        self.noise = OrnsteinUhlenbeckActionNoise(self.num_agents * self.action_size, seed)

        # Initialize replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, MINIBATCH_SIZE, seed)

        self.t_step = 0
        self.eps = 1
        self.eps_decay = 0.9995
        self.eps_min = 0.01

    def step(self, state, action: int, reward: float, next_state, done):
        """
        Save experiences in the replay memory and check if it's time to learn.

        :param state: (array_like) current state;
        :param action: action taken;
        :param reward: reward received;
        :param next_state: (array_like) next state;
        :param done: terminal state indicator; int or bool.
        """

        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            # Learn, if there is enough samples in memory
            if len(self.memory) > MINIBATCH_SIZE:
                for _ in range(NUM_UPDATES):
                    for agent in range(self.num_agents):
                        # sample experiences from memory
                        experiences = self.memory.sample()
                        # learn from sampled experiences
                        self.learn(agent, experiences)
                    self.soft_update()

    def act(self, state, explore=False):
        """
        Returns actions for given state as per current policy.

        :param state: (array_like) current state;
        :param explore: (bool) exploration or exploitation.
        """

        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if explore:
            actions += self.eps * np.random.normal(
                0, 1, self.action_size * self.num_agents
            ).reshape((self.num_agents, self.action_size))
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

        return np.clip(actions, self.action_low, self.action_high)

    def learn(self, agent: int, experiences):
        """
        Update value parameters using given batch of experience tuples.

        :param agent: agent index;
        :param experiences: (Tuple[torch.Tensor]) tuple of (s, a, r, s', done) tuples.
        """

        states, actions, rewards, next_states, dones = experiences
        states_full = states.view(MINIBATCH_SIZE, self.num_agents * self.observation_size)
        next_states_full = next_states.view(MINIBATCH_SIZE, self.num_agents * self.observation_size)

        # # # UPDATE CRITIC # # #
        # Collect next actions for each agent based on partial observations
        next_actions = self.actor_local(next_states)

        # Evaluate target Q_values for next (state, action) pairs
        Q_targets_next = self.critics_target[agent](
            next_states_full,
            next_actions.view(MINIBATCH_SIZE, self.num_agents * self.action_size)
        )

        Q_targets = rewards[:, agent].view(MINIBATCH_SIZE, 1) \
            + (DISCOUNT_FACTOR * Q_targets_next * (1 - dones[:, agent].view(MINIBATCH_SIZE, 1)))

        Q_expected = self.critics_local[agent](
            states_full,
            actions.view(MINIBATCH_SIZE, self.action_size * self.num_agents)
        )

        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # make graph
        # make_dot(critic_loss,
        #          params=dict(self.critics_local[agent].named_parameters())).render(filename='critic_loss', format='png')
        # make_dot(critic_loss).render(filename='critic_loss', format='png')

        self.critic_optims[agent].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics_local[agent].parameters(), 1)
        self.critic_optims[agent].step()

        # Update Actor
        # action_predictions = torch.zeros(MINIBATCH_SIZE * self.num_agents, self.action_size).to(DEVICE)
        # for i in range(self.num_agents):
        #     if i == agent:
        #         action_predictions[i::self.num_agents, :] = self.actor_local(states[i::self.num_agents])
        #     else:
        #         action_predictions[i::self.num_agents, :] = self.actor_local(states[i::self.num_agents]).detach()
        action_predictions = self.actor_local(states)

        actor_loss = -self.critics_local[agent](
            states_full,
            action_predictions.view(MINIBATCH_SIZE, self.num_agents * self.action_size)
        ).mean()

        # make graph
        # make_dot(actor_loss).render(filename='actor_loss', format='png')

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def soft_update(self):
        for agent in range(self.num_agents):
            soft_update(self.critics_local[agent], self.critics_target[agent], TAU)
        soft_update(self.actor_local, self.actor_target, TAU)

    def reset(self):
        self.noise.reset()
