from model import Actor, Critic
from utilities import hard_update, soft_update, OrnsteinUhlenbeckActionNoise, ReplayBuffer, DEVICE

from torch.optim import Adam
from torch.nn import functional as F
import numpy as np
import torch

import random


# # # HYPERPARAMETERS # # #
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
CRITIC_WD = 0
DISCOUNT_FACTOR = 0.95
BUFFER_SIZE = int(1e6)
MINIBATCH_SIZE = 32
UPDATE_EVERY = 1
TAU = 1e-2
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

        self.observation_size = observation_size
        self.action_size = action_size
        self.action_low = -1
        self.action_high = 1
        self.num_agents = num_agents
        random.seed(seed)

        # Initialize networks and optimizers
        # self.actors_local = [
        #     Actor(self.observation_size, self.action_size, seed=seed).to(DEVICE) for seed in range(self.num_agents)
        # ]
        # self.actors_target = [
        #     Actor(self.observation_size, self.action_size, seed=seed).to(DEVICE) for seed in range(self.num_agents)
        # ]

        self.actor_local = Actor(self.observation_size, self.action_size, seed=seed).to(DEVICE)
        self.actor_target = Actor(self.observation_size, self.action_size, seed=seed).to(DEVICE)
        hard_update(self.actor_local, self.actor_target)
        self.actor_optim = Adam(self.actor_local.parameters(), lr=ACTOR_LR)

        self.critic_local = Critic(self.num_agents * self.observation_size,
                                   self.num_agents * self.action_size,
                                   seed=seed).to(DEVICE)
        self.critic_target = Critic(self.num_agents * self.observation_size,
                                    self.num_agents * self.action_size,
                                    seed=seed).to(DEVICE)
        hard_update(self.critic_local, self.critic_target)
        self.critic_optim = Adam(self.critic_local.parameters(), lr=ACTOR_LR, weight_decay=CRITIC_WD)

        self.noise = OrnsteinUhlenbeckActionNoise(action_size, seed)

        # Initialize replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, MINIBATCH_SIZE, seed)

        self.t_step = 0
        # self.eps_t = 0
        # self.no_decay_steps = 0
        self.eps = 1
        self.eps_decay = 0.995
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
                # sample experiences from memory
                experiences = self.memory.sample()
                # learn from sampled experiences
                self.learn(experiences)

    def act(self, state, explore=False):
        """
        Returns actions for given state as per current policy.

        :param state: (array_like) current state;
        :param explore: (bool) exploration or exploitation.
        """
        # self.eps_t += 1

        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if explore:
            for action in actions:
                action += self.noise(self.eps)
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

        return np.clip(actions, self.action_low, self.action_high)

    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.

        :param experiences: (Tuple[torch.Tensor]) tuple of (s, a, r, s', done) tuples;
        """

        states, actions, rewards, next_states, dones = experiences
        states_full = states.view(MINIBATCH_SIZE, self.num_agents * self.observation_size)
        next_states_full = next_states.view(MINIBATCH_SIZE, self.num_agents * self.observation_size)

        # Update Critic
        next_actions = self.actor_target(next_states)
        # next_actions = torch.reshape(next_actions, (MINIBATCH_SIZE, self.num_agents * self.action_size))

        Q_targets_next = self.critic_target(
            next_states_full,
            next_actions.view(MINIBATCH_SIZE, self.action_size * self.num_agents)
        )

        # Q_targets_next = self.critic_target(
        #     next_states_full.repeat(1, 2).view(MINIBATCH_SIZE * self.num_agents, next_states_full.shape[1]),
        #     next_actions
        # )

        # Q_targets_next = self.critic_target(
        #     torch.reshape(next_states, (MINIBATCH_SIZE, self.num_agents * self.observation_size)), next_actions
        # )

        Q_targets = rewards.mean(-1).view(MINIBATCH_SIZE, 1) \
                    + (DISCOUNT_FACTOR * Q_targets_next * (1 - dones.byte().any(-1).view(MINIBATCH_SIZE, 1)))

        # Q_targets = rewards.view(MINIBATCH_SIZE * self.num_agents, 1) \
        #             + (DISCOUNT_FACTOR * Q_targets_next
        #             * (1 - dones.view(MINIBATCH_SIZE * self.num_agents, 1)))

        # Q_expected = self.critic_local(
        #     states_full.repeat(1, 2).view(MINIBATCH_SIZE * self.num_agents, states_full.shape[1]), actions
        # )

        Q_expected = self.critic_local(
            states_full,
            actions.view(MINIBATCH_SIZE, self.action_size * self.num_agents)
        )

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Update Actor

        action_predictions = self.actor_local(states)
        # actor_loss = -self.critic_local(
        #     states_full.repeat(1, 2).view(MINIBATCH_SIZE * self.num_agents, states_full.shape[1]),
        #     action_predictions
        # ).mean()

        actor_loss = -self.critic_local(
            states_full,
            action_predictions.view(MINIBATCH_SIZE, self.action_size * self.num_agents)
        ).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Target network soft update
        soft_update(self.critic_local, self.critic_target, TAU)
        soft_update(self.actor_local, self.actor_target, TAU)

    def reset(self):
        self.noise.reset()

    def make_checkpoint(self):
        torch.save(self.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(self.critic_local.state_dict(), 'checkpoint_critic.pth')
