from matplotlib import pyplot as plt
import numpy as np
import torch

from math import sqrt
from collections import deque, namedtuple
import random
import logging


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OrnsteinUhlenbeckActionNoise:
    """
    Ornstein-Uhlenbeck random process.
    """

    def __init__(self, action_size: int, seed: float, mu=0, theta=0.15, sigma=0.2):
        np.random.seed(seed)
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_size) * self.mu
        self.reset()

    def reset(self):
        """
        Reset the internal state (= noise) to mean (mu).
        """

        self.state = np.ones(self.action_size) * self.mu

    def __call__(self, scale):
        """
        Update internal state and return it as a noise sample.
        """

        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * scale


class ReplayBuffer:
    """
    Fixed-size memory buffer to store experience tuples.
    """

    def __init__(self, buffer_size: int, batch_size: int, seed: int):
        """
        Initialize a ReplayBuffer object.

        :param buffer_size: maximum size of buffer;
        :param batch_size: size of each training batch;
        :param seed: random seed.
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.experience = namedtuple(
            'Experience',
            field_names=('state', 'action', 'reward', 'next_state', 'done')
        )

        # initialize random number generator state
        random.seed(seed)

    def __len__(self):
        """
        Return the current size of internal memory.
        """

        return len(self.memory)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.

        :param state: state description;
        :param action: action taken in state;
        :param reward: reward received;
        :param next_state: next state;
        :param done: terminal state indicator.
        """

        self.memory.append(
            self.experience(state, action, reward, next_state, done)
        )

    # noinspection PyUnresolvedReferences
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.

        :return: torch tensors of states, action, rewards, next states and terminal state flags.
        """

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(DEVICE)

        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).float().to(DEVICE)

        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(DEVICE)

        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(DEVICE)

        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(DEVICE)

        return states, actions, rewards, next_states, dones


def hard_update(source: torch.nn.Module, target: torch.nn.Module):
    """
    Copy network parameters from source to target.

    :param source: Network whose parameters to copy;
    :param target: Network to copy parameters to.
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    """
    Mix in source network parameters to the target network.

    :param source: Network whose parameters to copy;
    :param target: Network to copy parameters to;
    :param tau: soft update coefficient; determines the fraction of the source network parameters
        to be mixed in the target network parameters.
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hidden_layer_init(layer):
    fan_in = layer.weight.data.size(0)
    limit = 1 / sqrt(fan_in)
    return -limit, limit


def train(agent, env, n_episodes=2000):
    """
    Train a Reinforcement Learning agent.

    :param agent: agent object to be trained;
    :param env: environment callable;
    :param n_episodes: maximum number of training episodes;
    :return: scores per episode.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, filename='training.log', filemode='w', format='%(asctime)s - %(message)s')

    brain_name = env.brain_names[0]

    scores = []                         # list containing scores from each episode
    avg_scores = []
    scores_window = deque(maxlen=100)   # last 100 scores

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        state = env_info.vector_observations                # get the current state
        agent.reset()
        score = np.zeros(len(env_info.agents))              # reset score for new episode

        while True:
            action = agent.act(state, explore=True)                 # select action
            env_info = env.step(action)[brain_name]                 # get environment response to the action
            next_state = env_info.vector_observations               # get the next state
            reward = env_info.rewards                               # get the reward
            done = env_info.local_done                              # terminal state flag
            agent.step(state, action, reward, next_state, done)     # process experience
            state = next_state
            score += reward
            if np.any(done):
                break

        # save recent scores
        scores_window.append(np.mean(score))
        scores.append(np.mean(score))
        avg_scores.append(np.mean(scores_window))

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            logging.info(f'Score average over the last 100 episodes reached {np.round(np.mean(scores_window), 5)} '
                         f'after {i_episode} episodes.')
        if np.mean(scores_window) >= 0.5:
            torch.save(agent.actor_local.state_dict(), 'final_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'final_checkpoint_critic.pth')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            logging.info(f'Environment solved in {i_episode - 100} episodes '
                         f'with average score of {np.round(np.mean(scores_window), 5)}')
            break
    return scores, avg_scores


def plot_scores(scores, avg_scores, filename):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(np.arange(len(scores)), scores, label='Scores')
    ax.plot(np.arange(len(scores)), avg_scores, label='Average scores')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xlabel('Episode #', fontweight='bold')
    ax.set_title('Score evolution over training', fontweight='bold')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
