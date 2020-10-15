from matplotlib import pyplot as plt
import numpy as np
import torch

from collections import namedtuple, deque
import random
import logging
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, size, seed, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.state = np.zeros(size)
        self.theta = theta
        self.sigma = sigma

        np.random.seed(seed)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state.fill(0)

    def __call__(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (-x) + self.sigma * np.random.randn(*x.shape)
        self.state = x + dx
        return self.state


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
    solved = False

    scores = []                         # list containing scores from each episode
    scores_avg = []
    scores_window = deque(maxlen=100)   # last 100 scores
    max_score = 0

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        state = env_info.vector_observations                # get the current state
        agent.reset()
        score = np.zeros(agent.num_agents)                  # reset score for new episode

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
        scores_avg.append(np.mean(scores_window))

        if max_score < np.mean(score):
            max_score = np.mean(score)
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            checkpoint = dict()
            for i in range(len(agent.agents)):
                checkpoint[f'actor_{i}'] = agent.agents[i].actor_local.state_dict()
                checkpoint[f'actor_target_{i}'] = agent.agents[i].actor_target.state_dict()
                checkpoint[f'critic_{i}'] = agent.agents[i].critic_local.state_dict()
                checkpoint[f'critic_target{i}'] = agent.agents[i].critic_target.state_dict()
            torch.save(checkpoint, 'checkpoints/max_maddpg.pth')

        print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_window)))
            logging.info(f'Score average over last 100 episodes reached {np.mean(scores_window)} '
                         f'after {i_episode} episodes.')
        if np.mean(scores_window) >= 0.5 and not solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            logging.info(f'Environment solved in {i_episode - 100} episodes '
                         f'with average score of {np.mean(scores_window)}')
            solved = True

    return scores, scores_avg


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

