from utilities import train, plot_scores
from agent import MultiAgent

from unityagents import UnityEnvironment
import torch

import os


# initialize environment
env = UnityEnvironment(file_name="Tennis.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size

# examine the state space
state = env_info.vector_observations[0]
state_size = len(state)


# Initialize agent
agent = MultiAgent(state_size, action_size, len(env_info.agents), seed=256)

# train with linear epsilon decrease
scores, avg_scores = train(agent, env, n_episodes=20000)

# plot the scores
if not os.path.exists('figures'):
    os.mkdir('figures')
plot_scores(scores, avg_scores, filename='figures/score_maddpg.png')

# save network weights
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')
checkpoint = dict()
for i in range(len(env_info.agents)):
    checkpoint[f'actor_{i}'] = agent.agents[i].actor_local.state_dict()
    checkpoint[f'actor_target_{i}'] = agent.agents[i].actor_target.state_dict()
    checkpoint[f'critic_{i}'] = agent.agents[i].critic_local.state_dict()
    checkpoint[f'critic_target{i}'] = agent.agents[i].critic_target.state_dict()
torch.save(checkpoint, 'checkpoints/maddpg.pth')

# close environment
env.close()
