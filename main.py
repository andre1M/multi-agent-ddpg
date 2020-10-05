from utilities import train, plot_scores
from agent import MultiAgentDeepDeterministicPolicyGradient

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

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

# Initialize agent
agent = MultiAgentDeepDeterministicPolicyGradient(state_size, action_size, len(env_info.agents), seed=0)

# train with linear epsilon decrease
scores, avg_scores = train(agent, env, n_episodes=5000)

if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

# save network weights
checkpoint = dict()
for i in range(len(env_info.agents)):
    checkpoint[f'actor_{i}'] = agent.actors_local.state_dict()
    checkpoint[f'critic_{i}'] = agent.critics_local.state_dict()

torch.save(checkpoint, 'checkpoints/maddpg.pth')

if not os.path.exists('figures'):
    os.mkdir('figures')

# plot the scores
plot_scores(scores, avg_scores, filename='figures/score_ddpg.png')

# close environment
env.close()
