from agent import MultiAgent

from unityagents import UnityEnvironment
import numpy as np
import torch


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
agent = MultiAgent(state_size, action_size, len(env_info.agents), seed=0)

parameters = torch.load('checkpoints/maddpg.pth')

for i in range(len(env_info.agents)):
    agent.agents[i].actor_local.load_state_dict(parameters[f'actor_{i}'])

for i in range(3):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations
    while True:
        action = agent.act(state, explore=False)
        env_info = env.step(action)[brain_name]
        state = env_info.vector_observations
        done = env_info.local_done
        if np.any(done):
            break

# close environment
env.close()
