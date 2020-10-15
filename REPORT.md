[//]: # (Image References)

[image1]: figures/scheme.png
[image2]: figures/algorithm.png
[image3]: figures/score_maddpg.png
[image4]: figures/graph.png

# Continuous control with Deep Deterministic Policy Gradient (DDPG)

In this project I solved a control task in continuous action space with two agents present. Similarly to my previous [project](https://github.com/andre1M/continuous-control) we deal with continuous action space here. Added complexity is motivated by the introduction of the second agent. This means the environment is no more stationary, since its state can be changed in the perspective of one agent when other agents take actions. To solve this problem I implemented Multi-Agent Deep Deterministic Policy Gradient algorithm presented in [this work](https://arxiv.org/abs/1706.02275) with several modifications and parameters tuning to speed up learning for this particular environment.

## Algorithm

MADDPG algorithm from [original work](https://arxiv.org/abs/1706.02275):

![MADDPG algorithm][image2]

The main idea that makes a significant difference and allows for stable learning in multi-agent environment is centralised learning and decentralized execution. This means each agent is taking an action using its actor at the execution time. However, during training all critics take as an input the observations of all agents and all actions to learn the action value, and then actor is trained with updated critic, as shown in the figure below:

![MADDPG scheme][image1] 

## Implementation

Similarly to DDPG project ([graph from previous project](https://github.com/andre1M/continuous-control/blob/master/REPORT.md#implementation)) I use the same algorithm for every agent. Figure below depicts the architecture of the Actor and Critic of a single agent.

![graph][image4]

### Hyperparameters

A big part of this project was tuning the hyperparameters. Values were chosen according to [original work](https://arxiv.org/abs/1509.02971) with further adjustments based on observed performance. Used values are given in the table below:

| Hyperparameter            	| Value   	|
|---------------------------	|---------	|
| Replay buffer size        	| 1e6     	|
| Minibatch size            	| 512     	|
| Discount factor           	| 0.98    	|
| Actor Learning Rate       	| 1e-5    	|
| Critic Learning Rate      	| 1e-4    	|
| Epsilon initial value     	| 1       	|
| Epsilon minimal value     	| 0.01    	|
| Epsilon exponential decay 	| 0.99995 	|
| Update frequency           	| 1       	|
| Soft update coefficient   	| 5e-3    	|

Ornstein - Uhlenbeck Random Process for exploration is initiated with &theta; = 0.15 amd &sigma; = 0.2. 

## Results

Agents' score evolution during training is given in the figure below. It took the agents 2244 episodes to solve the task (to reach an average reward of 0.5 or higher over the last 100 episodes). See [REEDME](README.md) for trained agent visualization.

![Score][image3]

## Ideas for future work

There are several improvements that can be done to this project:
- Environments with more then two agents should be tested, such as [Soccer](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos);
- Study the reasons behind an unstable learning process; Implement batch normalization;
- Implement advantage function to avoid redundant moves.