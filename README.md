[//]: # (Image References)

[image1]: animations/trained_agent.gif "Trained Agent"
[image2]: animations/random_agent.gif "Random agent"


# Multi-Agent Continuous Control in Collaborative-Competitive environment

### Introduction

For this project, I am using slightly modified versions of Unity [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. See the animation with trained agent in comparison to "dumb" agent below

**Trained Agent:**

![Trained Agent][image1]

**Random Agent**

![Random Agnet][image2]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play. The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the average over both agents).

### Getting Started

***Note: This guide was only tested for macOS with `pyenv` for python versions management.***

Please follow these steps to be able to run this project:

 1. Install build tools (such as C++ compiler and etc.) by installing Xcode and then Xcode command-line tools following [one of the various guides](https://macpaw.com/how-to/install-command-line-tools) .

 2. Install dependencies. It is highly recommended to install all dependencies in virtual environment (see [guide](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Using-Virtual-Environment.md)).

    - Install Unity ML-Agents Toolkit following instruction from [this page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) (official GitHub of Unity ML-Agents Toolkit). It is very likely that most of you will only need to install `mlagents` and `unityagents`  packages with the following command:
        ```shell script
        pip install mlagents unityagents
        ```
        It is highly recommended to use Python not higher then 3.7, because `TesnsorFlow` (one of the dependency for `mlagents`) is only compatible with Python 3.7).

    - Install PyTorch with
        ```shell script
        pip insall torch torchvision
        ```
        Please see [official installation guide](https://pytorch.org/get-started/locally/#mac-installation) for more information.
        
    ***Alternatively***, it is possible to install required dependencies using `requirements.txt`. To do that jsut run the following command in your terminal (preferably in project's virtual environment):
    ```shell script
    pip install -r requirements.txt
    ```
    Please note: this method is a bit of an overkill and has some packages that are not really used in this project. In fact this is a "R&D" environment for the experiments and testing.

3. Download the environment from one of the links below. You need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
4. Place the file in the root of the project repository.

5. Run `main.py` from terminal with
    ```shell script
    python main.py
    ```
    or simply run `main.py` in your IDE.

6. Run `visualize.py` to see intelligent agent with
    ```shell script
    python vizualize.py
    ```
   or, again, simply run this file in you IDE.

### Technical report
 
Check out [technical report](REPORT.md) for implementation details.