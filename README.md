# Tennis
## Introduction
This is a solution for the third project of the [Udacity deep reinforcement learning course](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). 
It includes scripts for training on both agents with a single SAC (Soft Actor Critique) algorithm and running it in a simulation environment.
The models were trained using the [Stable Baselines3 project](https://stable-baselines3.readthedocs.io/en/master/#).

## Example agents
The giff shows an example run with a model trained in this repo. The agent parameters can be found under `experiments/sac_zoo/eval/best_model.zip`.
![Agent test run](artifacts/screencast_tennis.gif)

## Problem description
The environment consists of two agents who control a racket and their goal is to pass a ball over the net of a virtual tennis court.

- Rewards:
  - +0.1 for each time an agent hits the ball over the net and for the agent who performed the action
  - -0.01 whenever an agent hit the ball out of bound, again for the agent who performed the action
- Input state:
  - 24 continuous variables corresponding to position and velocity of the ball and racket. Each agent has its own observations, therefor the total space has dimension 24
  Actions:
  - 2 continuous variables, corresponding to movement towards/away from the net and jumping. Once more, 2 per agent, thus an action space with 4 dimensions in total.
- Goal:
  - Reach a maximum score of +0.5 over 100 consecutive episodes. The maximum is taken over the two agents for each episode.

## Solution
The problem is solved with SAC using the [stable baselines framework](https://stable-baselines3.readthedocs.io/en/master/). In order to speed up development, the hyper parameters for the `HalfCheetahBulletEnv-v0` environment from the [hyper-parameter zoo](https://github.com/DLR-RM/rl-baselines3-zoo).
 This hint was pointed out by [Antonin Raffin](https://github.com/araffin) as an improvement for the previous project with the [reacher environment](https://github.com/koulakis/reacher-deep-reinforcement-learning). 
 For more details about the implementation look in the [corresponding report](https://github.com/koulakis/tennis-deep-reinforcement-learning/blob/master/Report.ipynb). 

## Setup project
To setup the project follow those steps:
- Provide an environment with `python 3.6.x` installed, ideally create a new one with e.g. pyenv or conda
- Clone and install the project: 
```
git clone git@github.com:koulakis/tennis-deep-reinforcement-learning.git
cd tennis-reinforcement-learning
pip install .
```
- Create a directory called `udacity_tennis_environment` under the root of the project and download and extract there the environment compatible with your architecture. 
You can find the [download links here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).
- Install a version of pytorch compatible with your architecture. The version used to develop the project was 1.5.0.
e.g. `pip install pytorch==1.5.0`

To check that everything is setup properly, run the following test which loads an environment and runs a random agent:

`python scripts/run_agent.py --random-agent`

## Training and testing the agent
The project comes along with some pre-trained agents, scripts to test them and train your own.

### Scripts
- `train_agent.py`: This one is used to train an agent. The parameter `experiment-name` is used to name your agent and
    the script will create by default a directory under `experiments` with the same name. The trained agent parameters
    will be saved there in the end of the training and during training several metrics are logged to a  tfevents file
    under the same directory. Here is an example call:
    ```python scripts/train_agent.py --experiment-name my_tennis_agent --total-timesteps 500000 --port 5005```
    
    To monitor the metrics one can launch a tensorboard server with:
    ```tensorboard --logdir experiments```
    This will read the metrics of all experiments and make the available under `localhost:6006`
    
    One can run multiple trainings in parallel by using different ports per environment with the `port` flag.
    
- `test_agent_in_environment`: This script can be used to test an agent on a given environment. As mentioned above, one
can access the saved agent models inside the sub-folders of `experiments`. An example usage:
    ```python scripts/run_agent.py --agent-parameters-path experiments/sac_zoo/eval/best_model.zip --port 5007```
    
### Pre-trained models
Under the `experiments` directory there are 3 pre-trained models one can used to run in the environment. A model which
solves the environment is `sac_zoo`.

## References
Given that this project is an assignment of an online course, it has been influenced heavily by code provided by
Udacity and several mainstream publications. Below you can find some links which can give some broader context.

### Frameworks & codebases
1. The SAC algorithm used was trained using the [Stable Baselines3 project](https://stable-baselines3.readthedocs.io/en/master/#)
1. The initial hyper-parameters used to setup SAC come from the [hyepr parameters zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
1. Most of the simulation setup comes from [this notebook](https://github.com/udacity/deep-reinforcement-learning/blob/master/p3_collab-compet/Tennis.ipynb)
1. The unity environment created by Udacity is a direct copy [from here](https://github.com/udacity/deep-reinforcement-learning/tree/master/python)
 
### Publications
The following publications were used:

1. *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine. arXiv:1602.01783. 2016.
1. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel. arXiv:1506.02438. 2015.
1. *Continuous control with deep reinforcement learning*. Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra. arXiv:1801.01290. 2018.