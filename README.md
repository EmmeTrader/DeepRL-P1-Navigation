[//]: # (Image References)

[image1]: https://github.com/EmmeTrader/DeepRL-P1-Navigation/blob/main/images/banana.gif "banana"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent](./images/trained.gif)

Below is the evolution of the reward function along the episodes.

![Rewards](./images/reward_762.png)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Create an environment with python3.6

`python3.6 -m venv <env_name>`

2. Clone the repository to your local machine

3. Unity has to be installed on your system. Then, run the next command to install python dependencies:

`source ./install.sh`

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!
