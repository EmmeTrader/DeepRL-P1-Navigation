#### Udacity Deep Reinforcement Learning Nanodegree
### Project 1: Navigation
# Training a RL Agent to collect yellow bananas, while avoiding blue ones

## Table of Contents  

[Environment details](#Environment details)

[Algorithm](#First)

[Hyperparameters](#Hyperparameters)

[Rewards](#Rewards)

[Future Work](#Future_Work)  

## Environment details

The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents)

Note: The project environment provided by Udacity is similar to, but not identical to the Banana Collector environment on the Unity ML-Agents GitHub page.

> The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. Agents can be trained using reinforcement learning, imitation learning, neuroevolution, or other machine learning methods through a simple-to-use Python API. 

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. 

Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and **in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.**

<a name="First"></a>
## Algorithm

The deployed solution implements a *Value Based* method called [Deep Q-Networks](https://deepmind.com/research/dqn/). 

**Q-Learning** is an algorithm that allows RL agents to learn a policy to decide which actions to take within an environment. The goal is to find an optimal policy, which means a policy that maximizes the rewards collected by the agent. The value of each possible actions in terms of rewards is not known in advance and an optimal policy must be learned through a process of trial and error by interacting with the environment and recording observations, a process that iteratively maps different environment states to the actions that yield the highest reward.

**Deep Q Learning** is a combination of two approaches :
- A Reinforcement Learning method called [Q Learning](https://en.wikipedia.org/wiki/Q-learning) 
- A Deep Neural Network to learn and approximate an optimal Q-table (action-values map)

Inspired by the work of [Deepmind](https://deepmind.com), as described in their [Nature publication : "Human-level control through deep reinforcement learning (2015)"](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), two improvements have been implemented:
- Experience Replay 
- Fixed Q Targets

Experience replay improves learning through repetition and past experience. By doing multiple passes over the data, the RL agent has multiple opportunities to learn from a single experience tuple. Each experience is stored in a replay buffer as the agent interacts with the environment. The replay buffer contains a collection of experience tuples with the state, action, reward, and next state `(s, a, r, s')`. Sampling from this buffer is part of the agent's learning step. Experiences are sampled randomly, so that the data correlation is potentially reduced and avoided. This allows action values to not oscillate or diverge dramatically, otherwise a basic Q-learning agent could become biased by correlations between sequential experience tuples.

The Fixed Q Targets solution was introduced by the DeepMind team. Using two DQNs instead of one, this method keeps the target values of one network (called the target network) fixed and periodically updates the network weights. 

**Code implementation**

The code implementation is derived from the "Lunar Lander" example from the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), and has been slightly adjusted for being used with the Unity banana environment instead of the original OpenAI environment.

The **model.py** file contains the definition of a QNetwork class, a fully connected deep neural network based in the [PyTorch Framework](https://pytorch.org/docs/0.4.0/).
The goal of this deep NN is to predict the best action to perform based on the environment observerd states. 
Here is the network architecture:
```
QNetwork(
  (fc1): Linear(in_features=37, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (out): Linear(in_features=64, out_features=4, bias=True)
)
```
The inputs are the 37 state variables from the Unity environment and the final output is 4 like the action space size.

The **dqn_agent.py** file contains the DQN class Agent and the class ReplayBuffer.

The DQN class Agent contains 5 methods:
- constructor, which initializes the memory buffer and two instances of the deep NN (target and local)
- step(), which allows to store a step taken by the RL agent (state, action, reward, next_state, done) in the Replay Buffer/Memory. Every four steps, it updates the target NN weights  with the current weight values from the local NN (Fixed Q Targets technique)
- act(), which returns actions for given state as per current policy through an Epsilon-greedy selection in order to balance exploration and exploitation in the Q Learning process
- learn(), which updates value parameters using given batch of experience tuples in the form of (state, action, reward, next_state, done) 
- soft_update(), which is used by the learn() method to softly updates the target NN weights from the local NN weights

The ReplayBuffer class consists of Fixed-size buffer to store experience tuples (state, action, reward, next_state, done)  and contains these methods:
- add(), which adds a new experience tuple to memory
- sample(), which randomly sample a batch of experiences from memory for the learning process of the agent
- len(), which returns the current size of internal memory

      
The Navigation.ipynb: This Jupyter notebooks allows to train the agent. More in details it allows to :
  - Import the Necessary Packages 
  - Examine the State and Action Spaces
  - Take Random Actions in the Environment (No display)
  - Train an agent using DQN
  - Plot the scores

### DQN parameters and results

The DQN agent uses the following parameters values (defined in dqn_agent.py)

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size 
GAMMA = 0.995           # discount factor 
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
```

The Neural Networks use the following architecture :

```
Input nodes (37) -> Fully Connected Layer (1024 nodes, Relu activation) -> Fully Connected Layer (1024 nodes, Relu activation) -> Ouput nodes (4)
```

The Neural Networks use the Adam optimizer with a learning rate LR=5e-4 and are trained using a BATCH_SIZE=64

Given the chosen architecture and parameters, our results are :

<a name="Hyperparameters"></a>
## Hyperparameters:
* Buffer Size = 100000
* Batch Size = 64
* Discount Factor (GAMMA)  = 0.999
* TAU (for soft update of target parameters) = 0.001
* Learning Rate = 0.0005
* Update Network every 4 step

<a name="Rewards"></a>
## Rewards
We have trained the Agent using the DQN, as input we have used the vector of state instead of an image so convolutional neural network is replaced with the next layers:

Fully connected layer - input: 37 (state size) output: 128
Fully connected layer - input: 128 output 64
Fully connected layer - input: 64 output: (action size)

We can observe the evolution of the reward along of the episodes. Below is the graphic:

![Rewards](./images/reward_762.png)

<a name="Future_Work"></a>
## Future Work
Reinforcement Learning has proven its worth in approximating real-world environments. It can help solve many problems that and with the combination of deep learning and RL, we’re much closer to solving these problems. We have thought in the next steps to improve our algorithm:

  1. Try with another hyperparameters

  2. Train the double deep Q Network

  3. Train the dueling deep Q Network

  4. Develop an algorithm to learn from pixels

Future applications of reinforcement learning include some of the following tasks:

* A Distributional Perspective on Reinforcement Learning [arxiv](https://arxiv.org/pdf/1707.06887.pdf)
* Rainbow: Combining Improvements in Deep Reinforcement Learning [arxiv](https://arxiv.org/abs/1710.02298)
* Hierarchical Deep Reinforcement Learning [arxiv](https://arxiv.org/abs/1604.06057)
