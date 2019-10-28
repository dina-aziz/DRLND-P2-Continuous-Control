[image1]: training.png "training"
[image2]: average_scores_plot.png "plot"
# Project 2: Continuous Control
--    
    
## **Introduction**

#### The environment:      
For this project, the agent is trained to interact with a [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.
In this environment, a double-jointed arm is trained to move to target locations.

#### The state space:    
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.
#### The action space:    
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.  
#### The reward:       
 A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.    
#### The environment solution:    
In order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

-- 

## **The Learning Algorithm**

The learning algorithm used here is the Deep Deterministic Policy Gradients (**DDPG**) algorithm. The implementation is based on the code provided by Udacity [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) to solve the [Pendulum](https://gym.openai.com/envs/Pendulum-v0/) Swingup Problem. 

Detailed description of the DDPG can be found [here](https://arxiv.org/abs/1509.02971) and [here](https://spinningup.openai.com/en/latest/algorithms/ddpg.html).
### Implementation notes:    
- The agent uses four Vanilla networks as local/target actor(a deterministic policy network) and local/target critic(a Q-network) networks, each consists of 2 hidden layers of sizes 400 and 300 units respectively. 

- Batch normalization and gradient clipping techniques are used when training the actor and critic networks. 

- A replay buffer is used to let the agent remember and reuse experiences from the past. Here, a fixed length buffer of length 10^5 is used to store a finite number of experiences tuples
<s<sub>t</sub>,a<sub>t</sub>,r<sub>t+1</sub>,s<sub>t+1</sub>>.
    
- A minibatch of size 1024 is sampled and used to train the agent. The samples of the minibatch are drawn from a uniform distribution in order to break any temporal/chronological relationship between them.

- Since the action space is continuous, exploring new actions occurs through adding Ornstein–Uhlenbeck Noise of parameters(mu = 0, theta = 0.15, sigma = 0.1) which decays overtime by 10-6 every time the learning step is performed.

- The learning step takes place every 20 iterations where the buffered states are sampled and used to train the networks for 10 times. 
   


###A list if the hyperparameters used for training is provided below: 
   
- The replay buffer size: BUFFER_SIZE = int(1e5)    
- The minibatch size: BATCH_SIZE = 1024
- The discount factor: GAMMA = 0.99 
- The target soft update parameter: TAU = 1e-3 
- The number of training episodes: n_episodes = 700    
#### Model Parameters:                
- The actor learning rate: LR_ACTOR = 5e-4         
- The actor first hidden layer size: fc1_units = 400   
- The actor Second hidden layer size: fc2_units = 300
- The critic learning rate: LR_CRITIC = 5e-4 
- The critic first hidden layer size: fcs1_units = 400   
- The critic Second hidden layer size: fc2_units = 300     
#### Ornstein–Uhlenbeck Noise Parameters:
- Exploration Noise level: EPSILON = 1 
- Noise level decay: EPSILON_DECAY = 1e-6 
- The network update interval: LEARN_EVERY = 20     
- Number of network updates: LEARN_NUM = 10
- Noise parameters:
    - Mu = 0
    - Theta = 0.15
    - Sigma = 0.1      
      
-- 
### Results:
The training process was designed to proceed for 700 episodes, where it can stop if the environment was solved (reached an average score of 30 over the last 100 episodes) in less episodes. In this case, the environment was solved in 383 episodes. 
  
![image1]
![image2]

-- 

### Future Ideas:
- **Exploring other training algorithms such as:**   
1- Trust Region Policy Optimization (TRPO)  
2- Truncated Natural Policy Gradient (TNPG)   
3- Proximal Policy Optimization (PPO)   
4- Distributed Distributional Deterministic Policy Gradients (D4PG)     



 
