import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# environment
env = gym.make("CartPole-v1", render_mode="human") # creates the env

# dqn
class DQN(nn.Module): # subclasses nn.Module
    def __init__(self, input_dimensions, output_dimensions, layers): # initializes the nn w/ the layers and input/output dimensions given
        super(DQN, self).__init__()
        self.nn = nn.ModuleList()
        self.nn.append(nn.Linear(input_dimensions, 128)) # input dimension -> 128
        for layer in range(layers-2):
            self.nn.append(nn.Linear(128, 128)) # layers in between
        self.nn.append(nn.Linear(128, output_dimensions)) # 128 -> output dimension

    def forward(self, x):
        for layer in range(len(self.nn)-1):
            x = torch.relu(self.nn[layer](x)) # passes input through all layers except last with relu activation function
        x = self.nn[-1](x) # returns output 
        return x

# hyper params
learning_rate = 0.001
gamma = 0.995
epsilon = 0.8
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory_size = 10000
episodes = 1000

# q-network initialization
input_dimensions = env.observation_space.shape[0] # based off the shape of the environment (4 for cart pole)
output_dimensions = env.action_space.n # based off the number of action spaces of the environment (2 for cart pole)
critic = DQN(input_dimensions, output_dimensions, 3) # critic network

optimizer = optim.Adam(critic.parameters(), lr = learning_rate) # optimizer for critic based off defined learning rate
memory = deque(maxlen=memory_size) # the memory of the optimizer with defined max length

# action selection
def select_action(state, epsilon): # selects an action (random or not based on epsilon)
    if random.random() < epsilon: # random for epsilon*100 percent of the time
        return env.action_space.sample() # explores
    else:
        state = torch.FloatTensor(state).unsqueeze(0) # adds dimension of size 1 
        q_values = critic(state) # gets the q values for the current state from the critic
        return torch.argmax(q_values).item() # exploits

# training loop
rewards = [] # rewards for each episode
steps = 0 # number of training(?) steps taken
for episode in range(5): # runs five episodes
    state, info = env.reset() # resets environment
    episode_reward = 0 # sets current reward to 0

    terminated = False 
    truncated = False

    while not terminated and not truncated:
        action = select_action(state, epsilon) # picks action based on current state and epsilon
        next, reward, terminated, truncated, info = env.step(action) # gets the feedback from the environment

        state = next # updates current state
        episode_reward += reward # adds step reward to episode reward

    rewards.append(episode_reward) # adds episode reward to rewards list
