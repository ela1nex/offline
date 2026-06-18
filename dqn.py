import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# environment
env = gym.make("CartPole-v1", render_mode="rgb_array") # creates the env

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
        state = torch.FloatTensor(state).unsqueeze(0) # adds dimension of size 1 at position 0
        q_values = critic(state) # gets the q values for the current state from the critic
        return torch.argmax(q_values).item() # exploits

def optimize_model():
    if len(memory) < batch_size: # if the memory is smaller than batch size then it wont optimize
        return

    # TODO: custom memory class 
    batch = random.sample(memory, batch_size) # randomly samples a batch of batch_size from the memory
    state_batch, action_batch, reward_batch, next_batch, done_batch = zip(*batch)

    # turn everything into tensors
    state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32) 
    action_batch = torch.LongTensor(action_batch).unsqueeze(1) # adds dimension but this time at position 1
    reward_batch = torch.FloatTensor(reward_batch) 
    next_batch = torch.tensor(np.array(next_batch), dtype=torch.float32)
    done_batch = torch.FloatTensor(done_batch)

    q_values = critic(state_batch).gather(1, action_batch).squeeze() # predicts q-values for all actions and extracts value of action actually taken

    with torch.no_grad(): # does not remember operations   
        max_next_q_values = critic(next_batch).max(1)[0] # outputs best possible q-value from next state
        target_q_values = reward_batch + gamma * max_next_q_values * (1-done_batch) # calculates immediate reward and future estimated reward
    
    loss = nn.MSELoss()(q_values, target_q_values) # calculates distance from predicated q-values to target q-value

    optimizer.zero_grad() # clears old gradients
    loss.backward() # computes gradients of loss w.r.t. model parameters
    optimizer.step() # updatse network weights 

# training loop
rewards = [] # rewards for each episode
steps = 0 # number of training steps taken
for episode in range(episodes): # runs five episodes
    state, info = env.reset() # resets environment
    episode_reward = 0 # sets current reward to 0

    terminated = False 
    truncated = False

    while not terminated and not truncated:
        action = select_action(state, epsilon) # picks action based on current state and epsilon
        next, reward, terminated, truncated, info = env.step(action) # gets the feedback from the environment

        memory.append((state, action, reward, next, terminated or truncated)) # add step to memory

        state = next # updates current state
        episode_reward += reward # adds step reward to episode reward

        optimize_model()

        steps += 1
    
    epsilon = max(epsilon_min, epsilon_decay * epsilon) # decays epsilon

    rewards.append(episode_reward) # adds episode reward to rewards list

# TODO testing loop
