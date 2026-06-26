from minari import DataCollector
from tqdm.auto import tqdm
import gymnasium as gym
import torch

from agent import *

# generate dataset using DataCollector wrapper
env = DataCollector(gym.make('CartPole-v1'))
agent = Agent()
agent.critic.load_state_dict(torch.load("critic_med.pth"))
agent.critic.eval()
agent.dynamic.load_state_dict(torch.load("dynamic_med.pth"))
agent.dynamic.eval()

total_episodes=1000
for i in tqdm(range(total_episodes)):
     state, info = env.reset()
     terminated = False
     truncated = False
     steps = 0
     while not terminated and not truncated:
          action = agent.select_action(state, 0, env)
          next_state, reward, terminated, truncated, info = env.step(action)
          state = next_state
          steps += 1
print(steps)

dataset = env.create_dataset(dataset_id="cartpole/medium-v0", algorithm_name="MediumDynamicDQN", author="ejx", author_email="elainejxia@gmail.com", code_permalink="https://github.com/ela1nex/customDQN", eval_env=gym.make('CartPole-v1'), description="")

