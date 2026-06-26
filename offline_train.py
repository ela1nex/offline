import gymnasium as gym
import torch
import minari
import numpy as np
import torch.nn as nn
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm.auto import tqdm
from configs import *
from agent import *

def average(data, window=10):
    window = min(len(data), window)
    return sum(data[-window:]) / window

env   = gym.make("CartPole-v1")
agent = Agent()

# load offline dataset
path1 = "cartpole/expert-v7"
path2 = "cartpole/medium-v0"
dataset1 = minari.load_dataset(path1)
dataset2 = minari.load_dataset(path2)

dataset = minari.combine_datasets(datasets_to_combine=[dataset1, dataset2], new_dataset_id="cartpole/expert-medium-v0")

print(f"Loading offline dataset … {path1} and {path2}")
total_terminations = 0
for episode in dataset.iterate_episodes():
    for i in range(len(episode.rewards)):
        done = bool(episode.terminations[i]) or bool(episode.truncations[i])
        agent.memory.push(
            episode.observations[i],
            episode.actions[i],
            episode.rewards[i],
            episode.observations[i + 1],
            done,
        )
        total_terminations += int(done)

episode_lengths = []
for ep in dataset.iterate_episodes():
    episode_lengths.append(len(ep.rewards))

print(f"min: {min(episode_lengths)}, max: {max(episode_lengths)}, mean: {sum(episode_lengths)/len(episode_lengths):.1f}")

print(f"replay buffer size : {len(agent.memory)} transitions")
print(f"total done signals : {total_terminations}")

state, _ = env.reset()
with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
    agent.critic(torch.FloatTensor(state).unsqueeze(0))

agent.optimize_model()  # one warm-up step

# offline training loop
loss_logs        = []
eval_reward_logs = []

pbar = tqdm(range(training_steps))
for step in pbar:
    critic_loss, cql_penalty = agent.optimize_model()
    loss_logs.append(critic_loss)

    # soft-update target network
    target_state_dict = agent.target.state_dict()
    critic_state_dict = agent.critic.state_dict()
    for key in critic_state_dict:
        target_state_dict[key] = (
            critic_state_dict[key] * tau
            + target_state_dict[key] * (1 - tau)
        )
    agent.target.load_state_dict(target_state_dict)

    # periodic evaluation
    if (step + 1) % log_interval == 0:
        ep_reward  = 0.0
        eval_state, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(eval_state, 0, env)
            eval_state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
        eval_reward_logs.append(ep_reward)

        if verbose == 1:
            print(
                f"\n─────────────────────\n"
                f"step          : {step + 1}\n"
                f"critic loss   : {critic_loss:.4f}\n"
                f"cql penalty   : {cql_penalty:.4f}\n"
                f"eval reward   : {average(eval_reward_logs, window=3):.2f}"
            )

        pbar.set_postfix(
            loss=f"{average(loss_logs):.4f}",
            reward=f"{average(eval_reward_logs, window=3):.2f}",
        )

torch.save(agent.critic.state_dict(), "offline_critic.pth")
print("fin")