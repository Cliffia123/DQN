#!/usr/bin/env python
# -*- coding: utf-8 -*
import gym
import torch.optim as optim
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from wrapper import *
from runner import *
from experience_Replay import *
from agent import *

env_name = 'SpaceInvadersNoFrameskip-v4'
total_episodes = 14
eval_epsilon = 0.01
num_stacked_frames = 4

def main():

    # 参数设置
    num_stacked_frames = 4

    replay_memory_size = 250000
    min_replay_size_to_update = 25000

    lr = 6e-5
    gamma = 0.99
    minibatch_size = 32
    steps_rollout = 16

    start_eps = 1
    final_eps = 0.1

    final_eps_frame = 1000000
    total_steps = 20000000

    target_net_update = 625  # 10000 steps

    save_model_steps = 500000

    # 游戏初始化
    raw_env = gym.make(env_name)
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames, use_add_done=True)

    in_channels = num_stacked_frames
    num_actions = env.action_space.n
    eps_interval = start_eps - final_eps

    agent = Agent(in_channels, num_actions, start_eps).to(device)
    target_agent = Agent(in_channels, num_actions, start_eps).to(device)
    target_agent.load_state_dict(agent.state_dict())

    replay = Experience_Replay(replay_memory_size)
    runner = Env_Runner(env, agent)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    huber_loss = torch.nn.SmoothL1Loss()

    num_steps = 0
    num_model_updates = 0

    start_time = time.time()
    while num_steps < total_steps:

        # agent探索| 在x个时间步长后对最终epsilon进行探索
        new_epsilon = np.maximum(final_eps, start_eps - (eps_interval * num_steps / final_eps_frame))
        agent.set_epsilon(new_epsilon)

        # 得到数据
        obs, actions, rewards, dones = runner.run(steps_rollout)
        transitions = make_transitions(obs, actions, rewards, dones)
        replay.insert(transitions)

        # 添加
        num_steps += steps_rollout

        # 检查是否需要更新
        if num_steps < min_replay_size_to_update:
            continue

        # 更新
        for update in range(4):
            optimizer.zero_grad()

            minibatch = replay.get(minibatch_size)

            # uint8 to float32 and normalize to 0-1
            obs = (torch.stack([i[0] for i in minibatch]).to(device).to(dtype)) / 255

            actions = np.stack([i[1] for i in minibatch])
            rewards = torch.tensor([i[2] for i in minibatch]).to(device)

            # uint8 to float32 and normalize to 0-1
            next_obs = (torch.stack([i[3] for i in minibatch]).to(device).to(dtype)) / 255

            dones = torch.tensor([i[4] for i in minibatch]).to(device)
            Qs = agent(torch.cat([obs, next_obs]))
            obs_Q, next_obs_Q = torch.split(Qs, minibatch_size, dim=0)

            obs_Q = obs_Q[range(minibatch_size), actions]

            # target

            next_obs_Q_max = torch.max(next_obs_Q, 1)[1].detach()
            target_Q = target_agent(next_obs)[range(minibatch_size), next_obs_Q_max].detach()

            target = rewards + gamma * target_Q * dones
            # loss
            loss = huber_loss(obs_Q, target)
            loss.backward()
            optimizer.step()

        num_model_updates += 1

        # 更新目标网络
        if num_model_updates % target_net_update == 0:
            target_agent.load_state_dict(agent.state_dict())

        # 每50000step更新一次
        if num_steps % 50000 < steps_rollout:
            end_time = time.time()
            print(f'*** total steps: {num_steps} | time(50K): {end_time - start_time} ***')
            start_time = time.time()

        # 保存模型
        if num_steps % save_model_steps < steps_rollout:
            torch.save(agent, f"{env_name}-{num_steps}.pt")

    env.close()
def evaluate():
    f = open(env_name + "-Eval" + ".csv", "w")
    for filename in os.listdir():

        if env_name not in filename or ".pt" not in filename:
            continue

        print("load file name", filename)
        agent = torch.load(filename).to(device)
        agent.set_epsilon(eval_epsilon)
        agent.eval()

        raw_env = gym.make(env_name)
        env = Atari_Wrapper(raw_env, env_name, num_stacked_frames)

        ob = env.reset()
        num_episode = 0
        returns = []
        while num_episode < total_episodes:

            action = agent.e_greedy(torch.tensor(ob, dtype=dtype).unsqueeze(0).to(device) / 255)
            action = action.detach().cpu().numpy()

            ob, _, done, info, _ = env.step(action, render=True)

            # time.sleep(0.016)

            if done:
                ob = env.reset()
                returns.append(info["return"])
                num_episode += 1

        env.close()
        steps = filename.strip().split(".")[0].split("-")[-1]
        f.write(f'{steps},{np.mean(returns)}\n')

def draw_fig():
    plt.rcParams["figure.figsize"] = (8, 5)
    filename = "training_info.csv"
    n_steps = 100
    f = pd.read_csv(filename)
    mean = f[" return"].rolling(n_steps).mean()
    plt.plot(f["training_step"], mean, linewidth=2)
    plt.xlabel("Number of steps")
    plt.ylabel("mean_100ep_rewards")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
    #evaluate
    evaluate()
    #draw
    draw_fig()
