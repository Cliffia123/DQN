#!/usr/bin/env python
# -*- coding: utf-8 -*
import torch
from logger import *
device = torch.device("cuda:0")
dtype = torch.float

class Env_Runner:

    def __init__(self, env, agent):
        super().__init__()

        self.env = env
        self.agent = agent

        self.logger = Logger("training_info")
        self.logger.log("training_step, return")

        self.ob = self.env.reset()
        self.total_steps = 0

    def run(self, steps):
        obs = []
        actions = []
        rewards = []
        dones = []

        for step in range(steps):

            self.ob = torch.tensor(self.ob)  # uint8
            action = self.agent.e_greedy(
                self.ob.to(device).to(dtype).unsqueeze(0) / 255)  # float32+norm
            action = action.detach().cpu().numpy()

            obs.append(self.ob)
            actions.append(action)
            self.ob, r, done, info, additional_done = self.env.step(action)
            if done:  # real environment reset, other add_dones are for q learning purposes
                self.ob = self.env.reset()
                if "return" in info:
                    self.logger.log(f'{self.total_steps + step},{info["return"]}')
            rewards.append(r)
            dones.append(done or additional_done)
        self.total_steps += steps
        return obs, actions, rewards, dones


def make_transitions(obs, actions, rewards, dones):
    # observations 应该是 uint8 格式
    tuples = []

    steps = len(obs) - 1
    for t in range(steps):
        tuples.append((obs[t],
                       actions[t],
                       rewards[t],
                       obs[t + 1],
                       int(not dones[t])))
    return tuples