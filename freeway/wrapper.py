#!/usr/bin/env python
# -*- coding: utf-8 -*
import numpy as np
import gym
import cv2
from scipy.misc import imresize

class Atari_Wrapper(gym.Wrapper):
    # env wrapper 用来调整图像大小，灰度和帧堆叠以及其他内容。
    def __init__(self, env, env_name, k, dsize=(84, 84), use_add_done=False):
        super(Atari_Wrapper, self).__init__(env)
        self.dsize = dsize
        self.k = k
        self.use_add_done = use_add_done

        # 设置图像的cutout

        self.frame_cutout_h = (25, -7)
        self.frame_cutout_w = (7, -7)


    def reset(self):

        self.Return = 0
        self.last_life_count = 0

        ob = self.env.reset()
        ob = self.preprocess_observation(ob)
        self.frame_stack = np.stack([ob for i in range(self.k)])

        return self.frame_stack

    def step(self, action):

        reward = 0
        done = False
        additional_done = False

        frames = []
        for i in range(self.k):

            ob, r, d, info = self.env.step(action)

            # 当agent掉血时，增加一个额外的done
            if self.use_add_done:
                if info['ale.lives'] < self.last_life_count:
                    additional_done = True
                self.last_life_count = info['ale.lives']

            ob = self.preprocess_observation(ob)
            frames.append(ob)
            reward += r

            if d:
                done = True
                break

        self.step_frame_stack(frames)
        self.Return += reward
        if done:
            info["return"] = self.Return

        # clip reward
        if reward > 0:
            reward = 1
        elif reward == 0:
            reward = 0
        else:
            reward = -1

        return self.frame_stack, reward, done, info, additional_done

    def step_frame_stack(self, frames):

        num_frames = len(frames)

        if num_frames == self.k:
            self.frame_stack = np.stack(frames)
        elif num_frames > self.k:
            self.frame_stack = np.array(frames[-self.k::])
        else:

            self.frame_stack[0: self.k - num_frames] = self.frame_stack[num_frames::]
            self.frame_stack[self.k - num_frames::] = np.array(frames)

    def preprocess_observation(self, ob):
        # 调整图像的大小和转为灰度图

        ob = cv2.cvtColor(ob[self.frame_cutout_h[0]:self.frame_cutout_h[1],
                          self.frame_cutout_w[0]:self.frame_cutout_w[1]], cv2.COLOR_BGR2GRAY)
        ob = cv2.resize(ob, dsize=self.dsize)

        return ob