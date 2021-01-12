import gym
import math
import random
import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from collections import deque
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

env = gym.make('BreakoutDeterministic-v4').unwrapped


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
print("Is python : {}".format(is_ipython))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : {}".format(device))

ACTIONS_NUM = env.action_space.n
print("Number of actions : {}".format(ACTIONS_NUM))
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

STATE_SIZE = 4
STATE_W = 84
STATE_H = 84
MEMSIZE = 50000

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory:

    def __init__(self, capacity=MEMSIZE):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=ACTIONS_NUM):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 1024, kernel_size=7, stride=1)
        self.advantage = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        advantage, value = torch.split(x, 512, dim=1)

        advantage = advantage.view(advantage.shape[0], -1)
        value = value.view(value.shape[0], -1)

        advantage = self.advantage(advantage)
        value = self.value(value)
        q_value = value.expand(value.shape[0], ACTIONS_NUM) + \
                  advantage - torch.mean(advantage, dim=1).unsqueeze(1).expand(advantage.shape[0], ACTIONS_NUM)

        return q_value

resize = T.Compose([T.ToPILImage(),
                    T.Resize( (STATE_W, STATE_H), interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    screen = env.render(mode='rgb_array')
    screen = np.dot(screen[...,:3], [0.299, 0.587, 0.114])
    screen = screen[30:195,:]
    screen = np.ascontiguousarray(screen, dtype=np.uint8).reshape(screen.shape[0],screen.shape[1],1)
    return resize(screen).unsqueeze(0).mul(255).type(torch.ByteTensor).to(device).detach()

env.reset()
plt.figure()
plt.imshow(get_screen().cpu().reshape(-1,84).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()
plt.figure()
plt.imshow(get_screen().cpu().reshape(-1,84).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-5)

memory = ReplayMemory()


def select_action(state, eps_threshold):
    global steps_done
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            state = state.float() / 255.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(ACTIONS_NUM)]], device=device, dtype=torch.long)


train_rewards = []

mean_size = 100
mean_step = 1


def plot_rewards(rewards=train_rewards, name="Train"):
    plt.figure(2)
    plt.clf()
    plt.title(name)
    plt.xlabel('Episode')
    plt.ylabel('Mean_100ep_reward')
    plt.plot(rewards)
    # plt.plot(range(len(mean)), mean, linewidth=2)
    if len(rewards) > mean_size:
        means = np.array([rewards[i:i + mean_size:] for i in range(0, len(rewards) - mean_size, mean_step)]).mean(1)
        means = np.concatenate((np.zeros(mean_size - 1), means))
        plt.plot(means)
    plt.savefig('train_reward.jpg')


BATCH_SIZE = 32
GAMMA = 0.99


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    # take new batch
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # mask and concatenate everything
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)

    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # callculate Q(s_t,a_t)
    state_batch = state_batch.float() / 255.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # calculate V(s_t)
    non_final_next_states = non_final_next_states.float() / 255.
    next_state_values = torch.zeros((BATCH_SIZE, 1), device=device)
    next_state_actions = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)

    next_state_actions[non_final_mask] = policy_net(non_final_next_states).max(1)[1]
    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_actions[
        non_final_mask].unsqueeze(1))
    next_state_values = next_state_values.squeeze(1)
    # expected Q(s,a)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    #  Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1).detach())
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    del non_final_mask
    del non_final_next_states
    del state_batch
    del action_batch
    del reward_batch
    del state_action_values
    del next_state_values
    del expected_state_action_values
    del loss


# full number of episodes
NUM_EPISODES = 100000

# frames between optimizing
OPTIMIZE_MODEL_STEP = 4
# target_model update
TARGET_UPDATE = 10000

# steps before strart learning
STEPS_BEFORE_TRAIN = 50000

#  e-greedy shedule
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000000

EPS_START_v2 = 0.1
EPS_END_v2 = 0.01

policy_net.train()
target_net.eval()
test_rewards = []
# number of full frames
steps_done = 0


for e in tqdm.tqdm(range(NUM_EPISODES)):
    env.reset()
    a1 = get_screen()
    a2 = get_screen()
    a3 = get_screen()
    a4 = get_screen()
    state = torch.cat([a4, a3, a2, a1], dim=1)

    ep_rewards = 0
    flag = 0
    lives = 5
    for t in range(18000):
        #  eps_threshold
        if steps_done < EPS_DECAY:
            if steps_done > STEPS_BEFORE_TRAIN:
                fraction = min(float(steps_done) / EPS_DECAY, 1)
                eps_threshold = EPS_START + (EPS_END - EPS_START) * fraction
                action = select_action(state, eps_threshold)
            else:
                action = torch.tensor([[random.randrange(ACTIONS_NUM)]], device=device, dtype=torch.long)

        else:
            fraction = min(float(steps_done) / 2 * EPS_DECAY, 1)
            eps_threshold = EPS_START_v2 + (EPS_END_v2 - EPS_START_v2) * fraction
            action = select_action(state, eps_threshold)

        steps_done += 1
        _, reward, done, info = env.step(action.item())

        ep_rewards += reward
        reward = np.clip(reward, -1.0, 1.0)
        reward = torch.tensor([reward], device=device)
        lives1 = info['ale.lives']

        if flag == 0:
            b1 = a2
            b2 = a3
            b3 = a4
            flag = 1
        else:
            b1 = b2
            b2 = b3
            b3 = b4

        b4 = get_screen()
        if not done:
            next_state = torch.cat([b4, b3, b2, b1], dim=1)
            if lives1 != lives:
                lives -= 1
                memory.push(state, action, None, reward)
            else:
                memory.push(state, action, next_state, reward)
        else:
            next_state = None
            memory.push(state, action, next_state, reward)
        state = next_state

        if (steps_done > STEPS_BEFORE_TRAIN) and steps_done % OPTIMIZE_MODEL_STEP == 0:
            optimize_model()

        if steps_done % TARGET_UPDATE == 0:
            print("Target net updated!")
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            train_rewards.append(np.sum(ep_rewards))
            break
    if e%100==0 and e!=0:
       print("Episode score : {}".format(train_rewards[-1]))
       print("Mean score : {}".format(np.mean(train_rewards[-100:])))
       plot_rewards()
    if e%2000==0 and e!=0:
        filepath = 'model_stats_policy_net_' + str(e)
        torch.save(policy_net.state_dict(), filepath)

filepath = 'model_stats_policy_net_' + str(e)
torch.save(policy_net.state_dict(), filepath)
 
     
    

            


