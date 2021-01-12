import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical
import matplotlib
import gym
env = gym.make('BreakoutDeterministic-v4').unwrapped
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
print("Is python : {}".format(is_ipython))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : {}".format(device))


ACTIONS_NUM = env.action_space.n

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
