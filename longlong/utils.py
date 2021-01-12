
from dqn import *
from replacy_Memory import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    plt.ylabel('mean_100ep_rewards')
    plt.plot(rewards)
    if len(rewards) > mean_size:
        means = np.array([rewards[i:i + mean_size:] for i in range(0, len(rewards) - mean_size, mean_step)]).mean(1)
        means = np.concatenate((np.zeros(mean_size - 1), means))
        plt.plot(means)
    plt.grid()
    plt.savefig("mean_100ep_rewards.jpg")

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
    state_batch =state_batch.float( ) /255.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # calculate V(s_t)
    non_final_next_states =non_final_next_states.float( ) /255.
    next_state_values = torch.zeros((BATCH_SIZE ,1), device=device)
    next_state_actions = torch.zeros(BATCH_SIZE ,dtype=torch.long, device=device)

    next_state_actions[non_final_mask] = policy_net(non_final_next_states).max(1)[1]
    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_actions
        [non_final_mask].unsqueeze(1))
    next_state_values =next_state_values.squeeze(1)
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