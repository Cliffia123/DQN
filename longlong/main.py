import gym
from utils import *
from replacy_Memory import *
from dqn import *

# episodes次数
NUM_EPISODES = 100000
# 优化的帧数
OPTIMIZE_MODEL_STEP = 4
# 目标网络更新频率
TARGET_UPDATE=10000
# 开始学习之前的步骤
STEPS_BEFORE_TRAIN = 50000
#  e-greedy 调度
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000000
EPS_START_v2 = 0.1
EPS_END_v2 = 0.01

policy_net.train()
target_net.eval()
test_rewards = []
global steps_done
steps_done = 0

#窗口显示
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

import os
if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY"))==0:
    os.system('bash ../xvfb start')
    os.system('%env DISPLAY=:1')

env = gym.make('BreakoutDeterministic-v4').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
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
                print("Episode score : {}".format(train_rewards[-1]))
                print("Mean score : {}".format(np.mean(train_rewards[-100:])))
                plot_rewards()
                break

            if done and e%2000==0 and e!=0:
                # 每2000次保存一下模型
                filepath = 'model_net_'+str(e)
                torch.save(policy_net.state_dict(), filepath)

def test():

    policy_net = DQN().to(device)
    model_name = 'model_net_12000'
    policy_net.load_state_dict(torch.load('model_net_12000'))
    TEST_EPS = 0.005
    env = gym.make('BreakoutDeterministic-v4').unwrapped

    def show_state(env, step=0, info=""):
        plt.figure(3)
        plt.clf()
        plt.imshow(env.render(mode='rgb_array'))
        plt.title("%s | Step: %d %s" % (env.spec.id, step, info))
        plt.axis('off')
        display.clear_output(wait=True)
        display.display(plt.gcf())

    policy_net.eval()
    env.reset()

    a1 = get_screen()
    a2 = get_screen()
    a3 = get_screen()
    a4 = get_screen()

    total_reward = 0

    for i in count():
        state = torch.cat([a4, a3, a2, a1], dim=1)
        action = select_action(state, TEST_EPS)
        _, reward, done, _ = env.step(action.item())
        total_reward += reward
        if not done:
            a1 = a2
            a2 = a3
            a3 = a4
            a4 = get_screen()
        else:
            break
        show_state(env, i)
    print("Total game reward : {}".format(total_reward))

if __name__ == '__main__':
    main()
    #testing
    test()
