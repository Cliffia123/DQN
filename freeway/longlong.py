import tensorflow as tf
tf.compat.v1.enable_eager_execution() 

#窗口显示
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

import os
if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY"))==0:
    os.system('bash ../xvfb start')
    os.system('env DISPLAY=:1')

# #使用gpu进行计算
from tensorflow.python.keras import backend as K

# tf.compat.v1.disable_eager_execution()

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 3 } )
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)
print(config)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


import gym
import logging
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.losses import *
from tensorflow.keras import activations

tf.config.experimental_run_functions_eagerly(True)



class DeepQNetwork(Model):
    def __init__(self,img_size,batch_size,num_actions,
                 training_cycle,target_update_cycle,enable_DDQN):
        super(DeepQNetwork,self).__init__()
        self.batch_size = batch_size
        self.evaluation_network = self.build_model(img_size,num_actions)
        self.target_network = self.build_model(img_size,num_actions)
        self.opt = tf.keras.optimizers.Adam(0.01)
        # Discount factor
        self.gamma = 0.90
        # tf.compat.v1.losses.mean_squared_error()
        # self.loss_function = tf.losses.mean_squared_error()
        # self.loss_function = MeanSquaredError()

        self.training_counter = 0
        self.training_cycle = training_cycle
        self.target_update_counter = 0
        self.target_update_cycle = target_update_cycle
        self.memroies_nameList = ['state','action','reward','next_state']
        self.memories_dict = {}
        for itemname in self.memroies_nameList:
            self.memories_dict[itemname] = None

        self.enable_DDQN = enable_DDQN
    def build_model(self,img_size,num_actions):
        model = tf.keras.Sequential([\
            Conv2D(32,8,8,input_shape=img_size),
            Activation('relu'),
            Conv2D(64,4,4),
            Activation('relu'),
            Conv2D(128,2,2),
            Activation('relu'),
            Flatten(),
            Dense(512),
            Activation('relu'),
            Dense(num_actions)])
        return model

    def train(self):
        # DQN - Experience Replay for Mini-batch
        random_select = np.random.choice(self.training_cycle,self.batch_size)
        states = self.memories_dict["state"][random_select]
        actions = self.memories_dict["action"][random_select]
        rewards = self.memories_dict["reward"][random_select]
        nextStates = self.memories_dict["next_state"][random_select]
        with tf.GradientTape() as tape:
            q_eval_arr = self.evaluation_network(states)
            q_eval = tf.reduce_max(q_eval_arr,axis=1)
            # print("q_eval: {}".format(q_eval))
            if self.enable_DDQN == True:
                # Double Deep Q-Network
                q_values = self.evaluation_network(nextStates)
                q_values_actions = tf.argmax(q_values,axis=1)
                target_q_values = self.target_network(nextStates)
                # discount_factor = target_q_values[range(self.batch_size),q_values_actions]
                indice = tf.stack([range(self.batch_size),q_values_actions],axis=1)
                discount_factor = tf.gather_nd(target_q_values,indice)
            else:
                # Deep Q-Network
                target_q_values = self.target_network(nextStates)
                discount_factor = tf.reduce_max(target_q_values,axis=1)
            
             # Q function
            q_target = rewards + self.gamma * discount_factor
            # print("q_target: {}".format(q_target))
            loss = tf.losses.mean_squared_error(q_eval,q_target)
        
        gradients_of_network = tape.gradient(loss,self.evaluation_network.trainable_variables)
        self.opt.apply_gradients(zip(gradients_of_network, self.evaluation_network.trainable_variables))
        self.target_update_counter += 1
        # DQN - Frozen update
        if self.target_update_counter % self.target_update_cycle == 0:
            self.target_network.set_weights(self.evaluation_network.get_weights())
        
    def call(self):
        return
        
    def append_experience(self,dict):
        tc = self.training_counter
        for itemname in self.memroies_nameList:
            if self.memories_dict[itemname] is None:
                self.memories_dict[itemname] = dict[itemname]
            else:
                self.memories_dict[itemname] = np.append(self.memories_dict[itemname],dict[itemname],axis=0)
        self.training_counter += 1

    def delete_experience(self):
        for itemname in self.memroies_nameList:
            self.memories_dict[itemname] = None
        self.training_counter = 0

def convert_gym_state(state):
    state = tf.image.convert_image_dtype(state,tf.float32)
    state = tf.expand_dims(state,0)
    return state

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # openai-gym config of this game
    env = gym.make('Freeway-v0')
    # observation_space: Box(210,160,3)
    NUM_STATES = env.observation_space.shape
    # action_space: Discrete(3)
    NUM_ACTIONS = env.action_space.n
    NUM_EPOSIDES = 4000
    NUM_BATCHES = 32
    INITIAL_EPSILON = 0.4
    FINAL_EPSILON = 0.05
    EPSILON_DECAY = 1000000
    TRAINING_CYCLE = 2000
    TARGET_UPDATE_CYCLE = 100
    epsilon = INITIAL_EPSILON

    # outdir = './results'
    # env = gym.wrappers.Monitor(env,directory=outdir,force=True)
    network = DeepQNetwork(NUM_STATES,NUM_BATCHES,NUM_ACTIONS,TRAINING_CYCLE,TARGET_UPDATE_CYCLE,False)
    episode_rewards = [0.0]
    avg_raward = []
    for episode in range(NUM_EPOSIDES):

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY
        
        episode_reward = 0
        state = env.reset()
        state = convert_gym_state(state)
        t = 0
        while True:
            # env.render()
            if np.random.uniform() < epsilon:
                action = np.random.randint(0,NUM_ACTIONS)
            else:
                tmp = network.evaluation_network(state)
                action = tf.argmax(tmp[0])
            next_state, reward, done, _ = env.step(action)
            next_state = convert_gym_state(next_state)
            network.append_experience({'state':state,
            'action':[action],'reward':[reward],'next_state': next_state})
            episode_reward += reward
            episode_rewards[-1]+=reward

            if network.training_counter >= network.training_cycle:
                network.train()
                network.delete_experience()
            state = next_state

            if done:
                logging.info('Episode {} finished after {} timesteps, total rewards {}'.format(episode, t+1, episode_reward))
                episode_rewards.append(0.0)
                break
            t += 1

        if episode%300 == 0 and episode!=0:
           plt.xlabel('Epoch')
           plt.ylabel('mean_episode_reward')
           plt.plot(avg_raward)
           plt.draw()
           plt.savefig('caoxz_reward_'+str(episode)+'.jpg')

           model = agent.create_model()
           model.save_weights('ll_weights'+str(episode)+'.h5')
           np.savetxt('rewards_per_episode.csv', episode_rewards, delimiter=',', fmt='%1.3f')

        mean_100ep_reward = round(np.mean(episode_rewards[:-1]), 1)
        avg_raward.append(mean_100ep_reward)

        if episode % 5 == 0:
            num_episodes = len(episode_rewards)
            mean_100ep_reward = round(np.mean(episode_rewards[:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t+1))
            print("episodes: {}".format(episode))
            print("mean {} episode reward: {}".format(num_episodes, mean_100ep_reward))
            print("********************************************************")
            network.save_weights("rl.h5")

    env.close()

