import gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from dqn import Double_DQN,DQN
from buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch
from gym.wrappers import RecordVideo
from time import time


Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)


env = gym.make('highway-v0')
# env.config['observation']['type'] = 'GrayscaleObservation'

env_config = {
    'observation':{"type": "Kinematics",
        "vehicles_count": 5, #控制observation 里面，观察near-driver 的数量,default
        "features": ['presence',"x", "y", "vx", "vy"],
        "absolute": False,
        "order": "sorted"},
        'vehicles_count': 50,
        'duration': 40,
        "lanes_count":4,
        'vehicle density':1,
        'policy_frequency': 1,
        'simulation_frequency':15,
        }
env.configure(env_config )
env.reset() ## initialize the environment

print(env.config)

# config for DQN
config = {
    'dim_obs':5*5,  # Q network input
    'dim_action': 5,  # action type ACTIONS_ALL = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}
    'dims_hidden_neurons': (256, 256),  # Q network hidden
    'lr': 0.00005,  # learning rate
    'C': 50,  # copy steps
    'discount': 0.7,  # discount factor
    'batch_size': 64,
    'buffer_size': 100000,
    'eps_min': 0.0001,
    'eps_max': 1.0,
    'eps_len': 4000,
    'seed': 1,
}

ddqn = Double_DQN(config)
# dqn = DQN(config)
buffer = ReplayBuffer(config)
train_writer = SummaryWriter(log_dir='tensorboard/ddqn_{date:%Y-%m-%d-%H-%M-%S}'.format(
                             date=datetime.datetime.now()))

def train(iterations):
    steps = 0  # total number of steps
    loss_counter = 0
    for i_episode in range(iterations):
        observation = env.reset()
        done = False
        t = 0  # time steps within each episode
        ret = 0.  # episodic return
        ep_return = []
        while done is False:
            env.render()  # render to screen
            obs = torch.tensor(observation)  # observe the environment state
            action = ddqn.act_probabilistic(obs[None, :])  # take action
            # action = dqn.act_probabilistic(obs[None, :])
            next_obs, reward, done, info = env.step(action)  # environment advance to next step

            buffer.append_memory(obs=obs,  # put the transition to memory
                                 action=torch.from_numpy(np.array([action])),
                                 reward=torch.from_numpy(np.array([reward])),
                                 next_obs=torch.from_numpy(next_obs),
                                 done=done)

            running_loss = ddqn.update(buffer)  # agent learn
            # running_loss = dqn.update(buffer)
            observation = next_obs
            t += 1
            steps += 1
            ret += reward  # update episodic return
            train_writer.add_scalar('Performance/training_loss', running_loss, loss_counter)  #
            loss_counter +=1
            if done:
                print("Episode {} finished after {} timesteps with return {}".format(i_episode, t+1, ret))
                train_writer.add_scalar('Performance/episodic_return', ret, i_episode)  # plot
        if i_episode % 50 ==0:
            ddqn.save_models()
            # dqn.save_models()
    env.close()
    train_writer.close()


train(iterations=2000) # train the network


## use for test
def test(env, ddqn):
    env = gym.wrappers.Monitor(env, './videos/' + str(time()) + '/')
    env = RecordVideo(env, video_folder="./ddqn_run_2",
                      episode_trigger=lambda e: True)  # record all episodes
    env.unwrapped.set_record_video_wrapper(env)
    t = 10
    while t>0:
        done = False
        obs = env.reset()
        # obs = torch.tensor(obs)
        while not done:
            obs = torch.tensor(obs)
            action = ddqn.act_deterministic(obs[None, :])
            next_obs, reward, done, info = env.step(action)
            env.render()
            obs = next_obs
        t -=1
    env.close()

# dqn.load_models()
# test(env, dqn)
