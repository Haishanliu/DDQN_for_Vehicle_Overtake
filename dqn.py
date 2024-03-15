import torch
import torch.nn as nn

from typing import Tuple
from numpy.random import binomial
from numpy.random import choice
import torch.nn.functional as F
import warnings
import os
import datetime
warnings.filterwarnings('ignore')

Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)


class DQN:
    def __init__(self, config):

        torch.manual_seed(config['seed'])

        self.lr = config['lr']  # learning rate
        self.C = config['C']  # copy steps
        self.eps_len = config['eps_len']  # length of epsilon greedy exploration
        self.eps_max = config['eps_max']
        self.eps_min = config['eps_min']
        self.discount = config['discount']  # discount factor
        self.batch_size = config['batch_size']  # mini batch size

        self.dims_hidden_neurons = config['dims_hidden_neurons']
        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']

        self.Q = QNetwork(dim_obs=self.dim_obs,
                          dim_action=self.dim_action,
                          name = 'Q')
        self.Q_tar = QNetwork(dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                              name = 'Q_tar')

        self.optimizer_Q = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.training_step = 0

    def update(self, buffer):
        t = buffer.sample(self.batch_size)
        s = t.obs.double()
        a = t.action.type(torch.int64)
        r = t.reward
        sp = t.next_obs.double()
        done = t.done

        # get Q(s,a)
        predicted_state_value = self.Q.forward(s) # use the current Q network to get the predicted value
        Q_values = torch.gather(predicted_state_value, 1, a) # get the Q value with the action in the tuple

        with torch.no_grad(): ## self.Q_tar.forward(sp).max(dim=1)[0]-- get the maximum action-value, unsqueeze(dim=1)-- reshape to (-1, 1)
            next_expected_return = self.Q_tar.forward(sp).max(dim=1)[0].unsqueeze(dim=1)
            y = r + self.discount * (1-done.int()) * next_expected_return

        criterion = nn.MSELoss() # Huber loss
        loss = criterion(Q_values, y)

        # optimize the model
        self.optimizer_Q.zero_grad()
        loss.backward()
        self.optimizer_Q.step()

        self.training_step += 1
        if self.training_step % self.C == 0: # update the target Q network every C Q network update steps
            self.Q_tar.load_state_dict(self.Q.state_dict())  # copy the parameters to target network
        return loss.item()

        # TODO: perform a single Q network update step. Also update the target Q network every C Q network update steps

    def act_probabilistic(self, observation: torch.Tensor):
        # epsilon greedy:
        first_term = self.eps_max * (self.eps_len - self.training_step) / self.eps_len
        eps = max(first_term, self.eps_min)

        explore = binomial(1, eps)

        if explore == 1:
            a = choice(self.dim_action)
        else:
            self.Q.eval()
            Q = self.Q(observation) # ??
            print('non per',Q)
            val, a = torch.max(Q, axis=1)
            a = a.item()
            self.Q.train()
        return a

    def act_deterministic(self, observation: torch.Tensor):
        self.Q.eval() # 这是什么fuction
        Q = self.Q(observation)
        val, a = torch.max(Q, axis=1)
        self.Q.train()
        return a.item()

    def save_models(self):
        self.Q.save_checkpoint()
        self.Q_tar.save_checkpoint()

    def load_models(self):
        self.Q.load_checkpoint()
        self.Q_tar.load_checkpoint()

class Double_DQN:
    def __init__(self, config):

        torch.manual_seed(config['seed'])

        self.lr = config['lr']  # learning rate
        self.C = config['C']  # copy steps
        self.eps_len = config['eps_len']  # length of epsilon greedy exploration
        self.eps_max = config['eps_max']
        self.eps_min = config['eps_min']
        self.discount = config['discount']  # discount factor
        self.batch_size = config['batch_size']  # mini batch size

        self.dims_hidden_neurons = config['dims_hidden_neurons']
        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']

        self.Q_action_pre = QNetwork(dim_obs=self.dim_obs,
                          dim_action=self.dim_action,
                         name='Q_action_pre') # online network to select action
        self.Q_action_tar = QNetwork(dim_obs=self.dim_obs,
                          dim_action=self.dim_action,
                         name='Q_action_tar') # online network to select action

        self.Q_value_pre = QNetwork(dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                             name = 'Q_value_pre') # target network to evalute

        self.Q_value_tar = QNetwork(dim_obs=self.dim_obs,
                                     dim_action=self.dim_action,
                                     name='Q_value_tar')  # online network to select action

        self.Q_value_pre.load_state_dict(self.Q_action_pre.state_dict())
        self.Q_action_tar.load_state_dict(self.Q_action_pre.state_dict()) # 初始化网络参数
        self.Q_value_tar.load_state_dict(self.Q_value_pre.state_dict())

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer_Q_action = torch.optim.Adam(self.Q_action_pre.parameters(), lr=self.lr)
        self.optimizer_Q_value = torch.optim.Adam(self.Q_value_pre.parameters(), lr=self.lr)
        self.training_step = 0

    def update(self, buffer):
        t = buffer.sample(self.batch_size)
        s = t.obs.double()
        a = t.action.type(torch.int64)
        r = t.reward
        sp = t.next_obs.double()
        done = t.done

        predicted_state_value = self.Q_action_pre.forward(s) # use the current Q network to get the predicted value
        Q_values = torch.gather(predicted_state_value, 1, a).to(self.device) # get the Q value with the action in the tuple

        with torch.no_grad(): ## self.Q_tar.forward(sp).max(dim=1)[0]-- get the maximum action-value, unsqueeze(dim=1)-- reshape to (-1, 1)
            next_max_action = self.Q_action_tar.forward(sp).argmax(dim=1).unsqueeze(dim=1)# get the greedy action
            y = r + self.discount * (1-done.int()) * torch.gather(self.Q_value_tar.forward(sp),1,next_max_action)
            y = y.to(self.device)

        criterion = nn.MSELoss() # Huber loss
        loss = criterion(Q_values, y)

        # optimize the model
        self.optimizer_Q_action.zero_grad()
        self.optimizer_Q_value.zero_grad()
        loss.backward()
        self.optimizer_Q_action.step()
        self.optimizer_Q_value.step()

        # loss.backward()
        self.training_step += 1
        if self.training_step % self.C == 0: # update the target Q network every C Q network update steps
            self.Q_action_tar.load_state_dict(self.Q_action_pre.state_dict())  # copy the parameters to target network
            self.Q_value_tar.load_state_dict(self.Q_value_pre.state_dict())

        return loss.item()

    def act_probabilistic(self, observation: torch.Tensor):
        # epsilon greedy:
        first_term = self.eps_max * (self.eps_len - self.training_step) / self.eps_len
        eps = max(first_term, self.eps_min)

        explore = binomial(1, eps)

        if explore == 1:
            a = choice(self.dim_action)
        else:
            self.Q_action_pre.eval()
            Q = self.Q_action_pre.forward(observation)  # ??
            val, a = torch.max(Q, axis=1)
            a = a.item()
            self.Q_action_pre.train()
        return a

    def act_deterministic(self, observation: torch.Tensor):
        self.Q_action_pre.eval()  # 这是什么fuction
        Q_value = self.Q_action_pre.forward(observation)
        val, a = torch.max(Q_value, axis=1)
        self.Q_action_pre.train()
        return a.item()

    def save_models(self):
        self.Q_action_pre.save_checkpoint()
        self.Q_action_tar.save_checkpoint()
        self.Q_value_pre.save_checkpoint()
        self.Q_value_tar.save_checkpoint()


    def load_models(self):
        self.Q_action_pre.load_checkpoint()
        self.Q_action_tar.load_checkpoint()
        self.Q_value_pre.load_checkpoint()
        self.Q_value_tar.load_checkpoint()

class QNetwork(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 name,
                 fc1_unit = 256,
                 fc2_unit= 256,
                 chkpt_dir='tmp_5_vehicle/highway_ddqn' #记录新的network
                 ):
        super(QNetwork, self).__init__() # Q network 会继承父类 nn.Module的所有方法
        self.chkpt_file = os.path.join(chkpt_dir, name + '_dqn'+'{date:%Y-%m-%d-%H-%M-%S}'.format(date=datetime.datetime.now()))
        self.fc_1 = nn.Linear(dim_obs,fc1_unit)
        self.fc_2 = nn.Linear(fc1_unit,fc2_unit)
        self.fc_3 = nn.Linear(fc2_unit,dim_action)


    def forward(self, observation: torch.Tensor): # why only use observation, not action as input
        x = observation.flatten(start_dim=1).double()
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        return self.fc_3(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        # self.load_state_dict(torch.load(self.chkpt_file))
        # self.load_state_dict(torch.load(self.chkpt_file))
        self.load_state_dict(torch.load(r'..\Q_dqn2022-05-26-15-55-42'))

