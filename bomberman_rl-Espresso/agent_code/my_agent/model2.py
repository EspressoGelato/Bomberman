import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# game_state dict_keys(['round', 'step', 'field', 'self', 'others', 'bombs', 'coins', 'user_input', 'explosion_map'])
# game_state['field'].shape is (17, 17), 'field': np.array(width, height)
# 'bombs': A list of tuples((x, y), t), t is the left time for bomb
#'explosion_map': shape (17,17)

class net(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(net, self).__init__()

        layers = []
        layers.append(nn.Linear(input_channels, 50))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(50,30))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(30, num_actions))
        self.func = nn.Sequential(*layers)

    def forward(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x)
        x = x.float()
        output = self.func(x)
        return output


class DQN:

    def __init__(self, state_feature_size, action_size):
        self.state_size = state_feature_size
        self.action_size = action_size


        self.BATCH_SIZE = 128 #128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10

        self.memory = ReplayMemory(1000)

        self.policy_net = self.initialize()
        self.target_net = self.initialize()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        self.steps_done = 0
        self.history_loss = []

    def select_action(self, state_feature):
        state_feature = state_feature.reshape(-1)

        state_feature = torch.from_numpy(state_feature)

        state_feature = state_feature.unsqueeze(0) # size: (1,5,17,17)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                Q_value = self.policy_net(state_feature)
                print(Q_value)
                action_index = Q_value.argmax()
                ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
                return ACTIONS[action_index] # choose the action with max Q_value

        else: # random choose
            return np.random.choice(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'], p=[.2, .2, .2, .2, .1, .1])

    def store_memory(self, state, action, next_state, reward):
        if type(state) is np.ndarray:
            state = state.reshape(-1)
            state = torch.from_numpy(state).unsqueeze(0)
        action = torch.Tensor([action])
        if type(next_state) is np.ndarray:
            next_state = next_state.reshape(-1)
            next_state = torch.from_numpy(next_state).unsqueeze(0)
        reward = torch.Tensor([reward])

        self.memory.push(state, action, next_state, reward)

    def initialize(self):
        return net(input_channels=self.state_size, num_actions=self.action_size)

    def optimize_model(self, episode):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        #print('batch', batch)
        #print('batch.state', batch.state)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])


        state_batch = torch.cat(batch.state)
        #print('state_batch.shape',state_batch.shape)
        action_batch = torch.cat(batch.action)
        #print('action_batch.shape', action_batch.shape)
        reward_batch = torch.cat(batch.reward)
        #print('reward_batch.shape', reward_batch.shape)
        #state_batch.shape torch.Size([10, 5, 17, 17])
        #action_batch.shape torch.Size([10])
        #reward_batch.shape torch.Size([10])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch) # shape (10,6)
        #logits_Q = logits_Q[range(mini_BATCH_SIZE), mini_prefix_length.long() - 1, :]
        state_action_values = state_action_values[range(self.BATCH_SIZE), action_batch.long()]

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        if episode % 100 == 0:
            self.history_loss.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        print('loss', loss)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        #if i_episode % TARGET_UPDATE == 0:
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_model(self, episode):
        torch.save(self.policy_net.state_dict(), './model_saved/dense_DQN_{}.pt'.format(episode))

    def plot_loss(self, episode):
        steps = len(self.history_loss[10:])
        x = range(steps)
        #index = range(100, 10)
        plt.plot(x, self.history_loss[10:])
        plt.xlabel('episode', fontsize=18)
        plt.ylabel('loss', fontsize=16)
        plt.title('training loss')
        plt.savefig('./loss_plot/dense_DQN_loss_episode{}.png'.format(episode))

