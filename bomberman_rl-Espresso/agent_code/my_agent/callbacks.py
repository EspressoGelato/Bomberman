import os
import pickle
import random

import numpy as np
import traceback
import sys
import pprint
from .model import *

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']



def setup(self):

    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    print('run my_agent/callback.py function setup')
    print('show me self', self)

    #if self.train or not os.path.isfile("my-saved-model.pt"):
        #self.logger.info("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()
    if self.train:
        self.deepQ = DQN(state_feature_size = 5, action_size = 6) #convolution net
        #self.deepQ = DQN(state_feature_size=5 * 17 * 17, action_size=6)  # fully-connected net
        print('!!!!!!!!!', self.deepQ.BATCH_SIZE)


    else:
        self.logger.info("Loading model from saved state.")
        #with open("./model_saved/DQN_1000.pt", "rb") as file:
        #    self.model = pickle.load(file)
        self.state_dict = torch.load("./model_saved/DQN_2000.pt")
        self.deepQ_eval = DQN(state_feature_size=5, action_size=6)
        #self.state_dict = torch.load("./model_saved/dense_DQN_1000.pt")
        #self.deepQ_eval = DQN(state_feature_size=5 * 17 * 17, action_size=6)
        self.deepQ_eval.policy_net.load_state_dict(self.state_dict)
        self.deepQ_eval.policy_net.eval()

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    print('run my_agent/callback.py function act')

    if self.train:
        print("select action based on Q_net")
        state_feature = state_to_features(game_state)
        action = self.deepQ.select_action(state_feature)
        return action
    else:
        print("select action based on Q_net")
        state_feature = state_to_features(game_state)
        action_Q = self.deepQ_eval.select_action(state_feature)
        return action_Q


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    print('run my_agent/callback.py function state_to_features')
    #print(sys._getframe().f_code.co_name)

    if game_state is None:
        return None
    else: # get information from game_state dict
    # game_state dict_keys(['round', 'step', 'field', 'self', 'others', 'bombs', 'coins', 'user_input', 'explosion_map'])
        field_ = 5 * game_state['field'] # shape(w,h)
        # in field: 1 for crates, -1 for stone walls and 0 for free tiles.
        bombs_position = np.zeros_like(field_)
        coins_position = np.zeros_like(field_)
        others_position = np.zeros_like(field_)
        self_position = np.zeros_like(field_)

        self_ = game_state['self'][-1] #tuple(x,y)
        self_position[self_[0], self_[1]] = 20

        for other_i in game_state['others']:
            other_i_ = other_i[-1]
            others_position[other_i_[0], other_i_[1]] = -20

        #tuple(x,y)

        #print(game_state)
        if game_state['bombs']:
            #print(game_state['bombs'])
            bombs_ = game_state['bombs'][0] #tuple(x,y)
            bombs_position[bombs_[0], bombs_[1]] = -100

        if game_state['coins']:
            coins_ = game_state['coins'][0] #tuple(x,y)
            coins_position[coins_[0], coins_[1]] = 50


        #state_feature = field_ + bombs_position + coins_position + self_position + others_position # shape(w,h)
        #print('sate_feature', state_feature)
        #state_feature = state_feature[None]

        state_feature = np.stack([field_, bombs_position, coins_position, self_position, others_position], 0)
        #print(state_feature.shape)

        return state_feature




    # For example, you could construct several channels of equal shape, ...
    #channels = []
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)
    # and return them as a vector
    #return stacked_channels.reshape(-1)
