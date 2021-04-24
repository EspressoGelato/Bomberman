import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
from .callbacks import act
from .model import *



# This is only an example!
#Transition = namedtuple('Transition',
                        #('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
#TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
#RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
from pprint import pprint

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    print('run my_agent/train.py function setup_training')
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    #print('self.transitions', self.transitions)
    # get parameters:
    # set up Q-model
    self.episode = 0
    self.score = []


    #pprint(vars(self))

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    print('run my_agent/train.py function game_events_occurred')
    print('action', self_action)
    print('events', events)
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if self.train and self_action is not None:
        reward = reward_from_events(self, events)

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
        state_old = state_to_features(old_game_state) #numpy.array
        state_new = state_to_features(new_game_state) #numpy.array

        ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
        action_index = ACTIONS.index(self_action) #int

        self.deepQ.store_memory(state_old, action_index, state_new, reward)

        self.deepQ.optimize_model(self.episode)




def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    print('run my_agent/train.py function end_of_round')
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    if self.train and last_action is not None:
        reward = reward_from_events(self, events) # float
        state_old = state_to_features(last_game_state)

        ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
        print('last_action', last_action)
        action_index = ACTIONS.index(last_action)  # int
        print(action_index)
        total_score = last_game_state['self'][1]
        print(total_score)
        if self.episode % 100 == 0:
            self.score.append(total_score)



        self.deepQ.store_memory(state_old, action_index, None, reward + total_score)
        self.episode += 1
        #print('episode', self.episode)


        if self.episode % self.deepQ.TARGET_UPDATE == 0:
            print('update target net')
            self.deepQ.update_target()

        if self.episode % 1000 == 0:
            print('plot loss')
            self.deepQ.plot_loss(self.episode)

        if self.episode % 1000 == 0:
            print('plot score')
            self.deepQ.plot_score(self.score, self.episode)

        if self.episode % 1000 == 0:
            print('save model')
            self.deepQ.save_model(self.episode)






    # Store the model
    #with open("my-saved-model.pt", "wb") as file:
     #   pickle.dump(self.deepQ, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    print('run my_agent/train.py function reward_from_events')
    game_rewards = {
        e.COIN_COLLECTED: 3,
        e.KILLED_OPPONENT: 5,
        #PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
        e.COIN_FOUND: 0.5,
        e.SURVIVED_ROUND: 0.5,
        e.GOT_KILLED: -5,
        e.KILLED_SELF: -5,
        e.CRATE_DESTROYED: 0.5,
        e.BOMB_DROPPED: 0.3,
        e.BOMB_EXPLODED: 0.3,
        e.INVALID_ACTION: -2

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum




