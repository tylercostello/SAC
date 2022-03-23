import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
from typing import Optional

import numpy as np


import gym
from gym import spaces
from gym.utils import seeding

class Motor(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # There are two actions, first will get reward of 1, second reward of -1.
        self.min_action = 0.0
        self.max_action = 1.0
        self.min_position = 0.0
        self.max_position = 1.0
        self.steps_taken = 0



        self.low_state = np.array(
            [self.min_position], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position], dtype=np.float32
        )

        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

    def step(self, action):

        # if we took an action, we were in state 1
        state = 1*action
        reward=-1*action
        print(reward)

        self.steps_taken += 1
        #print(self.steps_taken)
        # if action == 1.0:
        #     reward = 1
        # else:
        #     reward = -1
        #print(reward)
        # regardless of the action, game is done after a single step

        if action>=0.9:
            reward=100
            done = True
        else:
            done = False

        info = {}

        return state, reward, done, info

    def reset(self):
        state = 0
        self.steps_taken = 0
        return state

    def render(self, mode='human'):
        pass

    def close(self):
        pass