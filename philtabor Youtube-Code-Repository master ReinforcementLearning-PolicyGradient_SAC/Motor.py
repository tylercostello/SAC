import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
from typing import Optional

import numpy as np
import random

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
        self.min_target = 0.1
        self.max_target = 0.85

        self.observation_shape = (2,)

        self.low_state = np.array(
            [self.min_position, self.min_target], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_target], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                            high=np.ones(self.observation_shape),
                                            dtype=np.float16)
        self.screen = None
        self.clock = None
        self.isopen = True
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,1), dtype=np.float32)


    def step(self, action, target):

        # if we took an action, we were in state 1

        state=np.array([0,0], dtype=float)
        action=abs(action)
        state[0] = action[0]
        reward=-1*abs((target-action))
        state[1]=target
        #print(state)
        #print(reward)

        self.steps_taken += 1
        #print(self.steps_taken)
        # if action == 1.0:
        #     reward = 1
        # else:
        #     reward = -1
        #print(reward)
        # regardless of the action, game is done after a single step

        if state[0]>=target-0.05 and state[0]<=target+0.05 :
            reward=5
            done = True
        else:
            done = False

        info = {}

        return state, reward, done, info

    def reset(self):
        state=np.array([0,0])
        self.steps_taken = 0
        return state

    def render(self, mode='human'):
        pass

    def close(self):
        pass