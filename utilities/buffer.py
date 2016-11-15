import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, dim_state, dim_action):
        self.__dim_state = dim_state
        self.__dim_action = self.__dim_state + dim_action
        self.__dim_reward = self.__dim_action + 1
        self.__dim_new_state = self.__dim_reward + dim_state
        self.__index = 0
        self.__full = False
        self.__buffer_size = buffer_size
        self.__buffer = np.empty((buffer_size, 2 * dim_state + dim_action + 1))

    def get_batch(self, batch_size):
        if self.__full:
            rnd = np.random.choice(self.__buffer_size, batch_size, replace=False)
        elif self.__index >= batch_size:
            rnd = np.random.choice(self.__index, batch_size, replace=False)
        else:
            rnd = np.random.choice(self.__index, self.__index, replace=False)

        return self.__buffer[rnd, 0:self.__dim_state], \
               self.__buffer[rnd, self.__dim_state:self.__dim_action], \
               self.__buffer[rnd, self.__dim_action:self.__dim_reward], \
               self.__buffer[rnd, self.__dim_reward:self.__dim_new_state]

    def add(self, state, action, reward, new_state):
        self.__buffer[self.__index, 0:self.__dim_state] = state
        self.__buffer[self.__index, self.__dim_state:self.__dim_action] = action
        self.__buffer[self.__index, self.__dim_action:self.__dim_reward] = reward
        self.__buffer[self.__index, self.__dim_reward:self.__dim_new_state] = new_state

        if self.__index < self.__buffer_size:
            self.__index += 1
        else:
            self.__full = True
            self.__index = 0
