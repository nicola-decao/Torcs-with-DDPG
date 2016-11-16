import threading

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, dim_state, dim_action):
        self.__index = 0
        self.__full = False
        self.__buffer_size = buffer_size
        self.__dim_state = dim_state
        self.__dim_action = self.__dim_state + dim_action
        self.__dim_reward = self.__dim_action + 1
        self.__dim_new_state = self.__dim_reward + dim_state
        self.__dim_terminal = self.__dim_new_state + 1
        self.__buffer = np.empty((buffer_size, 2 * dim_state + dim_action + 2))
        self.__mutex = threading.Semaphore()

    def get_batch(self, batch_size):
        if self.__full:
            rnd = np.random.choice(self.__buffer_size, batch_size, replace=False)
        elif self.__index >= batch_size:
            rnd = np.random.choice(self.__index, batch_size, replace=False)
        else:
            rnd = np.random.choice(self.__index, self.__index, replace=False)

        self.__mutex.acquire()
        result = self.__buffer[rnd, 0:self.__dim_state], \
               self.__buffer[rnd, self.__dim_state:self.__dim_action], \
               self.__buffer[rnd, self.__dim_action:self.__dim_reward], \
               self.__buffer[rnd, self.__dim_reward:self.__dim_new_state], \
               self.__buffer[rnd, self.__dim_new_state:self.__dim_terminal]
        self.__mutex.release()

        return result

    def add(self, state, action, reward, new_state, terminal):
        self.__mutex.acquire()
        self.__buffer[self.__index, 0:self.__dim_state] = state
        self.__buffer[self.__index, self.__dim_state:self.__dim_action] = action
        self.__buffer[self.__index, self.__dim_action:self.__dim_reward] = reward
        self.__buffer[self.__index, self.__dim_reward:self.__dim_new_state] = new_state
        self.__buffer[self.__index, self.__dim_new_state:self.__dim_terminal] = terminal

        if self.__index < self.__buffer_size - 1:
            self.__index += 1
        else:
            self.__full = True
            self.__index = 0
        self.__mutex.release()

    def count(self):
        self.__mutex.acquire()
        result = self.__buffer_size if self.__full else self.__index
        self.__mutex.release()
        return result

    def is_empty(self):
        self.__mutex.acquire()
        result = self.__index == 0 and not self.__full
        self.__mutex.release()
        return result




