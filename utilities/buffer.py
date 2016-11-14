import random
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.__buffer_size, self.__index, self.__buffer = buffer_size, 0, deque()

    def get_batch(self, batch_size):
        if self.__index < batch_size:
            return random.sample(self.__buffer, self.__index)
        else:
            return random.sample(self.__buffer, batch_size)

    def size(self):
        return self.__buffer_size

    def add(self, state, action, reward, new_state):
        experience = (state, action, reward, new_state)
        if self.__index < self.__buffer_size:
            self.__buffer.append(experience)
            self.__index += 1
        else:
            self.__buffer.popleft()
            self.__buffer.append(experience)

    def count(self):
        return self.__index

    def erase(self):
        self.__buffer = deque()
        self.__index = 0
