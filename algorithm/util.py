
import random
import numpy as np
from collections import deque
from scipy.stats import norm


class OriginalRandom:

    @staticmethod
    def sample():
        return np.random.randn(1)


class BrownianMotion:

    def __init__(self, delta, dt):
        self.__x, self.__delta, self.__dt = 0.0, delta, dt

    def sample(self):
        self.__x += norm.rvs(scale=self.__delta ** 2 * self.__dt)
        return self.__x


class OrnstainUhlenbeck:

    def __init__(self, theta, mu, sigma, brownian_motion):
        self.__theta, self.__mu, self.__sigma, self.__brownian_motion = theta, mu, sigma, brownian_motion

    def sample(self, x):
        return self.__theta * (self.__mu - x) + self.__sigma * self.__brownian_motion.sample()


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

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
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
