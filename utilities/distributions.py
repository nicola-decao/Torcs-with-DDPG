from scipy.stats import norm
import numpy as np


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

