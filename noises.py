import numpy as np


class ExplorationNoise:
    def __init__(self, nb_steps, epsilon, steer, accel_brake, noise=1):
        self.__step = 1.0 / nb_steps
        self.__epsilon = epsilon
        self.__steer = steer
        self.__accel_brake = accel_brake
        self.__noise = noise

    def sample(self, state):
        self.__noise -= self.__step
        ab = self.__accel_brake.sample()[0]
        return self.__noise * self.__epsilon * np.array([self.__steer.sample()[0], ab])

    def get_noise(self):
        return self.__noise
