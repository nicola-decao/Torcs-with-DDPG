import numpy as np
import time


class ProgressiveSmoothingReward:
    def __init__(self, smoothing_factor=1e-05, max_smoothing=40, smoothing=1.0):
        self.__smoothing_factor = smoothing_factor
        self.__smoothing = smoothing
        self.__max_smoothing = max_smoothing
        self.__alpha = 1.0
        self.__theta_decay = 1e-6
        self.__previous_speed = 0

    def reward(self, observation):
        positioning_score = - observation['speedX']*(np.abs(observation['trackPos']) ** self.__smoothing) * (np.abs(np.sin(observation['angle'])))
        if self.__max_smoothing > self.__smoothing:
            self.__smoothing += self.__smoothing_factor

        speed_score = observation['speedX']

        r = positioning_score + speed_score
        self.__previous_speed = observation['speedX']
        return r


class HitReward:
    def __init__(self):
        self.__lastDFS = 0
        self.__last_diff = 0
        self.__last_time = 0
        self.__last_damage = 0

    def reward(self, sensors):
        current_time = time.time()
        if self.__last_time == 0:
            time_diff = 0
        else:
            time_diff = current_time - self.__last_time

        diff = (sensors['distFromStart'] - self.__lastDFS)
        damage_diff = sensors['damage'] - self.__last_damage
        self.__last_damage = sensors['damage']

        if diff > 10 or diff < 0:
            diff = self.__last_diff
        self.__last_diff = diff

        self.__lastDFS = sensors['distFromStart']
        if time_diff == 0:
            reward = 0
        elif np.abs(sensors['trackPos']) > 0.99:
            reward = -10000
        elif damage_diff > 0:
            reward = -1000*damage_diff
        elif sensors['speedX'] < 1:
            reward = -5
        else:
            reward = diff / time_diff * (
                np.cos(sensors['angle'])
                - np.abs(np.sin(sensors['angle']))
                - np.abs(sensors['trackPos']) ** 5)
        self.__last_time = current_time
        return reward
