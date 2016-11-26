import numpy as np


class DefaultReward:
    def reward(self, sensors):
        if np.abs(sensors['trackPos']) > 0.99:
            reward = -200
        else:
            reward = sensors['speedX'] * (
                np.cos(sensors['angle'])
                - np.abs(np.sin(sensors['angle']))
                - np.abs(sensors['trackPos']))
        return reward

    def get_minimum_reward(self):
        return -200


class ProgressiveSmoothingReward:
    def __init__(self, smoothing_factor=1e-05, max_smoothing=40, smoothing=1.0):
        self.__smoothing_factor = smoothing_factor
        self.__smoothing = smoothing
        self.__max_smoothing = max_smoothing
        self.__alpha = 1.0
        self.__theta_decay = 1e-6
        self.__previous_speed = 0

    def reward(self, observation):
        positioning_score = - observation['speedX'] * (np.abs(observation['trackPos']) ** self.__smoothing) * (
        np.abs(np.sin(observation['angle'])))
        if self.__max_smoothing > self.__smoothing:
            self.__smoothing += self.__smoothing_factor

        speed_score = observation['speedX']

        r = positioning_score + speed_score
        self.__previous_speed = observation['speedX']
        return r


class HitReward:
    def __init__(self):
        self.__last_damage = 0
        self.__exit_reward = -10000
        self.__damage_reward = -1000
        self.__idle_reward = -5

    def get_minimum_reward(self):
        return self.__exit_reward

    def reward(self, sensors):
        damage_diff = sensors['damage'] - self.__last_damage
        self.__last_damage = sensors['damage']
        angle = sensors['angle']
        speed = sensors['speedX']

        if np.abs(sensors['trackPos']) > 0.99:
            reward = self.__exit_reward
        elif damage_diff > 0:
            reward = self.__damage_reward * damage_diff
        elif speed < 1:
            reward = self.__idle_reward
        else:
            cosine = np.cos(angle)
            abs_sine = np.abs(np.sin(angle))
            abs_track_pos = np.abs(sensors['trackPos']) ** 10

            assert -1 <= cosine <= 1 and abs_sine <= 1 and abs_track_pos < 1

            reward = speed * (cosine - abs_sine - abs_track_pos)
        return reward
