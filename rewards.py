import numpy as np


class DefaultReward:
    def reward(self, sensors):

        damage = sensors['damage']
        angle = sensors['angle']
        speed_x = sensors['speedX']
        abs_speed_y = np.abs(sensors['speedY'])
        track_pos = sensors['trackPos']
        abs_track_pos = np.abs(track_pos)
        cosine = np.cos(angle)
        abs_sine = np.abs(np.sin(angle))

        if track_pos > 0.99 or damage > 0:
            reward = -500
        else:
            reward = (speed_x - abs_speed_y) * (
                cosine
                - abs_sine
                - abs_track_pos)
        return reward

    def get_minimum_reward(self):
        return -500


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
        self.__exit_reward = -10000
        self.__idle_reward = -5

    def get_minimum_reward(self):
        return self.__exit_reward

    def reward(self, sensors):
        damage = sensors['damage']
        angle = sensors['angle']
        speed = sensors['speedX']
        track_pos = sensors['trackPos']
        abs_track_pos = np.abs(track_pos)

        if abs_track_pos > 0.99 or damage > 0:
            reward = min(self.__exit_reward + sensors['distRaced'], 0)
        elif speed < 5:
            reward = self.__idle_reward
        else:
            cosine = np.cos(angle)
            abs_sine = np.abs(np.sin(angle))

            reward = 0.1 * speed * (cosine - abs_sine - abs_track_pos)
        return reward
