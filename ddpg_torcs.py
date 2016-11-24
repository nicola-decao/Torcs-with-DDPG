import json

import numpy as np
import time
from keras.layers import Dense, Flatten, Input, merge
from keras.models import Model
from keras.optimizers import Adam
from kerasRL.rl.agents import DDPGAgent
from kerasRL.rl.memory import SequentialMemory
from kerasRL.rl.random import OrnsteinUhlenbeckProcess
from torcs_gym import TorcsEnv, TRACK_LIST
import os

GAMMA = 0.99
TAU = 1e-3


class ProgressiveSmoothingReward:
    def __init__(self, smoothing_factor=1e-05, max_smoothing=40, smoothing=1.0):
        self.__smoothing_factor = smoothing_factor
        self.__smoothing = smoothing
        self.__max_smoothing = max_smoothing
        self.__alpha = 1.0
        self.__theta_decay = 1e-6
        self.__previous_speed = 0

    def reward(self, observation):
        positioning_score = - (np.abs(observation[20]) ** self.__smoothing) * (- np.abs(np.sin(observation[0])))
        if self.__max_smoothing > self.__smoothing:
            self.__smoothing += self.__smoothing_factor

        speed_score = observation[21]

        r = positioning_score + speed_score
        self.__previous_speed = observation[21]
        return r


class HitReward:
    def __init__(self):
        self.__lastDFS = 0
        self.__last_diff = 0
        self.__last_time = 0

    def reward(self, sensors):
        current_time = time.time()
        if self.__last_time == 0:
            time_diff = 0
        else:
            time_diff = current_time - self.__last_time

        diff = (sensors['distFromStart'] - self.__lastDFS)
        if diff > 10 or diff < 0:
            diff = self.__last_diff
        self.__last_diff = diff

        self.__lastDFS = sensors['distFromStart']
        if time_diff == 0:
            reward = 0
        elif np.abs(sensors['trackPos']) > 0.99:
            reward = -200
        elif sensors['speedX'] < 1:
            reward = -2
        else:
            reward = diff / time_diff * (
                np.cos(sensors['angle'])
                - np.abs(np.sin(sensors['angle']))
                - np.abs(sensors['trackPos']) ** 5)
        self.__last_time = current_time
        return reward


class DDPGTorcs:
    @staticmethod
    def __get_actor(env):
        observation_input = Input(shape=(1,) + env.observation_space.shape)
        h0 = Dense(300, activation='relu', init='he_normal')(Flatten()(observation_input))
        h1 = Dense(600, activation='relu', init='he_normal')(h0)
        output = Dense(env.action_space.shape[0], activation='tanh', init='he_normal')(h1)
        return Model(input=observation_input, output=output)

    @staticmethod
    def __get_critic(env):
        action_input = Input(shape=(env.action_space.shape[0],))
        observation_input = Input(shape=(1,) + env.observation_space.shape)
        h0 = Dense(300, activation='relu', init='he_normal')(merge([action_input, Flatten()(observation_input)], mode='concat'))
        h1 = Dense(300, activation='relu', init='he_normal')(h0)
        output = Dense(1, activation='linear', init='he_normal')(h1)
        return Model(input=[action_input, observation_input], output=output), action_input

    @staticmethod
    def export_dl4j(net, filename):
        r = []
        for w in net.get_weights():
            r += np.transpose(w).flatten().tolist()

        np.savetxt(filename, np.array(r))

    @staticmethod
    def __run(load=False, save=False, gui=True, load_file_path='', save_file_path='', timeout=10000, track='g-track-1',
              verbose=0, nb_steps=50000, nb_max_episode_steps=10000, train=False, epsilon=1.0):

        env = TorcsEnv(gui=gui, timeout=timeout, track=track, reward=HitReward().reward)

        actor = DDPGTorcs.__get_actor(env)
        critic, action_input = DDPGTorcs.__get_critic(env)

        memory = SequentialMemory(limit=100000, window_length=1)

        random_process = ExplorationNoise(nb_steps=nb_steps,
                                          epsilon=epsilon,
                                          steer=OrnsteinUhlenbeckProcess(theta=0.6, mu=0, sigma=0.3),
                                          accel_brake=OrnsteinUhlenbeckProcess(theta=1.0, mu=0.5, sigma=0.3))

        agent = DDPGAgent(nb_actions=env.action_space.shape[0],
                          actor=actor, critic=critic,
                          critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                          random_process=random_process, gamma=GAMMA, target_model_update=TAU)

        agent.compile((Adam(lr=.0001, clipnorm=1.), Adam(lr=.001, clipnorm=1.)), metrics=['mse'])

        if load:
            agent.load_weights(load_file_path)

        if train:
            agent.fit(env, nb_steps=nb_steps, visualize=False, verbose=verbose,
                      nb_max_episode_steps=nb_max_episode_steps)
        else:
            agent.test(env, visualize=False)

        if save:
            print('Saving..')
            agent.save_weights(save_file_path, overwrite=True)
            print('Saved!')

    @staticmethod
    def train(load=False, save=False, gui=True, load_file_path='', save_file_path='', timeout=10000, track='g-track-1',
              verbose=0, nb_steps=30000, nb_max_episode_steps=10000, epsilon=1.0):

        DDPGTorcs.__run(load=load, save=save, gui=gui, load_file_path=load_file_path, save_file_path=save_file_path,
                        timeout=timeout, track=track,
                        verbose=verbose, nb_steps=nb_steps, nb_max_episode_steps=nb_max_episode_steps, train=True,
                        epsilon=epsilon)

    @staticmethod
    def test(load_file_path, track='g-track-1', epsilon=1.0):
        DDPGTorcs.__run(load=True, gui=True, load_file_path=load_file_path, track=track, nb_steps=1,
                        nb_max_episode_steps=int(1e08), epsilon=epsilon)


class ExplorationNoise:
    def __init__(self, nb_steps, epsilon, steer, accel_brake):
        self.__step = 1.0 / nb_steps
        self.__epsilon = epsilon
        self.__steer = steer
        self.__accel_brake = accel_brake
        self.__noise = 1

    def sample(self, state):
        self.__noise -= self.__step
        ab = self.__accel_brake.sample()[0]
        if ab >= 0:
            ab /= 1 + state[0, 20] / 200
        else:
            ab *= state[0, 20] / 200
        return self.__noise * self.__epsilon * np.array([self.__steer.sample()[0] * (1 - state[0, 20] / 200), ab])


def create_tracks_list(epsilons):
    tracks = {}
    for epsilon in epsilons:
        tracks[str(epsilon)] = []
    for epsilon in epsilons:
        for track in TRACK_LIST.keys():
            if TRACK_LIST[track] != 'dirt':
                if track != 'b-speedway' and track != 'c-speedway' and track != 'd-speedway' and track != 'e-speedway' and track != 'f-speedway' and track != 'g-speedway':
                    tracks[str(epsilon)].append(track)
    return tracks


def save_remaining_tracks(tracks):
    with open('tracks_to_test.json', 'w+') as f:
        json.dump(tracks, f, sort_keys=True, indent=4)


def load_tracks(track_filename):
    if os.path.isfile(track_filename):
        with open(track_filename, 'r') as f:
            return json.load(f)
    else:
        return create_tracks_list(epsilons)


def load_last_network_path(track_filename):
    if os.path.isfile(track_filename):
        with open(track_filename) as f:
            network_name = f.readline().replace('\n', '')
            i = f.readline()
            return network_name, int(i)
    else:
        return '', 0


def save_last_network_path(last_network_file_path, save_file_path, i):
    with open(last_network_file_path, 'w+') as f:
        print(save_file_path, file=f)
        print(str(i), file=f)


def order_tracks(tracks):
    for key in tracks.keys():
        tracks[key].sort()


if __name__ == "__main__":
    # This is used if you want to restart everything but you want to have a trained network at the start
    start_with_trained_network = False

    epsilons = [0.5, 0.1, 0]
    tracks_to_test = 'tracks_to_test.json'
    file_path = 'trained_networks/test_'
    last_network_file_path = 'trained_networks/last_network.txt'

    tracks = load_tracks(tracks_to_test)
    order_tracks(tracks)

    save_file_path = ''
    load_file_path = ''

    # load the right network
    if start_with_trained_network:
        load_file_path = 'trained_networks/pre_trained.h5f'
        i = 0
    else:
        load_file_path, i = load_last_network_path(last_network_file_path)

    save_file_path = load_file_path

    for epsilon in epsilons:
        while len(tracks[str(epsilon)]) > 0:
            if i != 0:
                load_file_path = save_file_path
            save_file_path = file_path + str(i) + '_' + tracks[str(epsilon)][0] + '_' + str(epsilon) + '.h5f'

            print('track: ' + tracks[str(epsilon)][0])

            # write track name
            with open('rewards.csv', 'a') as f:
                print(tracks[str(epsilon)][0], file=f)

            try:
                DDPGTorcs.train(load=True, gui=True, save=True, track=tracks[str(epsilon)][0], nb_steps=100000,
                            load_file_path=load_file_path, save_file_path=save_file_path, verbose=1, timeout=40000,
                            epsilon=epsilon)

                i += 1
                tracks[str(epsilon)].remove(tracks[str(epsilon)][0])
                save_remaining_tracks(tracks)
                save_last_network_path(last_network_file_path, save_file_path, i)

                with open('rewards.csv', 'a') as f:
                    print('', file=f)
            except:
                # Torcs fucked up, so now we fix everything
                save_file_path = load_file_path
                with open('rewards.csv', 'a') as f:
                    print('BAD RUN BAD RUN BAD RUN BAD RUN', file=f)
                    print('', file=f)
