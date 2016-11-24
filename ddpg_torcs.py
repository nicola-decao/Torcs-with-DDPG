import json
import os
import time

import numpy as np
from keras.layers import Dense, Flatten, Input, merge
from keras.models import Model
from keras.optimizers import Adam

from kerasRL.rl.agents import DDPGAgent
from kerasRL.rl.memory import SequentialMemory
from kerasRL.rl.random import OrnsteinUhlenbeckProcess
from reward_writer import RewardWriter
from torcs_gym import TorcsEnv, TRACK_LIST

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
        elif sensors['damage'] > 0:
            reward = -100
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
        h0 = Dense(200, activation='relu', init='he_normal')(Flatten()(observation_input))
        h1 = Dense(200, activation='relu', init='he_normal')(h0)
        output = Dense(env.action_space.shape[0], activation='tanh', init='he_normal')(h1)
        return Model(input=observation_input, output=output)

    @staticmethod
    def __get_critic(env):
        action_input = Input(shape=(env.action_space.shape[0],))
        observation_input = Input(shape=(1,) + env.observation_space.shape)

        w1 = Dense(100, activation='relu', init='he_normal')(Flatten()(observation_input))
        a1 = Dense(100, activation='linear', init='he_normal')(action_input)
        h1 = Dense(100, activation='linear', init='he_normal')(w1)
        h2 = merge([h1, a1], mode='sum')
        h3 = Dense(100, activation='relu', init='he_normal')(h2)
        output = Dense(1, activation='linear', init='he_normal')(h3)
        return Model(input=[action_input, observation_input], output=output), action_input

    @staticmethod
    def export_dl4j(net, filename):
        r = []
        for w in net.get_weights():
            r += np.transpose(w).flatten().tolist()

        np.savetxt(filename, np.array(r))

    @staticmethod
    def __run(reward_writer, load=False, save=False, gui=True, load_file_path='', save_file_path='', timeout=10000, track='g-track-1',
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

        agent.compile((Adam(lr=.0001, clipnorm=1.), Adam(lr=.001, clipnorm=1.)), metrics=['mae'])
        if load:
            agent.load_weights(load_file_path)
        if train:
            agent.fit(env, reward_writer, nb_steps=nb_steps, visualize=False, verbose=verbose,
                      nb_max_episode_steps=nb_max_episode_steps)
        else:
            agent.test(env, visualize=False, nb_max_episode_steps=nb_max_episode_steps)
            return env.did_one_lap()

        if save:
            print('Saving..')
            agent.save_weights(save_file_path, overwrite=True)
            print('Saved!')

    @staticmethod
    def train(reward_writer, load=False, save=False, gui=True, load_file_path='', save_file_path='', timeout=10000, track='g-track-1',
              verbose=0, nb_steps=30000, nb_max_episode_steps=10000, epsilon=1.0):

        DDPGTorcs.__run(reward_writer, load=load, save=save, gui=gui, load_file_path=load_file_path, save_file_path=save_file_path,
                        timeout=timeout, track=track,
                        verbose=verbose, nb_steps=nb_steps, nb_max_episode_steps=nb_max_episode_steps, train=True,
                        epsilon=epsilon)

    @staticmethod
    def test(load_file_path, track='g-track-1', gui=False, nb_max_episode_steps=10000):
        return DDPGTorcs.__run(load=True, gui=gui, load_file_path=load_file_path, track=track,
                               epsilon=0, nb_max_episode_steps=nb_max_episode_steps)


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
            ab /= (1 + state[0, 21] / 300)
        else:
            ab *= state[0, 21] / 300
        return self.__noise * self.__epsilon * np.array([self.__steer.sample()[0] * (1 - state[0, 21] / 300), ab])


def create_complete_tracks_list(epsilons):
    tracks = {}
    for epsilon in epsilons:
        tracks[str(epsilon)] = []
    for epsilon in epsilons:
        for track in TRACK_LIST.keys():
            if TRACK_LIST[track] != 'dirt':
                if track != 'b-speedway' and track != 'c-speedway' and track != 'd-speedway' and track != 'e-speedway' and track != 'f-speedway' and track != 'g-speedway':
                    tracks[str(epsilon)].append(track)
    return tracks


def save_remaining_tracks(tracks, filepath):
    with open(filepath, 'w+') as f:
        json.dump(tracks, f, sort_keys=True, indent=4)


def load_tracks(track_filename):
    if os.path.isfile(track_filename):
        with open(track_filename, 'r') as f:
            return json.load(f)
    else:
        return False


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


def train_on_all_tracks():
    # This is used if you want to restart everything but you want to have a trained network at the start
    start_with_trained_network = False

    epsilons = [0.5, 0.1, 0]
    tracks_to_test = 'tracks_to_test.json'
    file_path = 'trained_networks/test_'
    last_network_file_path = 'trained_networks/last_network.txt'
    tracks_to_test_filepath = 'tracks_to_test.json'

    reward_writer = RewardWriter('rewards.csv')

    tracks = load_tracks(tracks_to_test)
    if not tracks:
        tracks = create_complete_tracks_list(epsilons)
    order_tracks(tracks)

    # load the right network
    if start_with_trained_network:
        load_file_path = 'trained_networks/pre_trained.h5f'
        i = 0
    else:
        load_file_path, i = load_last_network_path(last_network_file_path)

    save_file_path = load_file_path

    for epsilon in epsilons:
        while len(tracks[str(epsilon)]) > 0:
            track = tracks[str(epsilon)][0]

            if i != 0:
                load_file_path = save_file_path
            save_file_path = file_path + str(i) + '_' + track + '_' + str(epsilon) + '.h5f'

            print('Track name:', track)
            print('Epsilon:', epsilon)
            print()

            # write track name
            reward_writer.write_track(track, epsilon)

            try:
                DDPGTorcs.train(reward_writer, load=True, gui=True, save=True, track=track, nb_steps=100000,
                                load_file_path=load_file_path, save_file_path=save_file_path, verbose=1, timeout=40000,
                                epsilon=epsilon)

                i += 1
                tracks[str(epsilon)].remove(track)
                save_remaining_tracks(tracks, tracks_to_test_filepath)
                save_last_network_path(last_network_file_path, save_file_path, i)

                reward_writer.completed_track()
            except:
                # Torcs fucked up, so now we fix everything
                save_file_path = load_file_path
                reward_writer.bad_run()
                reward_writer.completed_track()


def train_on_single_track(track):
    epsilon = 0.5
    gui = False
    load = True
    save = True
    steps = 500000

    if load:
        load_file_path = 'trained_networks/single_track.h5f'
    else:
        load_file_path = ''
    save_file_path = 'trained_networks/single_track.h5f'

    DDPGTorcs.train(load=load, gui=gui, save=save, track=track, nb_steps=steps, load_file_path=load_file_path,
                    save_file_path=save_file_path, verbose=1, timeout=40000, epsilon=epsilon)


def create_tracks_list(chosen_tracks, epsilons):
    tracks = {}
    for epsilon in epsilons:
        tracks[str(epsilon)] = []
        for track in chosen_tracks:
            tracks[str(epsilon)].append(track)
    return tracks


def train_on_chosen_tracks(chosen_tracks, epsilons, steps, root_dir):
    root_dir = 'runs/' + root_dir + '/'

    rewards_filepath = root_dir + 'rewards.csv'
    remaining_tracks_filepath = root_dir + 'tracks_to_test.json'
    last_network_filepath = root_dir + 'last_network.txt'

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        load_filepath = ''
        i = 0
    else:
        load_filepath, i = load_last_network_path(last_network_filepath)

    reward_writer = RewardWriter(rewards_filepath)

    tracks = load_tracks(remaining_tracks_filepath)
    if not tracks:
        tracks = create_tracks_list(chosen_tracks, epsilons)
    order_tracks(tracks)

    save_filepath = load_filepath

    for epsilon in epsilons:
        while len(tracks[str(epsilon)]) > 0:
            track = tracks[str(epsilon)][0]

            load_filepath = save_filepath
            save_filepath = root_dir + str(i) + '_' + track + '_' + str(epsilon) + '.h5f'

            reward_writer.write_track(track, epsilon)

            print('Track name:', tracks[str(epsilon)][0])
            print('Epsilon:', epsilon)

            DDPGTorcs.train(reward_writer, load=True, gui=True, save=True, track=track,
                            nb_steps=steps, load_file_path=load_filepath, save_file_path=save_filepath,
                            verbose=1, timeout=40000, epsilon=epsilon)

            tracks[str(epsilon)].remove(track)
            i += 1
            save_remaining_tracks(tracks, remaining_tracks_filepath)
            save_last_network_path(last_network_filepath, save_filepath, i)
            reward_writer.completed_track()
            print()
            print()


if __name__ == "__main__":
    # train_on_all_tracks()
    #train_on_single_track('aalborg')
    train_on_chosen_tracks(['aalborg', 'a-speedway', 'e-track-6'], [0.5, 0.2, 0], 100000, 'three_tracks')
