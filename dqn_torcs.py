import numpy as np
from keras.layers import Dense, Flatten, Input, merge, Activation
from keras.models import Model, Sequential
from keras.optimizers import Adam
import os

from kerasRL.rl.agents import ContinuousDQNAgent
from kerasRL.rl.agents import DDPGAgent
from kerasRL.rl.core import Processor
from kerasRL.rl.memory import SequentialMemory
from kerasRL.rl.random import OrnsteinUhlenbeckProcess
from noises import ExplorationNoise
from rewards import DefaultReward

GAMMA = 0.99
TAU = 1e-3

class CDQNTorcs:
    @staticmethod
    def __get_mu_model(observation_shape, action_shape):
        mu_model = Sequential()
        mu_model.add(Flatten(input_shape=(1,) + observation_shape))
        mu_model.add(Dense(100))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(100))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(100))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(action_shape[0]))
        mu_model.add(Activation('linear'))
        return mu_model

    @staticmethod
    def __get_V_model(observation_shape):
        V_model = Sequential()
        V_model.add(Flatten(input_shape=(1,) + observation_shape))
        V_model.add(Dense(100))
        V_model.add(Activation('relu'))
        V_model.add(Dense(100))
        V_model.add(Activation('relu'))
        V_model.add(Dense(100))
        V_model.add(Activation('relu'))
        V_model.add(Dense(1))
        V_model.add(Activation('linear'))
        return V_model

    @staticmethod
    def __get_L_model(observation_shape, action_shape):
        action_input = Input(shape=action_shape, name='action_input')
        observation_input = Input(shape=(1,) + observation_shape, name='observation_input')
        x = merge([action_input, Flatten()(observation_input)], mode='concat')
        x = Dense(200)(x)
        x = Activation('relu')(x)
        x = Dense(200)(x)
        x = Activation('relu')(x)
        x = Dense(200)(x)
        x = Activation('relu')(x)
        x = Dense(((action_shape[0] * action_shape[0] + action_shape[0]) / 2))(x)
        x = Activation('linear')(x)
        L_model = Model(input=[action_input, observation_input], output=x)
        return L_model

    @staticmethod
    def __run(reward_writer, load=False, save=False, gui=True, load_file_path='', save_file_path='', timeout=10000,
              track='g-track-1',
              verbose=0, nb_steps=50000, nb_max_episode_steps=10000, train=False, epsilon=1.0, noise=1, limit_action=None, n_lap=None):
        from torcs_gym import TorcsEnv
        env = TorcsEnv(gui=gui, timeout=timeout, track=track, reward=DefaultReward(), n_lap=n_lap)

        mu_model = CDQNTorcs.__get_mu_model(env.observation_space.shape, env.action_space.shape)
        V_model = CDQNTorcs.__get_V_model(env.observation_space.shape)
        L_model = CDQNTorcs.__get_L_model(env.observation_space.shape, env.action_space.shape)

        memory = SequentialMemory(limit=100000, window_length=1)

        random_process = ExplorationNoise(nb_steps=nb_steps,
                                          epsilon=epsilon,
                                          steer=OrnsteinUhlenbeckProcess(theta=0.6, mu=0, sigma=0.3),
                                          accel_brake=OrnsteinUhlenbeckProcess(theta=1.0, mu=0.5, sigma=0.3),
                                          noise=noise)

        agent = ContinuousDQNAgent(nb_actions=env.action_space.shape[0], V_model=V_model, L_model=L_model, mu_model=mu_model,
                                   memory=memory, nb_steps_warmup=100, random_process=random_process,
                                   gamma=GAMMA, target_model_update=0.0001)
        agent.compile(Adam(lr=0.0001, clipnorm=1.), metrics=['mae'])
        #agent.load_weights(load_file_path)
        # if load:
        #     agent.load_weights(load_file_path)
        # if train:
        #     agent.fit(env, reward_writer, nb_steps=nb_steps, visualize=False, verbose=verbose,
        #               nb_max_episode_steps=nb_max_episode_steps)
        # else:
        #     agent.test(env, visualize=False, nb_max_episode_steps=nb_max_episode_steps)
        #     return env.did_one_lap()

        # if save:
        #     print('Saving..')
        #     agent.save_weights(save_file_path, overwrite=True)
        #     print('Noise:', random_process.get_noise())
        #     print('steps:', agent.step, '/', nb_steps, '-', nb_steps-agent.step)
        #     print('Saved!')
        agent.fit(env, reward_writer, nb_steps=nb_steps, visualize=False, verbose=1,
                  nb_max_episode_steps=nb_max_episode_steps)

    @staticmethod
    def train(reward_writer, load=False, save=False, gui=True, load_file_path='', save_file_path='', timeout=10000,
              track='g-track-1',
              verbose=0, nb_steps=300000, nb_max_episode_steps=50000, epsilon=1.0, noise=1, action_limit_function=None, n_lap=None):

        CDQNTorcs.__run(reward_writer, load=load, save=save, gui=gui, load_file_path=load_file_path,
                        save_file_path=save_file_path,
                        timeout=timeout, track=track,
                        verbose=verbose, nb_steps=nb_steps, nb_max_episode_steps=nb_max_episode_steps, train=True,
                        epsilon=epsilon, noise=noise, limit_action=action_limit_function, n_lap=n_lap)

    @staticmethod
    def test(reward_writer, load_file_path, track='g-track-1', gui=True, nb_max_episode_steps=10000):
        return CDQNTorcs.__run(reward_writer, load=True, gui=gui, load_file_path=load_file_path, track=track,
                               epsilon=0, nb_max_episode_steps=nb_max_episode_steps, noise=1)

CDQNTorcs().train(None)

