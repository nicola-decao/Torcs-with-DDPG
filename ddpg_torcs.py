import numpy as np
from keras.layers import Dense, Flatten, Input, merge
from keras.models import Model
from keras.optimizers import Adam

from kerasRL.rl.agents import DDPGAgent
from kerasRL.rl.memory import SequentialMemory
from kerasRL.rl.random import OrnsteinUhlenbeckProcess
from noises import ExplorationNoise
from rewards import DefaultReward
from torcs_gym import TorcsEnv

GAMMA = 0.99
TAU = 1e-3


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
    def __run(reward_writer, load=False, save=False, gui=True, load_file_path='', save_file_path='', timeout=10000,
              track='g-track-1',
              verbose=0, nb_steps=50000, nb_max_episode_steps=10000, train=False, epsilon=1.0, noise=1, limit_action=None, n_lap=None):

        env = TorcsEnv(gui=gui, timeout=timeout, track=track, reward=DefaultReward(), n_lap=n_lap)

        actor = DDPGTorcs.__get_actor(env)
        critic, action_input = DDPGTorcs.__get_critic(env)

        memory = SequentialMemory(limit=100000, window_length=1)

        random_process = ExplorationNoise(nb_steps=nb_steps,
                                          epsilon=epsilon,
                                          steer=OrnsteinUhlenbeckProcess(theta=0.6, mu=0, sigma=0.3),
                                          accel_brake=OrnsteinUhlenbeckProcess(theta=1.0, mu=0.5, sigma=0.3),
                                          noise=noise)

        agent = DDPGAgent(nb_actions=env.action_space.shape[0],
                          actor=actor, critic=critic,
                          critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                          random_process=random_process, gamma=GAMMA, target_model_update=TAU,
                          limit_action=limit_action)

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
            print('Noise:', random_process.get_noise())
            print('steps:', agent.step, '/', nb_steps, '-', nb_steps-agent.step)
            print('Saved!')

    @staticmethod
    def train(reward_writer, load=False, save=False, gui=True, load_file_path='', save_file_path='', timeout=10000,
              track='g-track-1',
              verbose=0, nb_steps=30000, nb_max_episode_steps=50000, epsilon=1.0, noise=1, action_limit_function=None, n_lap=None):

        DDPGTorcs.__run(reward_writer, load=load, save=save, gui=gui, load_file_path=load_file_path,
                        save_file_path=save_file_path,
                        timeout=timeout, track=track,
                        verbose=verbose, nb_steps=nb_steps, nb_max_episode_steps=nb_max_episode_steps, train=True,
                        epsilon=epsilon, noise=noise, limit_action=action_limit_function, n_lap=n_lap)

    @staticmethod
    def test(reward_writer, load_file_path, track='g-track-1', gui=True, nb_max_episode_steps=10000):
        return DDPGTorcs.__run(reward_writer, load=True, gui=gui, load_file_path=load_file_path, track=track,
                               epsilon=0, nb_max_episode_steps=nb_max_episode_steps, noise=1)
