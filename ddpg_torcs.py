
import numpy as np
from keras.layers import Dense, Flatten, Input, merge
from keras.models import Model
from keras.optimizers import Adam
from kerasRL.rl.agents import DDPGAgent
from kerasRL.rl.memory import SequentialMemory
from kerasRL.rl.random import OrnsteinUhlenbeckProcess
from torcs_gym import TorcsEnv

GAMMA = 0.99
EPSILON = 0.3
TAU = 1e-3


class DDPGTorcs:

    @staticmethod
    def __get_actor(env):
        observation_input = Input(shape=(1,) + env.observation_space.shape)
        h0 = Dense(300, activation='relu')(Flatten()(observation_input))
        h1 = Dense(600, activation='relu')(h0)
        Steering = Dense(1, activation='tanh')(h1)
        Acceleration = Dense(1, activation='sigmoid')(h1)
        Brake = Dense(1, activation='sigmoid')(h1)
        output = merge([Steering, Acceleration, Brake], mode='concat')
        return Model(input=observation_input, output=output)

    @staticmethod
    def __get_critic(env):
        action_input = Input(shape=(env.action_space.shape[0],))
        observation_input = Input(shape=(1,) + env.observation_space.shape)
        h0 = Dense(300, activation='relu')(merge([action_input, Flatten()(observation_input)], mode='concat'))
        h1 = Dense(300, activation='relu')(h0)
        output = Dense(1, activation='linear')(h1)
        return Model(input=[action_input, observation_input], output=output), action_input

    @staticmethod
    def export_dl4j(net, filename):
        r = []
        for w in net.get_weights():
            r += np.transpose(w).flatten().tolist()

        np.savetxt(filename, np.array(r))

    @staticmethod
    def __run(load=False, save=False, gui=True, file_path='', timeout=10000, track='g-track-1',
              verbose=0, nb_steps=50000, nb_max_episode_steps=10000, train=False):

        env = TorcsEnv(gui=gui, timeout=timeout, track=track)

        actor = DDPGTorcs.__get_actor(env)
        critic, action_input = DDPGTorcs.__get_critic(env)

        memory = SequentialMemory(limit=100000, window_length=1)

        random_process = ExplorationNoise(nb_steps=nb_steps,
                                          epsilon=0.5,
                                          steer=OrnsteinUhlenbeckProcess(theta=0.6, mu=0, sigma=0.1),
                                          accel=OrnsteinUhlenbeckProcess(theta=1.0, mu=0.8, sigma=0.1),
                                          brake=OrnsteinUhlenbeckProcess(theta=0.2, mu=-1.0, sigma=0.1))

        agent = DDPGAgent(nb_actions=env.action_space.shape[0],
                          actor=actor, critic=critic,
                          critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                          random_process=random_process, gamma=GAMMA, target_model_update=TAU, epsilon=EPSILON)

        agent.compile((Adam(lr=.0001, clipnorm=1.), Adam(lr=.001, clipnorm=1.)), metrics=['mae'])

        if load:
            agent.load_weights(file_path)

        if train:
            agent.fit(env, nb_steps=nb_steps, visualize=False, verbose=verbose,
                      nb_max_episode_steps=nb_max_episode_steps)
        else:
            agent.test(env, visualize=False)

        if save:
            agent.save_weights(file_path, overwrite=True)

    @staticmethod
    def train(load=False, save=False, gui=True, file_path='', timeout=10000, track='g-track-1',
              verbose=0, nb_steps=50000, nb_max_episode_steps=10000):

        DDPGTorcs.__run(load=load, save=save, gui=gui, file_path=file_path, timeout=timeout, track=track,
                        verbose=verbose, nb_steps=nb_steps, nb_max_episode_steps=nb_max_episode_steps, train=True)

    @staticmethod
    def test(file_path, track='g-track-1'):
        DDPGTorcs.__run(load=True, gui=True, file_path=file_path, track=track, nb_steps=1,
                        nb_max_episode_steps=int(1e08))


class ExplorationNoise:
    def __init__(self, nb_steps, epsilon, steer, accel, brake):
        self.__step = 1.0 / nb_steps
        self.__epsilon = epsilon
        self.__steer = steer
        self.__accel = accel
        self.__brake = brake
        self.__noise = 1

    def sample(self):
        self.__noise -= self.__step
        return self.__noise * self.__epsilon * np.array([self.__steer.sample()[0],
                                                         self.__accel.sample()[0],
                                                         self.__brake.sample()[0]])

if __name__ == "__main__":
    DDPGTorcs.train(load=True, gui=True, save=True, timeout=40000, file_path='trained_networks/weights.h5f', track='spring', verbose=0)
    #DDPGTorcs.test('trained_networks/weights.h5f')

