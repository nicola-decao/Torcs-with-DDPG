import numpy as np
from keras.layers import Dense, Flatten, Input, merge
from keras.models import Model
from keras.optimizers import Adam
from kerasRL.rl.agents import DDPGAgent
from kerasRL.rl.memory import SequentialMemory
from kerasRL.rl.random import OrnsteinUhlenbeckProcess
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


class DDPGTorcs:
    @staticmethod
    def __get_actor(env):
        observation_input = Input(shape=(1,) + env.observation_space.shape)
        h0 = Dense(300, activation='relu')(Flatten()(observation_input))
        h1 = Dense(600, activation='relu')(h0)
        output = Dense(env.action_space.shape[0], activation='tanh')(h1)
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
              verbose=0, nb_steps=50000, nb_max_episode_steps=10000, train=False, epsilon=1.0):

        env = TorcsEnv(gui=gui, timeout=timeout, track=track)

        actor = DDPGTorcs.__get_actor(env)
        critic, action_input = DDPGTorcs.__get_critic(env)

        memory = SequentialMemory(limit=100000, window_length=1)

        random_process = ExplorationNoise(nb_steps=nb_steps,
                                          epsilon=0.3,
                                          steer=OrnsteinUhlenbeckProcess(theta=0.6, mu=0, sigma=0.3),
                                          accel_brake=OrnsteinUhlenbeckProcess(theta=1.0, mu=0.5, sigma=0.3))

        agent = DDPGAgent(nb_actions=env.action_space.shape[0],
                          actor=actor, critic=critic,
                          critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                          random_process=random_process, gamma=GAMMA, target_model_update=TAU, epsilon=epsilon)

        agent.compile((Adam(lr=.0001, clipnorm=1.), Adam(lr=.001, clipnorm=1.)), metrics=['mae'])

        if load:
            agent.load_weights(file_path)

        if train:
            agent.fit(env, nb_steps=nb_steps, visualize=False, verbose=verbose,
                      nb_max_episode_steps=nb_max_episode_steps)
        else:
            agent.test(env, visualize=False)

        if save:
            print('Saving..')
            agent.save_weights(file_path, overwrite=True)
            print('Saved!')

    @staticmethod
    def train(load=False, save=False, gui=True, file_path='', timeout=10000, track='g-track-1',
              verbose=0, nb_steps=50000, nb_max_episode_steps=10000, epsilon=1.0):

        DDPGTorcs.__run(load=load, save=save, gui=gui, file_path=file_path, timeout=timeout, track=track,
                        verbose=verbose, nb_steps=nb_steps, nb_max_episode_steps=nb_max_episode_steps, train=True,
                        epsilon=epsilon)

    @staticmethod
    def test(file_path, track='g-track-1', epsilon=1.0):
        DDPGTorcs.__run(load=True, gui=True, file_path=file_path, track=track, nb_steps=1,
                        nb_max_episode_steps=int(1e08), epsilon=epsilon)


class ExplorationNoise:
    def __init__(self, nb_steps, epsilon, steer, accel_brake):
        self.__step = 1.0 / nb_steps
        self.__epsilon = epsilon
        self.__steer = steer
        self.__accel_brake = accel_brake
        self.__noise = 1

    def sample(self):
        self.__noise -= self.__step
        return self.__noise * self.__epsilon * np.array([self.__steer.sample()[0], self.__accel_brake.sample()[0]])


if __name__ == "__main__":

    DDPGTorcs.train(load=False, gui=True, save=True,
                    file_path='trained_networks/reward_test.h5f', verbose=1, timeout=40000)
    # DDPGTorcs.test('trained_networks/weights.h5f')

    # epsilon = 1.0

    # tracks = [track for track in TRACK_LIST]
    # np.random.shuffle(tracks)
    # epsilon = 1.0
    #
    # for i, track in enumerate(tracks):
    #     DDPGTorcs.train(load=True, gui=True, save=True, file_path='trained_networks/gear_test01.h5f',
    #                     track=track, verbose=0, epsilon=epsilon)
    #     epsilon = 1.0 - float(i + 1) / len(tracks)
