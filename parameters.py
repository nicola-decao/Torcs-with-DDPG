import numpy as np

from keras.initializations import normal
from keras.layers import Dense, Input, merge
from keras.models import Model

from utilities.distributions import OrnstainUhlenbeck, BrownianMotion


class DataHandler:
    @staticmethod
    def encode_state_data(sensors):
        state = np.empty((1, 29))
        state[0, 0] = sensors['angle'] / np.pi
        state[0, 1:20] = np.array(sensors['track']) / 200.0
        state[0, 20] = sensors['trackPos']
        state[0, 21] = sensors['speedX'] / 300.0
        state[0, 22] = sensors['speedY'] / 300.0
        state[0, 23] = sensors['speedZ'] / 300.0
        state[0, 24:28] = np.array(sensors['wheelSpinVel']) / 100.0
        state[0, 28] = sensors['rpm'] / 10000.0
        return state

    @staticmethod
    def decode_action_data(actions_dic, actions_vec):
        actions_dic['steer'] = actions_vec[0, 0]
        actions_dic['accel'] = actions_vec[0, 1]
        actions_dic['brake'] = actions_vec[0, 2]


class NetorksStructure:
    @staticmethod
    def create_actor_net():
        state = Input(shape=[29])
        h0 = Dense(300, activation='relu')(state)
        h1 = Dense(600, activation='relu')(h0)
        steering = Dense(1, activation='tanh', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        acceleration = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        brake = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        return Model(input=state, output=merge([steering, acceleration, brake], mode='concat'))

    @staticmethod
    def create_critic_net():
        hidden_units_1 = 300
        hidden_units_2 = 600
        state = Input(shape=[29])
        action = Input(shape=[3])
        s1 = Dense(hidden_units_1, activation='relu')(state)
        a1 = Dense(hidden_units_2, activation='linear')(action)
        h1 = Dense(hidden_units_2, activation='linear')(s1)
        h2 = merge([h1, a1], mode='sum')
        h3 = Dense(hidden_units_2, activation='relu')(h2)
        return Model(input=[state, action], output=Dense(1, activation='linear')(h3))


class ExplorativeNoise:
    def __init__(self):
        self.__steering_noise = OrnstainUhlenbeck(theta=0.6, mu=0, sigma=0.3,
                                                  brownian_motion=BrownianMotion(delta=0.25, dt=0.1))
        self.__acceleration_noise = OrnstainUhlenbeck(theta=1.0, mu=0.45, sigma=0.1,
                                                      brownian_motion=BrownianMotion(delta=0.25, dt=0.1))
        self.__brake_noise = OrnstainUhlenbeck(theta=1.0, mu=-0.1, sigma=0.05,
                                               brownian_motion=BrownianMotion(delta=0.25, dt=0.1))
        self.__magnitude = 1.0
        self.__samples = 0

    def add_noise(self, action):
        action[0, 0] += self.__magnitude * self.__steering_noise.sample(action[0, 0])
        action[0, 1] += self.__magnitude * self.__acceleration_noise.sample(action[0, 1])
        action[0, 2] += self.__magnitude * self.__brake_noise.sample(action[0, 2])

        # noise magnitude decay
        self.__samples += 1
        self.__magnitude -= 1.0 / 100000 # TODO log didn't work: decrease too quickly
        return action


class ModelTargetNeuralNetworkParams:
    def __init__(self, learning_rate, tau, net):
        self.LEARNING_RATE = learning_rate
        self.TAU = tau
        self.NET = net


class DDPGParams:
    def __init__(self):
        self.__explorative_noise = ExplorativeNoise()

        self.BUFFER_SIZE = 10000
        self.BATCH_SIZE = 32
        self.STATE_SIZE = 29
        self.ACTION_SIZE = 3
        self.GAMMA = 0.99
        self.ACTOR_PARAMS = ModelTargetNeuralNetworkParams(learning_rate=0.0001, tau=0.001,
                                                           net=NetorksStructure.create_actor_net)
        self.CRITIC_PARAMS = ModelTargetNeuralNetworkParams(learning_rate=0.001, tau=0.001,
                                                            net=NetorksStructure.create_critic_net)
        self.NOISE_FUNCTION = self.__explorative_noise.add_noise
        self.REWARD_FUNCTION = self.__reward

    @staticmethod
    def __reward(state):
        return 0
