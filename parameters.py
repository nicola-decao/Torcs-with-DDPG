from keras.initializations import normal
from keras.layers import Dense, Input, merge
from keras.models import Model

from utilities.distributions import OrnstainUhlenbeck, BrownianMotion


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
        action[0] += self.__magnitude * self.__steering_noise.sample(action[0])
        action[1] += self.__magnitude * self.__acceleration_noise.sample(action[1])
        action[2] += self.__magnitude * self.__brake_noise.sample(action[2])

        # noise magnitude decay
        self.__samples += 1
        self.__magnitude = 1.0  # TODO implement a decay like 1/ math.log(self.__samples + 1)

        return action


class ModelTargetNeuralNetworkParams:
    def __init__(self, learning_rate, tau, net, output_size=None):
        self.LEARNING_RATE = learning_rate
        self.TAU = tau
        self.NET = net
        self.OUTPUT_SIZE = output_size


class DDPGParams:
    def __init__(self):
        self.__explorative_noise = ExplorativeNoise()

        self.BUFFER_SIZE = 200
        self.ACTOR_PARAMS = ModelTargetNeuralNetworkParams(learning_rate=0.0001, tau=0.001,
                                                           net=NetorksStructure.create_actor_net, output_size=3)
        self.CRITIC_PARAMS = ModelTargetNeuralNetworkParams(learning_rate=0.001, tau=0.001,
                                                            net=NetorksStructure.create_critic_net)
        self.NOISE_FUNCTION = self.__explorative_noise.add_noise
        self.REWARD_FUNCTION = self.__reward

    @staticmethod
    def __reward(state):
        return 0  # TODO implement a rewoard function
