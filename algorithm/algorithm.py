import numpy as np
import tensorflow as tf

from keras import backend
from algorithm.actor_critic import Actor, Critic
from algorithm.util import ReplayBuffer, BrownianMotion, OrnstainUhlenbeck


class DeepDeterministicPolicyGradient:
    def __init__(self, params):
        # tensorflow and keras setup
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.__session = tf.Session(config=config)
        backend.set_session(self.__session)

        # create actor and critic
        self.__actor = Actor(self.__session, params['ACTOR_PARAMS'])
        self.__critic = Critic(self.__session, params['CRITC_PARAMS'])
        self.__session.run(tf.initialize_all_variables())
        self.__actor.init_target_weights()
        self.__critic.init_target_weights()

        # init replay buffer
        self.__buffer = ReplayBuffer(params['BUFFER_SIZE'])

        # init noise for action exploration
        self.__acceleration_noise = OrnstainUhlenbeck(theta=params['ACCELERATION_NOISE']['THETA'],
                                                      mu=params['ACCELERATION_NOISE']['MU'],
                                                      sigma=params['ACCELERATION_NOISE']['SIGMA'],
                                                      brownian_motion=BrownianMotion(delta=params['BROWNIAN']['DELTA'],
                                                                                     dt=params['BROWNIAN']['DT']))
        self.__steering_noise = OrnstainUhlenbeck(theta=params['STEERING_NOISE']['THETA'],
                                                  mu=params['STEERING_NOISE']['MU'],
                                                  sigma=params['STEERING_NOISE']['SIGMA'],
                                                  brownian_motion=BrownianMotion(delta=params['BROWNIAN']['DELTA'],
                                                                                 dt=params['BROWNIAN']['DT']))
        self.__brake_noise = OrnstainUhlenbeck(theta=params['BRAKE_NOISE']['THETA'],
                                               mu=params['BRAKE_NOISE']['MU'],
                                               sigma=params['BRAKE_NOISE']['SIGMA'],
                                               brownian_motion=BrownianMotion(delta=params['BROWNIAN']['DELTA'],
                                                                              dt=params['BROWNIAN']['DT']))

        # init last sensors and actions values
        self.__last_sensors = None
        self.__last_actions = None

    def load_models(self, f_actor, f_critic):
        self.__actor.load(f_actor)
        self.__critic.load(f_critic)

    def save_models(self, f_actor, f_critic):
        self.__actor.save(f_actor)
        self.__critic.save(f_critic)

    def export_dl4j(self, filename):
        self.__actor.export_dl4j(filename)

    def reset(self):
        self.__last_sensors = None

    def eval_step(self, actions, sensors):
        return None  # TODO

    @staticmethod
    def __sensors2array(sensors):
        state = np.zeros(29)
        state[0] = sensors['angle'] / 3.1416  # TODO
        state[1:20] = np.array(sensors['track']) / 200.0
        state[20] = sensors['trackPos']
        state[21] = sensors['speedX'] / 300.0
        state[22] = sensors['speedY'] / 300.0
        state[23] = sensors['speedZ'] / 300.0
        state[24:28] = np.array(sensors['wheelSpinVel']) / 100.0
        state[28] = sensors['rpm'] / 10000.0
        return np.reshape(state, (1, 29))

    def __noise(self, actions_vec):
        return np.array([self.__steering_noise.sample(actions_vec[0]),
                         self.__acceleration_noise.sample(actions_vec[1]),
                         self.__brake_noise.sample(actions_vec[2])])

    @staticmethod
    def __array2actions(actions_vec, actions_dic):
        actions_dic['steer'] = actions_vec[0]
        actions_dic['accel'] = actions_vec[1]
        actions_dic['brake'] = actions_vec[2]

    def __reward(self, sensors):
        return 0

    def train_step(self, actions, sensors):
        if not self.__last_sensors:
            self.__last_sensors = sensors

        if not self.__last_actions:
            self.__last_actions = actions

        last_state = self.__sensors2array(self.__last_sensors)

        actions_vec = self.__actor.predict(last_state)
        actions_vec += self.__noise(actions_vec)

        reward = self.__reward(sensors)
        self.__buffer.add(state=self.__last_sensors, action=self.__last_actions, reward=reward, new_state=sensors)






        self.__last_sensors = sensors
        self.__last_actions = actions
        self.__array2actions(actions_vec, actions)
