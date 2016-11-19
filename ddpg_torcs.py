import gym
import numpy as np
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.models import Model
from keras.optimizers import Adam

from kerasRL.rl.agents import DDPGAgent
from kerasRL.rl.memory import SequentialMemory
from kerasRL.rl.random import OrnsteinUhlenbeckProcess
from torcs_gym import TorcsEnv


GAMMA = 0.99
EPSILON = 0.3
TAU = 1e-3

def get_actor(env):
    observation_input = Input(shape=(1,) + env.observation_space.shape)
    h0 = Dense(300, activation='relu')(Flatten()(observation_input))
    h1 = Dense(600, activation='relu')(h0)
    Steering = Dense(1, activation='tanh')(h1)
    Acceleration = Dense(1, activation='sigmoid')(h1)
    Brake = Dense(1, activation='sigmoid')(h1)
    output = merge([Steering, Acceleration, Brake], mode='concat')
    return Model(input=observation_input, output=output)


def get_critic(env):
    action_input = Input(shape=(env.action_space.shape[0],))
    observation_input = Input(shape=(1,) + env.observation_space.shape)
    h0 = Dense(300, activation='relu')(merge([action_input, Flatten()(observation_input)], mode='concat'))
    h1 = Dense(300, activation='relu')(h0)
    output = Dense(1, activation='linear')(h1)
    return Model(input=[action_input, observation_input], output=output), action_input


class OrnsteinUhlenbeckTriple:
    def __init__(self, steer, accel, brake):
        self.__steer = steer
        self.__accel = accel
        self.__brake = brake

    def sample(self):
        return np.array([self.__steer.sample()[0], self.__accel.sample()[0], self.__brake.sample()[0]])


def export_dl4j(self, filename):
    r = []
    for w in self._model.get_weights():
        r += np.transpose(w).flatten().tolist()

    np.savetxt(filename, np.array(r))

def train(load=False, save=False, gui=True, file_path='', timeout=10000, track='g-track-1', verbose=0,
          nb_steps=50000, nb_max_episode_steps=10000):
    env = TorcsEnv(gui=gui, timeout=timeout, track=track)

    actor = get_actor(env)
    critic, action_input = get_critic(env)

    memory = SequentialMemory(limit=100000, window_length=1)

    random_process = OrnsteinUhlenbeckTriple(OrnsteinUhlenbeckProcess(theta=0.6, mu=0, sigma=0.1),
                                             OrnsteinUhlenbeckProcess(theta=1.0, mu=0.8, sigma=0.1),
                                             OrnsteinUhlenbeckProcess(theta=0.2, mu=1.0, sigma=0.1))

    agent = DDPGAgent(nb_actions=env.action_space.shape[0], actor=actor, critic=critic,
                      critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=GAMMA, target_model_update=TAU, epsilon=EPSILON)

    agent.compile((Adam(lr=.0001, clipnorm=1.), Adam(lr=.001, clipnorm=1.)), metrics=['mae'])

    if load:
        agent.load_weights(file_path)

    agent.fit(env, nb_steps=nb_steps, visualize=False, verbose=verbose, nb_max_episode_steps=nb_max_episode_steps)

    if save:
        agent.save_weights(file_path, overwrite=True)

def test(file_path, track='g-track-1', timeout=40000):
    env = TorcsEnv(gui=True, track=track, timeout=timeout)

    actor = get_actor(env)
    critic, action_input = get_critic(env)

    memory = SequentialMemory(limit=100000, window_length=1)

    random_process = OrnsteinUhlenbeckTriple(OrnsteinUhlenbeckProcess(theta=0.6, mu=0, sigma=0.1),
                                             OrnsteinUhlenbeckProcess(theta=1.0, mu=0.8, sigma=0.1),
                                             OrnsteinUhlenbeckProcess(theta=0.2, mu=1.0, sigma=0.1))

    agent = DDPGAgent(nb_actions=env.action_space.shape[0], actor=actor, critic=critic,
                      critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=GAMMA, target_model_update=TAU, epsilon=EPSILON)

    agent.compile((Adam(lr=.0001, clipnorm=1.), Adam(lr=.001, clipnorm=1.)), metrics=['mae'])
    agent.load_weights(file_path)
    agent.test(env, visualize=False)

if __name__ == "__main__":
    train(load=True, gui=True, save=True, file_path='trained_networks/weights.h5f', timeout=40000, verbose=1, track='f-speedway')
    #test('trained_networks/weights.h5f', track="f-speedway", timeout=40000)

