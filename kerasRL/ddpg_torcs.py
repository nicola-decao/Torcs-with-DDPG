import numpy as np
import gym

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam

from kerasRL.torcs_gym import TorcsEnv
from kerasRL.rl.agents import DDPGAgent
from kerasRL.rl.memory import SequentialMemory
from kerasRL.rl.random import OrnsteinUhlenbeckProcess

class OUN:
    def __init__(self, steer, accel, brake):
        self.__steer = steer
        self.__accel = accel
        self.__brake = brake

    def sample(self):
        if np.random.rand(1) < 0.1:
            return np.array([self.__steer.sample()[0], self.__accel.sample()[0], self.__brake.sample()[0]])
        else:
            return np.array([self.__steer.sample()[0], self.__accel.sample()[0], 0])

ENV_NAME = 'Torcs-v0'
gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = TorcsEnv(gui=True)


assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.

x = Input(shape=(1,) + env.observation_space.shape)
S = Flatten()(x)
h0 = Dense(300, activation='relu')(S)
h1 = Dense(600, activation='relu')(h0)
Steering = Dense(1, activation='tanh')(h1)
Acceleration = Dense(1, activation='sigmoid')(h1)
Brake = Dense(1, activation='sigmoid')(h1)
V = merge([Steering, Acceleration, Brake], mode='concat')
actor = Model(input=x, output=V)
print(actor.summary())


action_input = Input(shape=(nb_actions,))
observation_input = Input(shape=(1,) + env.observation_space.shape)
flattened_observation = Flatten()(observation_input)
x = merge([action_input, flattened_observation], mode='concat')
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(input=[action_input, observation_input], output=x)

# x = Input(shape=(1,) + env.observation_space.shape)
# S = Flatten()(x)
# A = Input(shape=(3,))
# w1 = Dense(300, activation='relu')(S)
# a1 = Dense(300, activation='linear')(A)
# h1 = Dense(300, activation='linear')(w1)
# h2 = merge([h1, a1], mode='sum')
# h3 = Dense(300, activation='relu')(h2)
# V = Dense(1, activation='linear')(h3)
# critic = Model(input=[x, A], output=V)
# print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
#random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3)

random_process = OUN(OrnsteinUhlenbeckProcess(theta=0.6, mu=0, sigma=0.3),
                      OrnsteinUhlenbeckProcess(theta=1.0, mu=0.5, sigma=0.1),
                      OrnsteinUhlenbeckProcess(theta=0.2, mu=1.0, sigma=0.1))

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3, epsilon=0.3)

agent.compile((Adam(lr=.0001, clipnorm=1.), Adam(lr=.001, clipnorm=1.)), metrics=['mae'])
agent.load_weights('good_on_g-track-1_ddpg_{}_weights.h5f'.format(ENV_NAME))

agent.fit(env, nb_steps=50000, visualize=False, verbose=0, nb_max_episode_steps=10000)

agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
#agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=20000)
