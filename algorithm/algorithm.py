from time import sleep

import numpy as np
import tensorflow as tf
from keras import backend as K1
from keras import backend as K2
import threading

from algorithm.actor_critic import Actor, Critic
from utilities.buffer import ReplayBuffer


class DeepDeterministicPolicyGradient:
    def __init__(self, params):
        # tensorflow and keras setup
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # create critic
        self.__critic_session = tf.Session(config=config)
        K1.set_session(self.__critic_session)
        self.__critic = Critic(self.__critic_session, params.CRITIC_PARAMS)
        self.__critic_session.run(tf.initialize_all_variables())
        self.__critic.init_target_weights()

        # create actor
        self.__actor_session = tf.Session(config=config)
        K2.set_session(self.__actor_session)
        self.__actor = Actor(self.__actor_session, params.ACTOR_PARAMS)
        self.__actor_session.run(tf.initialize_all_variables())
        self.__actor.init_target_weights()

        # set the reward function
        self.__reward = params.REWARD_FUNCTION

        # init replay buffer
        self.__batch_size = params.BATCH_SIZE
        self.__buffer = ReplayBuffer(params.BUFFER_SIZE, params.STATE_SIZE, params.ACTION_SIZE)

        # init noise for action exploration
        self.__add_noise = params.NOISE_FUNCTION

        # init last sensors and actions values
        self.__last_state = np.array([])
        self.__last_action = np.array([])

        # init gamma
        self.__gamma = params.GAMMA

        self.__train_networks = True
        self.__training_thread = threading.Thread(target=self.__train)
        self.__training_thread.daemon = True
        self.__training_thread.start()

    def load_models(self, f_actor, f_critic):
        self.__actor.load(f_actor)
        self.__critic.load(f_critic)

    def save_models(self, f_actor, f_critic):
        self.__actor.save(f_actor)
        self.__critic.save(f_critic)

    def export_dl4j(self, filename):
        self.__actor.export_dl4j(filename)

    def reset(self):
        self.__last_state = np.array([])
        self.__last_action = np.array([])

    def eval_step(self, state):
        return self.__actor.predict(state)

    def stop(self):
        self.__train_networks = False
        sleep(1)

    def __train(self):
        while self.__train_networks:
            if not self.__buffer.is_empty():

                # Sample batch from buffer
                states, actions, y, new_states, terminals = self.__buffer.get_batch(self.__batch_size)
                terminals = np.ma.make_mask(terminals)

                with self.__actor_session.graph.as_default():
                    a = self.__actor.target_predict(new_states)

                with self.__critic_session.graph.as_default():
                    y[terminals] += self.__gamma * self.__critic.target_predict(new_states, a)[terminals]

                    self.__critic.train_on_batch(states, actions, y)
                    self.__critic.update_target()

                with self.__actor_session.graph.as_default():
                    self.__actor.train(states, self.__critic.gradients(states, self.__actor.predict(states)))
                    self.__actor.update_target()

            else:
                sleep(0.1)

    def train_step(self, state):
        # Predict the action using the actor network
        with self.__actor_session.graph.as_default():
            action = self.__actor.predict(state)

        # Adding explorative noise to the prediction
        action = self.__add_noise(action)

        if self.__last_action.any() and self.__last_state.any():
            # Compute reward
            reward, terminal = self.__reward(state)

            # Update replay buffer
            self.__buffer.add(state=self.__last_state, action=self.__last_action, reward=reward, new_state=state, terminal=terminal)

        self.__last_state = state
        self.__last_action = action
        return action
