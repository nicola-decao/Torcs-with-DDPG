import numpy as np
import tensorflow as tf
from keras import backend

from algorithm.actor_critic import Actor, Critic
from utilities.buffer import ReplayBuffer


class DeepDeterministicPolicyGradient:
    def __init__(self, params):
        # tensorflow and keras setup
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.__session = tf.Session(config=config)
        backend.set_session(self.__session)

        # create actor and critic
        self.__actor = Actor(self.__session, params.ACTOR_PARAMS)
        self.__critic = Critic(self.__session, params.CRITIC_PARAMS)
        self.__session.run(tf.initialize_all_variables())
        self.__actor.init_target_weights()
        self.__critic.init_target_weights()

        # set the reward function
        self.__reward = params.REWARD_FUNCTION

        # init replay buffer
        self.__batch_size = params.BATCH_SIZE
        self.__buffer = ReplayBuffer(params.BUFFER_SIZE)

        # init noise for action exploration
        self.__add_noise = params.NOISE_FUNCTION

        # init last sensors and actions values
        self.__last_state = np.array([])
        self.__last_action = np.array([])

        # init gamma
        self.__gamma = params.GAMMA

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

    def train_step(self, state):
        # Predict the action using the actor network
        action = self.__actor.predict(state)

        # Adding explorative noise to the prediction
        action = self.__add_noise(action)

        print('lol')
        print(self.__buffer)

        if self.__last_action.any() and self.__last_state.any():
            # Compute reward
            reward = self.__reward(state)

            # Update replay buffer
            self.__buffer.add(state=self.__last_state, action=self.__last_action, reward=reward, new_state=state)

            # Sample batch from buffer
            minibacth = self.__buffer.get_batch(self.__batch_size)


            states = []
            actions = []
            y = []

            for s, a, r, ns in minibacth:
                # TODO check final state
                states.append(s[0])
                actions.append(a[0])
                y.append(r + self.__gamma*self.__critic.target_predict(ns, self.__actor.target_predict(ns)))

            states = np.array(states)
            actions = np.array(actions)
            y = np.array(y)

            self.__critic.train_on_batch(states, actions, y)
            cc = self.__actor.predict(states)
            print(cc)
            grads = self.__critic.gradients(states,  cc)
            self.__actor.train(states, grads)
            self.__actor.update_target()
            self.__critic.update_target()

        self.__last_state = state
        self.__last_action = action
        return action
