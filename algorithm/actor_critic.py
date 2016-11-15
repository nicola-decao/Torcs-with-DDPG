import tensorflow as tf
import numpy as np
from keras.optimizers import Adam

from algorithm.model_target_neural_network import ModelTargetNeuralNetwork


class Actor(ModelTargetNeuralNetwork):
    def __init__(self, session, params):
        super().__init__(session, params)

        self.__action_gradient = tf.placeholder(tf.float32, [None, self._model.output[0].get_shape()[0]])
        self.__gradient = tf.gradients(self._model.output, self._model.trainable_weights, -self.__action_gradient)
        self.__optimize = tf.train.AdamOptimizer(params.LEARNING_RATE).apply_gradients(
            zip(self.__gradient, self._model.trainable_weights))

    def train(self, states, action_gradient):
        action_gradient = action_gradient
        self._session.run(self.__optimize, feed_dict={
            self._model.input: states,
            self.__action_gradient: action_gradient
        })

    def predict(self, state):
        result = self._model.predict(state)
        return np.reshape(result, (state.shape[0], self._model.output[0].get_shape()[0]))

    def target_predict(self, state):
        return np.reshape(self._target.predict(state), (state.shape[0], self._model.output[0].get_shape()[0]))


class Critic(ModelTargetNeuralNetwork):
    def __init__(self, session, params):
        super().__init__(session, params)

        self._model.compile(loss='mse', optimizer=Adam(lr=params.LEARNING_RATE))
        self.__gradient = tf.gradients(self._model.output, self._model.input[1])

    def gradients(self, state, action):
        return self._session.run(self.__gradient, feed_dict={
            self._model.input[0]: state,
            self._model.input[1]: action
        })[0]

    def predict(self, state, action):
        return self._model.predict([state, action])[0, 0]

    def target_predict(self, state, action):
        return self._target.predict([state, action])[0, 0]

    def train_on_batch(self, states, actions, y):
        self._model.train_on_batch([states, actions], y)
