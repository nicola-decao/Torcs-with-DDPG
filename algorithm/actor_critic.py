import tensorflow as tf
from keras.optimizers import Adam
import threading

from algorithm.model_target_neural_network import ModelTargetNeuralNetwork


class Actor(ModelTargetNeuralNetwork):
    def __init__(self, session, params):
        super().__init__(session, params)

        self.__action_gradient = tf.placeholder(tf.float32, [None, self._model.output[0].get_shape()[0]])
        self.__gradient = tf.gradients(self._model.output, self._model.trainable_weights, self.__action_gradient)
        self.__optimize = tf.train.AdamOptimizer(params.LEARNING_RATE).apply_gradients(
            zip(self.__gradient, self._model.trainable_weights))
        self.__mutex = threading.Semaphore()

    def train(self, states, action_gradient):
        self.__mutex.acquire()
        self._session.run(self.__optimize, feed_dict={
            self._model.input: states,
            self.__action_gradient: action_gradient
        })
        self.__mutex.release()

    def predict(self, state):
        self.__mutex.acquire()
        result = self._model.predict(state)
        self.__mutex.release()
        return result

    def target_predict(self, state):
        return self._target.predict(state)


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
        return self._model.predict([state, action])

    def target_predict(self, state, action):
        return self._target.predict([state, action])

    def train_on_batch(self, states, actions, y):
        return self._model.train_on_batch([states, actions], y)
