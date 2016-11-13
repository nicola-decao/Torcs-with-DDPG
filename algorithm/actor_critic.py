import tensorflow as tf

from algorithm.neural_network import ModelTargetNeuralNetwork


class Actor(ModelTargetNeuralNetwork):
    def __init__(self, session, params):
        super().__init__(params)

        self.__action_gradient = tf.placeholder(tf.float32, [None, 3])
        self.__gradient = tf.gradients(self._model.output, self._model.trainable_weights, -self.__action_gradient)
        self.__optimize = tf.train.AdamOptimizer(params['LEARNING_RATE']).apply_gradients(
            zip(self.__gradient, self._model.trainable_weights))
        self.__session = session

    def train(self, states, action_gradient):
        self.__session.run(self.__optimize, feed_dict={
            self._model.input: states,
            self.__action_gradient: action_gradient
        })

    def predict(self, state):
        return self._model.predict(state)[0]


class Critic(ModelTargetNeuralNetwork):
    def __init__(self, session, params):
        super().__init__(params)
