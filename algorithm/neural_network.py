import numpy as np


class ModelTargetNeuralNetwork:
    def __init__(self, params):
        self._model = params['NET']()
        self._target = params['NET']()
        self.__tau = params['TAU']

    def init_target_weights(self):
        self._target.set_weights(self._model.get_weights())

    def load(self, filename):
        self._model.load_weights(filename)
        self.init_target_weights()

    def save(self, filename):
        self._model.save_weights(filename, overwrite=True)

    def export_dl4j(self, filename):
        r = []
        for w in self._model.get_weights():
            r += np.transpose(w).flatten().tolist()

        np.savetxt(filename, np.array(r))

    def update_target(self):
        weights = self._model.get_weights()
        target_weights = self._target.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.__tau * weights[i] + (1 - self.__tau) * target_weights[i]
        self._target.set_weights(target_weights)
