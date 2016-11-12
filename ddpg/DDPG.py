import numpy as np
import tensorflow as tf

from keras import backend


class DDPG:

    def load_models(self, f_actor, f_critic):
        self.__actor.load(f_actor)
        self.__critic.load(f_critic)

    def save_models(self, f_actor, f_critic):
        self.__actor.save(f_actor)
        self.__critic.save(f_critic)

    def export_dl4j(self, filename):
        r = []
        for w in self.__actor.get_weights():
            r += np.transpose(w).flatten().tolist()

        np.savetxt(filename, np.array(r))

    def __init__(self):
        # tensorflow and keras setup
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.__sess = tf.Session(config=config)
        backend.set_session(self.__sess)

        # create actor and critic
        self.__actor = None #TODO
        self.__critic = None #TODO

    def eval_step(self, sensors):
        return None  # TODO

    def train_step(self, sensors):
        return None #TODO