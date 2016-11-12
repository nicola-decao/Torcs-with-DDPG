import numpy as np
import math

import tensorflow as tf

from keras import backend


class DeepDeterministicPolicyGradient:
    def __init__(self):
        # tensorflow and keras setup
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.__sess = tf.Session(config=config)
        backend.set_session(self.__sess)

        # create actor and critic
        self.__actor = None #TODO
        self.__critic = None #TODO

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



    def eval_step(self, actions, sensors):
        return None  # TODO

    def train_step(self, actions, sensors):
        # TODO this is just a dummy implementation of the algorithm
        target_speed = 100

        # Steer To Corner
        actions['steer'] = sensors['angle'] * 10 / math.pi
        # Steer To Center
        actions['steer'] -= sensors['trackPos'] * .10

        # Throttle Control
        if sensors['speedX'] < target_speed - (actions['steer'] * 50):
            actions['accel'] += .01
        else:
            actions['accel'] -= .01
        if sensors['speedX'] < 10:
            actions['accel'] += 1 / (sensors['speedX'] + .1)

        # Traction Control System
        if ((sensors['wheelSpinVel'][2] + sensors['wheelSpinVel'][3]) -
                (sensors['wheelSpinVel'][0] + sensors['wheelSpinVel'][1]) > 5):
            actions['accel'] -= .2

        # Automatic Transmission
        actions['gear'] = 1
        if sensors['speedX'] > 50:
            actions['gear'] = 2
        if sensors['speedX'] > 80:
            actions['gear'] = 3
        if sensors['speedX'] > 110:
            actions['gear'] = 4
        if sensors['speedX'] > 140:
            actions['gear'] = 5
        if sensors['speedX'] > 170:
            actions['gear'] = 6
        return actions