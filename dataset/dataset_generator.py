import json_tricks
import numpy as np
from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.initializations import normal
from keras.layers import Dense

import utils


class DatasetGenerator:
    def __init__(self, track):
        self.hidden_neurons_1 = 300
        self.hidden_neurons_2 = 600
        self.state_size = 29
        self.network = self.create_actor_network()
        self.network.load_weights('dataset/trained_network.h5')
        self.observations = []
        self.track = track

    def create_actor_network(self):
        s = Input(shape=[self.state_size])
        h0 = Dense(self.hidden_neurons_1, activation='relu')(s)
        h1 = Dense(self.hidden_neurons_2, activation='relu')(h0)
        steering = Dense(1, activation='tanh', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        acceleration = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        brake = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        v = merge([steering, acceleration, brake], mode='concat')
        return Model(input=s, output=v)

    def get_output(self, actions, sensors):
        state = np.hstack((
            np.array(sensors['angle']) / 3.1416,
            np.array(sensors['track']) / 200.,
            np.array(sensors['trackPos']) / 1.,
            np.array(sensors['speedX']) / 300.0,
            np.array(sensors['speedY']) / 300.0,
            np.array(sensors['speedZ']) / 300.0,
            np.array(sensors['wheelSpinVel']) / 100.0,
            np.array(sensors['rpm']) / 10000))

        prediction = self.network.predict(state.reshape(1, state.shape[0]))[0]
        new_actions = utils.get_empty_actions()
        new_actions['steer'] = prediction[0]
        new_actions['accel'] = prediction[1]
        new_actions['brake'] = prediction[2]
        new_actions['gear'] = actions['gear']

        new_actions['gear'] = 1
        if sensors['speedX'] > 50:
            new_actions['gear'] = 2
        if sensors['speedX'] > 80:
            new_actions['gear'] = 3
        if sensors['speedX'] > 110:
            new_actions['gear'] = 4
        if sensors['speedX'] > 140:
            new_actions['gear'] = 5
        if sensors['speedX'] > 170:
            new_actions['gear'] = 6

        return new_actions

    def add_observation(self, actions, sensors):
        encoded_actions = utils.encode_np_dict(actions)
        encoded_sensors = utils.encode_np_dict(sensors)
        self.observations.append((encoded_actions, encoded_sensors))

    def write_dataset(self):
        with open('training_datasets/' + self.track + '.json', 'w') as f:
            json_tricks.dump(self.observations, f)
            f.flush()
