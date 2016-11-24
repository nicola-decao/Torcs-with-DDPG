import glob
import json
import os

import numpy as np
from keras import optimizers
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, Flatten

observation_input = Input(shape=(1, 29))
h0 = Dense(300, activation='relu')(Flatten()(observation_input))
h1 = Dense(600, activation='relu')(h0)
output = Dense(2, activation='tanh')(h1)
actor = Model(input=observation_input, output=output)

actor.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adamax(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00))

states = []
actions = []

for fn in glob.glob('training_datasets/*.json'):
    if os.path.isfile(fn):
        print(fn)
        with open('training_datasets/g-track-1.json') as data_file:
            data = json.load(data_file)

        for pair in data:
            temp_actions = pair[0]
            sensors = pair[1]

            state = np.empty(29)
            state[0] = sensors['angle'] / np.pi
            state[1:20] = np.array(sensors['track']) / 200.0
            state[20] = sensors['trackPos'] / 1.0
            state[21] = sensors['speedX'] / 300.0
            state[22] = sensors['speedY'] / 300.0
            state[23] = sensors['speedZ'] / 300.0
            state[24:28] = np.array(sensors['wheelSpinVel']) / 100.0
            state[28] = sensors['rpm'] / 10000.0

            action = np.empty(2)
            action[0] = temp_actions['steer']
            action[1] = temp_actions['accel'] - temp_actions['brake']

            states.append(np.array(state).reshape(1, 29))
            actions.append(action)
states = np.array(states)
actor.fit(states, np.array(actions), nb_epoch=200)
actor.save_weights('pre_trained.h5f')
