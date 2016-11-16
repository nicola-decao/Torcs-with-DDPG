from environment import Environment
from utilities.tracks_utils import track_list
import json_tricks
import numpy as np
from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.initializations import normal
from keras.layers import Dense

from utilities.tracks_utils import get_empty_actions, encode_np_dict


class DatasetGenerator:
    def __init__(self, track):
        self.hidden_neurons_1 = 300
        self.hidden_neurons_2 = 600
        self.state_size = 29
        self.network = self.create_actor_network()
        self.network.load_weights('trained_networks/base_network.h5')
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
        new_actions = get_empty_actions()
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
        encoded_actions = encode_np_dict(actions)
        encoded_sensors = encode_np_dict(sensors)
        self.observations.append((encoded_actions, encoded_sensors))

    def write_dataset(self):
        with open('training_datasets/' + self.track + '.json', 'w') as f:
            json_tricks.dump(self.observations, f)
            f.flush()


def generate_training_dataset(gui=True):
    non_valid_runs = 0
    non_valid_tracks = []
    print('Starting datasets generation')

    for key in track_list.keys():
        track = key
        track_type = track_list[track]

        print('Track: ' + track)
        print('Track type: ' + track_type)

        env = Environment(track=track, track_type=track_type, gui=gui)
        model = DatasetGenerator(track)

        actions = None

        last_distance = -1
        lap = -1
        laps = 3
        out_of_track = False
        while True:
            actions, sensors = env.step(actions)
            lap, last_distance = check_if_lap(lap, last_distance, sensors['distFromStart'])
            if lap == laps:
                break
            if env.check_sensors(sensors) == 1:
                out_of_track = True
                non_valid_runs += 1
                non_valid_tracks.append(track)
                print('Out of track! Not saved')
                break
            actions = model.get_output(actions=actions, sensors=sensors)
            model.add_observation(actions, sensors)

        if not out_of_track:
            model.write_dataset()
        env.shutdown()
        print('end')
        print()

    print(
        'The network failed to complete ' + str(non_valid_runs) + ' tracks out of ' + str(len(track_list.keys())))
    print('The failed tracks are:')
    for track in non_valid_tracks:
        print(track)


def check_if_lap(lap, last_distance, current_distance):
    if last_distance == -1:
        return lap, current_distance
    if last_distance > current_distance:
        return lap + 1, current_distance
    return lap, current_distance

if __name__ == "__main__":
    generate_training_dataset(gui=False)