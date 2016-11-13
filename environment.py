import os
import socket
from xml.etree import ElementTree as etree
import time
import sys
from algorithm import algorithm
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

data_size = 2 ** 17


def create_actor_net():
    state = Input(shape=[29])
    h0 = Dense(300, activation='relu')(state)
    h1 = Dense(600, activation='relu')(h0)
    steering = Dense(1, activation='tanh', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
    acceleration = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
    brake = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
    return Model(input=state, output=merge([steering, acceleration, brake], mode='concat'))


def create_critic_net():
    hidden_units_1 = 300
    hidden_units_2 = 600
    state = Input(shape=[29])
    action = Input(shape=[3])
    s1 = Dense(hidden_units_1, activation='relu')(state)
    a1 = Dense(hidden_units_2, activation='linear')(action)
    h1 = Dense(hidden_units_2, activation='linear')(s1)
    h2 = merge([h1, a1], mode='sum')
    h3 = Dense(hidden_units_2, activation='relu')(h2)
    return Model(input=[state, action], output=Dense(1, activation='linear')(h3))

DDPGParams = {
    'BUFFER_SIZE': 200,
    'ACTOR_PARAMS': {
        'LEARNING_RATE': 0.0001,
        'TAU': 0.001,
        'NET': create_actor_net
    },
    'CRITC_PARAMS': {
        'LEARNING_RATE': 0.001,
        'TAU': 0.001,
        'NET': create_critic_net
    },
    'ACCELERATION_NOISE': {
        'THETA': 1.0,
        'MU': 0.45,
        'SIGMA': 0.1
    },
    'BRAKE_NOISE': {
        'THETA': 1.0,
        'MU': -0.1,
        'SIGMA': 0.05
    },
    'STEERING_NOISE': {
        'THETA': 0.6,
        'MU': 0.0,
        'SIGMA': 0.3
    },
    'BROWNIAN': {
        'DT': 0.1,
        'DELTA': 0.25
    }
}

class Environment:
    def __init__(self, host='localhost', port=3001, sid='SCR', track='forza', track_type='road', gui=True):
        self.server = Server(track, track_type, gui)
        self.client = Client(self.server, host, port, sid)

    def step(self, actions):
        return self.client.step(actions)

    def empty_actions(self):
        return {'steer': 0, 'accel': 0, 'gear': 0, }


class Server:
    quickrace_xml_path = os.path.expanduser('~') + '/.torcs/config/raceman/quickrace.xml'

    def __init__(self, track, track_type, gui):
        self.track = track
        self.track_type = track_type
        self.gui = gui

        self.create_race_xml()
        self.init_server()

    def init_server(self):
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.gui is True:
            os.system('torcs -nofuel -nodamage -nolaptime &')
            time.sleep(2)
            os.system('sh autostart.sh')
        else:
            os.system('torcs -nofuel nodamage -nolaptime -r ' + self.quickrace_xml_path + ' &')
        print('Server created!')
        time.sleep(0.5)

    def restart(self):
        print('Restarting server...')
        os.system('pkill torcs')
        time.sleep(0.5)
        self.init_server()

    def create_race_xml(self):
        root = etree.parse(self.quickrace_xml_path)
        track_name = root.find('section[@name="Tracks"]/section[@name="1"]/attstr[@name="name"]')
        track_name.set('val', self.track)
        track_type = root.find('section[@name="Tracks"]/section[@name="1"]/attstr[@name="category"]')
        track_type.set('val', self.track_type)
        root.write(self.quickrace_xml_path)


class Client:
    def __init__(self, server, host, port, sid):
        self.server = server
        self.host = host
        self.port = port
        self.sid = sid

        self.socket = self.create_socket()
        self.connect_to_server()

    @staticmethod
    def create_socket():
        try:
            so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error:
            print('Error: Could not create socket...')
            sys.exit(-1)
        so.settimeout(1)
        return so

    def connect_to_server(self):
        tries = 5
        while True:
            sensor_angles = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = '%s(init %s)' % (self.sid, sensor_angles)

            try:
                self.socket.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error:
                sys.exit(-1)
            sockdata = str()

            try:
                sockdata, address = self.socket.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error:
                print("Waiting for server on port " + str(self.port))
                tries -= 1
                if tries == 0:
                    print("Server didn't answer, sending restart signal")
                    self.server.restart()
                time.sleep(0.5)

            identify = '***identified***'
            if identify in sockdata:
                print("Client connected on port " + str(self.port))
                break

    @staticmethod
    def encode_actions(actions):
        out = str()
        for k in actions:
            out += '(' + k + ' '
            v = actions[k]
            if not type(v) is list:
                out += '%.3f' % v
            else:
                out += ' '.join([str(x) for x in v])
            out += ')'
        return out

    @staticmethod
    def limit_action(v, lo, hi):
        if v < lo:
            return lo
        elif v > hi:
            return hi
        else:
            return v

    def limit_actions(self, actions):
        actions['steer'] = self.limit_action(actions['steer'], -1, 1)
        actions['brake'] = self.limit_action(actions['brake'], 0, 1)
        actions['accel'] = self.limit_action(actions['accel'], 0, 1)
        actions['clutch'] = self.limit_action(actions['clutch'], 0, 1)
        if actions['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            actions['gear'] = 0
        if actions['meta'] not in [0, 1]:
            actions['meta'] = 0
        if type(actions['focus']) is not list or min(actions['focus']) < -180 or max(
                actions['focus']) > 180:
            actions['focus'] = 0

    def step(self, actions):
        if actions is None:
            actions = {'accel': 0.2,
                       'brake': 0,
                       'clutch': 0,
                       'gear': 1,
                       'steer': 0,
                       'focus': [-90, -45, 0, 45, 90],
                       'meta': 0
                       }

        if not self.socket:
            print('Client socket problem!')
            return
        try:
            self.limit_actions(actions)
            message = self.encode_actions(actions)
            self.socket.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1], str(emsg[0])))
            sys.exit(-1)

        return actions, self.get_server_input()

    def parse_server_string(self, server_string):
        track_data = {}
        server_string = server_string.strip()[:-1]
        server_string_list = server_string.strip().lstrip('(').rstrip(')').split(')(')
        for i in server_string_list:
            w = i.split(' ')
            track_data[w[0]] = self.destringify(w[1:])
        return track_data

    def destringify(self, string):
        if not string:
            return string
        if type(string) is str:
            try:
                return float(string)
            except ValueError:
                print("Could not find a value in %s" % string)
                return string
        elif type(string) is list:
            if len(string) < 2:
                return self.destringify(string[0])
            else:
                return [self.destringify(i) for i in string]

    def get_server_input(self):
        sockdata = str()

        while True:
            try:
                sockdata, address = self.socket.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error:
                print('', end='')
            if sockdata:
                return self.parse_server_string(sockdata)

def train():
    env = Environment()
    model = algorithm.DeepDeterministicPolicyGradient(DDPGParams)

    actions = None
    for i in range(10000000):
        actions, sensors = env.step(actions)
        model.train_step(actions=actions, sensors=sensors)

if __name__ == "__main__":
    train()
