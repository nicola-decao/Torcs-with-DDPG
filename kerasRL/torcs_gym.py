import subprocess

import numpy as np
from gym import spaces
from gym.core import Env
import os
import socket
import sys
import time
from xml.etree import ElementTree as etree

data_size = 2 ** 17


def cmd_exists(cmd):
    return subprocess.call("type " + cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0


class TorcsEnv(Env):
    def __init__(self, host='localhost', port=3001, sid='SCR', track='g-track-1', track_type='road', gui=True):
        self.gui = gui
        self.server = self.Server(track, track_type, gui)
        self.client = self.Client(self.server, host, port, sid)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
        low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
        self.observation_space = spaces.Box(low=0, high=200, shape=(29,))
        self.__terminal_judge_start = 200  # If after 100 timestep still no progress, terminated
        # self.__termination_limit_progress = 20  # [km/h], episode terminates if car is running slower than this limit
        self.__time_stop = 0
        self.__last_speedX = 0

    def restart_environment(self):
        self.server.restart()
        self.client.restart()

    def restart_race(self):
        if self.gui:
            self.client.send_restart_request()
            self.client.restart()
        else:
            self.restart_environment()

    def _reset(self):
        self.restart_race()
        self.__time_stop = 0
        return self.encode_state_data(self.client.step(None))

    def _step(self, action):
        a = self.decode_action_data(action)

        if self.__last_speedX < 50:
            a['gear'] = 1
        elif self.__last_speedX < 80:
            a['gear'] = 2
        elif self.__last_speedX < 110:
            a['gear'] = 3
        elif self.__last_speedX < 140:
            a['gear'] = 4
        elif self.__last_speedX < 170:
            a['gear'] = 5
        else:
            a['gear'] = 6

        sensors = self.client.step(a)
        observation = self.encode_state_data(sensors)
        self.__last_speedX = sensors['speedX']

        if np.abs(observation[20]) > 0.99:
            reward, done = -200, True
        else:
            reward = 300 * observation[21] * (
                            np.cos(observation[0] * (np.pi ** 2))
                            - np.abs(np.sin(observation[0] * (np.pi ** 2))))

            # reward = 300 * observation[21] * (
            #                 np.cos(observation[0] * (np.pi ** 2))
            #                 - np.abs(np.sin(observation[0] * (np.pi ** 2)))
            #                 - np.abs(observation[20]))

            done = False  # self.__terminal_judge_start < self.__time_step and reward < self.__termination_limit_progress
            if 300 * observation[21] < 20:
                self.__time_stop += 1
            else:
                self.__time_stop = 0

            if self.__time_stop > self.__terminal_judge_start:
                done = True

        return observation, reward, done, {}

    def decode_action_data(self, actions_vec):
        actions_dic = self.client.get_empty_actions()
        actions_dic['steer'] = actions_vec[0]
        actions_dic['accel'] = actions_vec[1]
        actions_dic['brake'] = actions_vec[2]
        return actions_dic

    @staticmethod
    def encode_state_data(sensors):
        state = np.empty(29)
        state[0] = sensors['angle'] / np.pi
        state[1:20] = np.array(sensors['track']) / 200.0
        state[20] = sensors['trackPos'] / 1.0
        state[21] = sensors['speedX'] / 300.0
        state[22] = sensors['speedY'] / 300.0
        state[23] = sensors['speedZ'] / 300.0
        state[24:28] = np.array(sensors['wheelSpinVel']) / 100.0
        state[28] = sensors['rpm'] / 10000.0
        return state

    @staticmethod
    def encode_np_dict(dictionary):
        encoded = {}
        for key in dictionary.keys():
            if dictionary[key] is list:
                encoded[key] = dictionary[key].tolist()
            elif type(dictionary[key]) == np.float64 or type(dictionary[key]) == np.float32:
                encoded[key] = float(dictionary[key])
            else:
                encoded[key] = dictionary[key]
        return encoded

    @staticmethod
    def print_progress(i, total):
        if i == 1:
            print('[', end='')
        if i % (total / 10) == 0:
            print('#', end='')
        if i == total:
            print(']')
        sys.stdout.flush()

    class Server:
        quickrace_xml_path = os.path.expanduser('~') + '/.torcs/config/raceman/quickrace.xml'

        def __init__(self, track, track_type, gui):
            self.track = track
            self.track_type = track_type
            self.gui = gui

            self.create_race_xml()
            self.init_server()

        def init_server(self):
            self.shutdown()
            time.sleep(0.1)
            if self.gui is True:
                if cmd_exists('optirun'):
                    os.system('optirun torcs -nofuel -nodamage -nolaptime &')
                else:
                    os.system('torcs -nofuel -nodamage -nolaptime &')
                time.sleep(2)
                os.system('sh autostart.sh')
            else:
                os.system('torcs -nofuel nodamage -nolaptime -r ' + self.quickrace_xml_path + ' &')
            print('Server created!')
            time.sleep(0.1)

        def restart(self):
            print('Restarting server...')
            os.system('pkill torcs')
            time.sleep(0.2)
            self.init_server()

        @staticmethod
        def shutdown():
            os.system('pkill torcs')

        def create_race_xml(self):
            root = etree.parse(self.quickrace_xml_path)
            track_name = root.find('section[@name="Tracks"]/section[@name="1"]/attstr[@name="name"]')
            track_name.set('val', self.track)
            track_type = root.find('section[@name="Tracks"]/section[@name="1"]/attstr[@name="category"]')
            track_type.set('val', self.track_type)
            laps = root.find('section[@name="Quick Race"]/attnum[@name="laps"]')
            laps.set('val', '1000')
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
        def get_empty_actions():
            return {'steer': 0, 'accel': 0, 'gear': 1, 'brake': 0, 'clutch': 0, 'meta': 0,
                    'focus': [-90, -45, 0, 45, 90]}

        def restart(self):
            self.socket = self.create_socket()
            self.connect_to_server()

        def send_restart_request(self):
            actions = self.get_empty_actions()
            actions['meta'] = True
            message = self.encode_actions(actions)
            self.send_message(message)

        def shutdown(self):
            self.socket.close()

        def send_message(self, message):
            try:
                self.socket.sendto(message.encode(), (self.host, self.port))
            except socket.error as emsg:
                print(u"Error sending to server: %s Message %s" % (emsg[1], str(emsg[0])))
                sys.exit(-1)

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
            tries = 3
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
                actions = self.get_empty_actions()

            if not self.socket:
                print('Client socket problem!')
                return
            self.limit_actions(actions)
            message = self.encode_actions(actions)
            self.send_message(message)

            return self.get_server_input()

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