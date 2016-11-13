import subprocess
import sys

import numpy as np


def cmd_exists(cmd):
    return subprocess.call("type " + cmd, shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0


def print_progress(i, total):
    if i == 1:
        print('[', end='')
    if i % (total / 10) == 0:
        print('#', end='')
    if i == total:
        print(']')
    sys.stdout.flush()


def get_empty_actions():
    return {'steer': 0, 'accel': 0, 'gear': 0, 'brake': 0, 'clutch': 0, 'meta': 0,
            'focus': [-90, -45, 0, 45, 90]}


track_list = {'aalborg': 'road',
              'alpine-1': 'road',
              'alpine-2': 'road',
              'brondehach': 'road',
              'corkscrew': 'road',
              'eroad': 'road',
              'e-track-1': 'road',
              'e-track-2': 'road',
              'e-track-3': 'road',
              'e-track-4': 'road',
              'e-track-6': 'road',
              'forza': 'road',
              'g-track-1': 'road',
              'g-track-2': 'road',
              'g-track-3': 'road',
              'ole-road-1': 'road',
              'ruudskogen': 'road',
              'spring': 'road',
              'street-1': 'road',
              'wheel-1': 'road',
              'wheel-2': 'road',
              'a-speedway': 'oval',
              'b-speedway': 'oval',
              'c-speedway': 'oval',
              'd-speedway': 'oval',
              'e-speedway': 'oval',
              'e-track-5': 'oval',
              'f-speedway': 'oval',
              'g-speedway': 'oval',
              'michigan': 'oval',
              'dirt-1': 'dirt',
              'dirt-2': 'dirt',
              'dirt-3': 'dirt',
              'dirt-4': 'dirt',
              'dirt-5': 'dirt',
              'dirt-6': 'dirt',
              'mixed-1': 'dirt',
              'mixed-2': 'dirt'}


def greetings():
    print('This race has been brought to you by:')
    print('HitLuca')
    print('M4RC0SX')
    print('nicola-decao')
    print()
    print('And remember, drive safe!')


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
