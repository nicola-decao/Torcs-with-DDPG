import subprocess

import sys


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
