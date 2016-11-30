import os
import time


def speed_up_time():
    speed_up = False
    if speed_up:
        time.sleep(0.1)
        os.system('xdotool key --window "$(xdotool search --name "/usr/local/lib/torcs/torcs-bin" | head -n 1)" 35')
