import os
import time


def speed_up_time():
    speed_up = True
    if speed_up:
        time.sleep(1)
        os.system('xdotool key --window "$(xdotool search --name "/usr/local/lib/torcs/torcs-bin" | head -n 1)" 35')
