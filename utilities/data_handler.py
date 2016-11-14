import numpy as np

def encode_state_data(sensors):
    state = np.zeros(29)
    state[0] = sensors['angle'] / 3.1416  # TODO
    state[1:20] = np.array(sensors['track']) / 200.0
    state[20] = sensors['trackPos']
    state[21] = sensors['speedX'] / 300.0
    state[22] = sensors['speedY'] / 300.0
    state[23] = sensors['speedZ'] / 300.0
    state[24:28] = np.array(sensors['wheelSpinVel']) / 100.0
    state[28] = sensors['rpm'] / 10000.0
    return np.reshape(state, (1, 29))

def decode_action_data(actions_dic, actions_vec):
    actions_dic['steer'] = actions_vec[0]
    actions_dic['accel'] = actions_vec[1]
    actions_dic['brake'] = actions_vec[2]
