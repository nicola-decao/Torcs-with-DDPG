from algorithm import algorithm
from environment import Environment
from parameters import DDPGParams
from utilities.data_handler import encode_state_data, decode_action_data


def train():
    env = Environment(gui=True)
    parameters = DDPGParams()
    model = algorithm.DeepDeterministicPolicyGradient(parameters)

    action = None
    for i in range(10000000):
        action, sensors = env.step(action)

        # Encoding of the sensor into a vector
        state_vec = encode_state_data(sensors)

        # Evaluating the corresponding action
        action_vec = model.train_step(state_vec)

        # Decoding the data from prediction vector
        decode_action_data(action, action_vec)


if __name__ == "__main__":
    train()
