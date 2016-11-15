import time

from algorithm import algorithm
from environment import Environment
from parameters import DDPGParams, DataHandler
from utilities.tracks_utils import track_list, greetings


def train(episodes, steps_per_episode, gui=True):
    track = 'ole-road-1'
    track_type = track_list[track]

    print('Starting simulation...')
    print('Track: ' + track)
    print('Track type: ' + track_type)
    print('GUI: ' + str(gui))
    print()

    model = algorithm.DeepDeterministicPolicyGradient(DDPGParams())
    time.sleep(3)
    env = Environment(track=track, track_type=track_type, gui=gui)


    for i in range(episodes):
        action = None
        start = time.time()
        print('Episode ' + str(i + 1) + '/' + str(episodes))
        for j in range(steps_per_episode):
            # utils.print_progress(j + 1, steps_per_episode)
            action, sensors = env.step(action)
            if env.check_sensors(sensors):
                print("out of track!")
                env.restart_race()

            # Encoding of the sensor into a vector
            state_vec = DataHandler.encode_state_data(sensors)

            # Evaluating the corresponding action
            action_vec = model.train_step(state_vec)

            # Decoding the data from prediction vector
            DataHandler.decode_action_data(action, action_vec)

        env.restart_environment()
        print()
    print("Episode last {}".format(time.time()-start))
    print('Simulation finished!')
    print()
    model.stop()
    env.shutdown()
    greetings()

if __name__ == "__main__":
    train(3, 10000, gui=False)
