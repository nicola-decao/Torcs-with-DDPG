import time
import numpy as np
from algorithm import algorithm
from environment import Environment
from parameters import DDPGParams, DataHandler
from utilities.tracks_utils import track_list, greetings


def train(episodes, steps_per_episode, gui=True, load=False, save=True):

    track = 'dirt-1'
    track_type = track_list[track]

    print('Starting simulation...')
    print('Track: ' + track)
    print('Track type: ' + track_type)
    print('GUI: ' + str(gui))
    print()

    model = algorithm.DeepDeterministicPolicyGradient(DDPGParams())
    env = Environment(track=track, track_type=track_type, gui=gui)

    if load:
        print('loading..')
        model.load_models('actor.h5', 'critic.h5')
        print('loaded!')

    dist_from_start = []
    for e in range(episodes):

        action = None
        start = time.time()
        crashed = False
        t = -1

        print('Episode ' + str(e + 1) + '/' + str(episodes))

        for _ in range(steps_per_episode):
            # utils.print_progress(j + 1, steps_per_episode)
            action, sensors = env.step(action)

            if env.check_sensors(sensors) and not crashed:
                dist_from_start.append(sensors['distFromStart'])
                t = time.time()
                crashed = True

            if crashed and (time.time() - t) > 0.5:
                break

            # Encoding of the sensor into a vector
            state_vec = DataHandler.encode_state_data(sensors)

            # Evaluating the corresponding action
            #action_vec = model.train_step(state_vec)
            action_vec = model.eval_step(state_vec)
            # Decoding the data from prediction vector
            DataHandler.decode_action_data(action, action_vec)

        if e % 10 == 9:
            if save:
                print('saving models..')
                model.save_models('actor.h5', 'critic.h5')
                print('saved!')
            env.restart_environment()
        else:
            env.restart_race()

        print('distFromStart mean: ', np.mean(dist_from_start), 'of', len(dist_from_start), ' last:', dist_from_start[-1])
        print("Episode last {}".format(time.time() - start))
        print()

    model.stop()
    env.shutdown()

    greetings()

if __name__ == "__main__":
    # first time set load to false then true
    train(3000000, 100000, gui=True, load=True, save=True)
