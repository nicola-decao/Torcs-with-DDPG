import time
import numpy as np
from algorithm import algorithm
from environment import Environment
from parameters import DDPGParams, DataHandler
from utilities.tracks_utils import track_list, greetings


def train(episodes, steps_per_episode, batch=50, gui=True, load=False, save=True):

    track = 'ole-road-1'
    track_type = track_list[track]

    print('Starting simulation...')
    print('Track: ' + track)
    print('Track type: ' + track_type)
    print('GUI: ' + str(gui))
    print()

    model = algorithm.DeepDeterministicPolicyGradient(DDPGParams())
    time.sleep(3)

    if load:
        print('loading..')
        model.load_models('actor.h5', 'critic.h5')
        print('loaded!')

    dist_from_start = []
    for p in range(int(episodes/batch)):

        time.sleep(3)
        env = Environment(track=track, track_type=track_type, gui=gui)

        for i in range(batch):
            action = None
            start = time.time()

            print('Episode ' + str(batch*p + i + 1) + '/' + str(episodes))

            for j in range(steps_per_episode):
                # utils.print_progress(j + 1, steps_per_episode)
                action, sensors = env.step(action)

                if env.check_sensors(sensors):
                    # print("out of track!")
                    dist_from_start.append(sensors['distFromStart'])
                    break

                # Encoding of the sensor into a vector
                state_vec = DataHandler.encode_state_data(sensors)

                # Evaluating the corresponding action
                action_vec = model.train_step(state_vec)

                # Decoding the data from prediction vector
                DataHandler.decode_action_data(action, action_vec)

            if i % 10 == 9:
                if save:
                    print('saving models..')
                    model.save_models('actor.h5', 'critic.h5')
                    print('saved!')
                env.restart_environment()
            else:
                env.restart_race()

            print('distFromStart mean: ', np.mean(dist_from_start), 'of', len(dist_from_start))
            print("Episode last {}".format(time.time() - start))
            print()

        print('distFromStart mean: ', np.mean(dist_from_start[-batch:]), 'of last', batch)
        print('Simulation finished!')
        print()

        model.stop()
        env.shutdown()

    greetings()

if __name__ == "__main__":
    # first time set load to false then true
    train(3000000, 10000, 50, gui=False, load=True, save=True)
