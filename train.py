import time
import numpy as np
from algorithm import algorithm
from environment import Environment
from parameters import DDPGParams, DataHandler
from utilities.tracks_utils import track_list, greetings


def train(episodes, steps_per_episode, gui=True, load=False, save=True):
    try:
        track = 'ole-road-1'
        track_type = track_list[track]

        print('Starting simulation...')
        print('Track: ' + track)
        print('Track type: ' + track_type)
        print('GUI: ' + str(gui))
        print()

        model = algorithm.DeepDeterministicPolicyGradient(DDPGParams())

        if load:
            model.load_models('actor.h5', 'critic.h5')

        time.sleep(6)
        env = Environment(track=track, track_type=track_type, gui=gui)
        distFromStart = []

        c=0

        for i in range(episodes):
            action = None
            start = time.time()

            print('Episode ' + str(i + 1) + '/' + str(episodes))

            for j in range(steps_per_episode):
                # utils.print_progress(j + 1, steps_per_episode)
                action, sensors = env.step(action)

                if env.check_sensors(sensors):
                    # print("out of track!")
                    distFromStart.append(sensors['distFromStart'])
                    break

                # Encoding of the sensor into a vector
                state_vec = DataHandler.encode_state_data(sensors)

                # Evaluating the corresponding action
                action_vec = model.train_step(state_vec)

                # Decoding the data from prediction vector
                DataHandler.decode_action_data(action, action_vec)


            if c%10==0:
                c=1
                if save:
                    print('saving models..')
                    model.save_models('actor.h5', 'critic.h5')
                    print('saved!')
                env.restart_environment()
            print('distFromStart mean: ', np.mean(distFromStart))
            print("Episode last {}".format(time.time()-start))
            print()
            c+=1
            env.restart_race()
            #env.restart_environment()

        print('Simulation finished!')
        print()
        model.stop()

        env.shutdown()
        greetings()
    except:
        train(3000000, 10000, gui=False, load=True, save=True)

if __name__ == "__main__":
    train(3000000, 10000, gui=False, load=False, save=True)
