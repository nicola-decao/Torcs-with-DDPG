from track_utilities import TrackUtilities
from utilities.reward_writer import RewardWriter

if __name__ == "__main__":
    # reward = RewardWriter('test.test')
    # TrackUtilities.test_network('aalborg', 'aalborg_speed60_actor.h5f', n_lap=3)
    # TrackUtilities.curriculum_learning_on_track('alpine-1',
    #                                             'alpine-1_curriculum',
    #                                             initial_speed=150,
    #                                             max_speed=250,
    #                                             speed_step=20,
    #                                             n_lap=1,
    #                                             validation_lap_number=3)
    # TrackUtilities.curriculum_learning_on_track('corkscrew',
    #                                             'corkscrew_curriculum',
    #                                             initial_speed=140,
    #                                             max_speed=250,
    #                                             speed_step=20,
    #                                             n_lap=2,
    #                                             validation_lap_number=4)
    TrackUtilities.curriculum_learning_on_track('alpine-2',
                                                'alpine-2_curriculum',
                                                initial_speed=130,
                                                max_speed=250,
                                                speed_step=20,
                                                n_lap=2,
                                                validation_lap_number=5)
    # TrackUtilities.test_ensemble(['tested/b-speedway.h5f',
    #                              'tested/alpine-1.h5f',
    #                               'tested/corkscrew.h5f',
    #                               'tested/street-1.h5f',
    #                               'tested/e-track-4.h5f',
    #                               ],
    #                               track='brondehach',
    #                              n_lap=3)
    # TrackUtilities.test_network('g-track-1', 'runs/validated/g-track-1_curriculum/g-track-1_speed240_validated_actor.h5f', 3)
