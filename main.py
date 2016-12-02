from track_utilities import TrackUtilities
from utilities.reward_writer import RewardWriter

if __name__ == "__main__":
    # TrackUtilities.curriculum_learning_on_track('eroad',
    #                                             'eroad',
    #                                             initial_speed=130,
    #                                             max_speed=250,
    #                                             speed_step=10,
    #                                             n_lap=3,
    #                                             validation_lap_number=10)
    #
    #
    # rw = RewardWriter('runs/forza/rewards.csv')
    #
    # TrackUtilities.validate_network('runs/forza/forza_speed110.h5f',
    #                                 'forza',
    #                                 110,
    #                                 rw,
    #                                 n_lap=10)

    # TrackUtilities.curriculum_learning_on_track('dirt-2',
    #                                             'dirt-2',
    #                                             initial_speed=60,
    #                                             max_speed=250,
    #                                             speed_step=10,
    #                                             n_lap=3,
    #                                             validation_lap_number=10)

    TrackUtilities.curriculum_learning_on_track('aalborg',
                                                'aalborg',
                                                initial_speed=70,
                                                max_speed=250,
                                                speed_step=10,
                                                n_lap=3,
                                                validation_lap_number=10)

    # TrackUtilities.test_ensemble(['runs/a-speedway_curriculum/0_a-speedway_speed300_actor.h5f',
    #                               'runs/alpine-1_curriculum/0_alpine-1_speed100_actor.h5f',
    #                               'runs/corkscrew_curriculum/backup/0_corkscrew_speed180_actor.h5f',
    #                               'runs/e-track-5_curriculum/0_e-track-5_speed300_actor.h5f',
    #                               'runs/g-track-1_curriculum/0_g-track-1_speed210_actor.h5f',
    #                               'runs/michigan_curriculum/0_michigan_speed310_actor.h5f',
    #                               'runs/corkscrew_curriculum/backup/0_corkscrew_speed170_actor.h5f',
    #                               'runs/corkscrew_curriculum/backup/0_corkscrew_speed160_actor.h5f',
    #                               ],
    #                              'alpine-1', 3)
