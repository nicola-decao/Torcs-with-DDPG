from track_utilities import TrackUtilities

if __name__ == "__main__":
    TrackUtilities.curriculum_learning_on_track('e-track-2',
                                                'e-track-2_curriculum',
                                                initial_speed=80,
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
