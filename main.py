from track_utilities import TrackUtilities

if __name__ == "__main__":
    # TrackUtilities.train_on_all_tracks()
    TrackUtilities.train_on_single_track('e-track-2_train',
                                         steps=803552,
                                         noise=0.8034569999943483,
                                         epsilon=0.5,
                                         load=True,
                                         load_filepath='runs/e-track-2_train/alpine-1_0.5.h5f',
                                         track='alpine-1')
    # TrackUtilities.train_on_chosen_tracks(['aalborg', 'a-speedway', 'e-track-6'], [0.5, 0.2, 0], 100000, 'three_tracks')
    #TrackUtilities.test_network('aalborg', 'runs/aalborg_train/aalborg_0.5.h5f')
