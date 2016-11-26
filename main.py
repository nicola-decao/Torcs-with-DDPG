from track_utilities import TrackUtilities

if __name__ == "__main__":
    # TrackUtilities.train_on_all_tracks()
    TrackUtilities.train_on_single_track('aalborg_train',
                                         steps=1000000,
                                         noise=0.5,
                                         load=True,
                                         load_filepath='runs/aalborg_train/aalborg_0.5.h5f')
    # TrackUtilities.train_on_chosen_tracks(['aalborg', 'a-speedway', 'e-track-6'], [0.5, 0.2, 0], 100000, 'three_tracks')
    #TrackUtilities.test_network('aalborg', 'runs/aalborg_train/backup/first_network/backup_2/aalborg_0.5.h5f')
