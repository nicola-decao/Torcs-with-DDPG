class RewardWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.f = open(filepath, 'a')

    def write_track(self, track, epsilon):
        print(track + ' ' + str(epsilon), file=self.f)
        self.flush()

    def completed_track(self):
        print('', file=self.f)
        self.flush()

    def write_reward(self, episode_reward, steps):
        print("{0:.4f}".format(episode_reward) + ', ' + str(steps) + ', ' + "{0:.4f}".format(episode_reward / steps),
              file=self.f)
        self.flush()

    def bad_run(self):
        print('BAD RUN BAD RUN BAD RUN BAD RUN', file=self.f)
        self.flush()

    def flush(self):
        self.f.flush()
