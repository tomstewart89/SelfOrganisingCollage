import time


def flatten(list_of_lists):
    return [item for l in list_of_lists for item in l]


# Execution Timing
class TimeIt:
    def __init__(self, s):
        self.s = s

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        print('%s: %s' % (self.s, time.time() - self.t0))
