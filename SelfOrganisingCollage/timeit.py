import time

# Execution Timing
class TimeIt:
    def __init__(self, s):
        self.s = s

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        print('%s: %s' % (self.s, time.time() - self.t0))
