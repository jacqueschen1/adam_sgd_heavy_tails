import numpy as np
import glob
from scipy.stats import levy_stable
import sys


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


sys.stdout = Logger()

for np_name in glob.glob("*.np[yz]"):
    norm = np.load(np_name)
    print(np_name)
    norm = norm[:-1]
    norm = norm - norm.mean()
    print(levy_stable.fit(norm))
