import numpy as np

class Oracle:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.queried = []

    def query(self, loc):
        self.queried.append(loc)
        return self.y(loc)

    def random_query(self):
        while True:
            loc = np.random.random_integers(0, len(self.X)-1, 1)[0]
            if loc not in self.queried:
                self.queried.append(loc)
                break
        return loc, self.y[loc]