import numpy as np

class EuroCall():
    def __init__(self, ST, K, dt, r):
        self.ST = ST
        self.K = K
        self.dt = dt
        self.r = r

    def payoff(self):
        C = np.exp(-self.r * self.dt) * np.maximum(self.ST - self.K, 0)
        return C