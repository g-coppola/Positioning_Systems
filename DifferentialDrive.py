import numpy as np
from utils.angle import normalize_angle

class DifferentialDrive:
    def f(self, X, U):
        Xdot = np.copy(X)
        Xdot[0, :] += U['t'] * np.cos(X[2, :] + U['r1'])
        Xdot[1, :] += U['t'] * np.sin(X[2, :] + U['r1'])
        Xdot[2, :] += U['r1'] + U['r2']
        Xdot[2, :] = normalize_angle(Xdot[2, :])
        return Xdot

    def g(self, X, mx, my):
        r = np.sqrt((mx - X[0, :]) ** 2 + (my - X[1, :]) ** 2)
        phi = normalize_angle(np.arctan2(my - X[1, :], mx - X[0, :]) - X[2, :])
        return np.vstack((r, phi))