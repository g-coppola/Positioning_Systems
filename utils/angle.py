import numpy as np

def normalize_angle(angle):
    return (angle+np.pi)%(2*np.pi)-np.pi
