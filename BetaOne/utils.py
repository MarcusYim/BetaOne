import sys, os, random, time, warnings
from hashlib import sha1

import numpy as np
from numpy import all, array, uint8


def sample(probs, T=1.0):
    keys = list(probs.keys())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = np.exp(np.array([probs[key] for key in keys]) / T).clip(0, 1e9)
    return keys[np.random.choice(len(keys), p=values / values.sum())]
