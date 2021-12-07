import numpy as np

def fs(value):
    return "{:.3f}".format(value)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]