#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
from numpy.random import rand, seed
from time import time
import cyvec

if __name__ == '__main__':
    seed(1337)
    # Generate 5000 50x50 symmetric matrices
    matrices = np.asarray([A + np.transpose(A) for A in rand(50000, 10, 10)])

    t1 = time()
    evs = cyvec.eigenvalue_sums_preallocated(matrices)
    print('Preallocated | Result : {}, Time: {}'.format(np.sum(evs), time() - t1))

    t1 = time()
    evs = cyvec.eigenvalue_sums_vector(matrices)
    print('Vector       | Result : {}, Time: {}'.format(np.sum(evs), time() - t1))
