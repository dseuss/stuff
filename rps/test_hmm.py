#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
from hmm import _stoch_map, HMM

n_hid = 4
n_obs = 3
steps = 10

hmm_ref = HMM(n_obs, n_hid)
hmm_test = HMM(n_obs, n_hid)

obs = hmm_ref.get_sequence(steps)
hmm_test.fit(obs)

norm = lambda x: np.max(np.abs(x))
print(norm(hmm_ref._mkh - hmm_test._mkh))
print(norm(hmm_ref._mko - hmm_test._mko))
print(norm(hmm_ref._init - hmm_test._init))
