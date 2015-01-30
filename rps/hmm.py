#!/usr/bin/env python
# encoding: utf-8
"""Finite hidden markov models"""

from __future__ import division, print_function
import numpy as np

from numpy.random import rand
try:
    from tools.helpers import Progress
except ImportError:
    Progress = lambda *args: args


def _stoch_map(mat):
    """Normalizes the array `mat` such that the sum over the first dimension
    yields an array with ones on the diagonal. Blows up if that sum is 0
    somewhere for mat.

    :param mat: Matrix to be normalized
    :returns: Matrix of same shape as `mat` but normalized

    """
    normalization = np.sum(mat, axis=0)
    return mat / normalization[None]


def _random_index(p):
    """Randomly returns an index in range(len(p)) according to the probabilties
    given by p. If p is not normalized to sum_i p[i] = 1, we normalize
    beforehand.

    :param p: array; p[i] is the probability of getting i
    :returns: int in range(len(p))

    """
    xs = np.cumsum(p)
    return np.argmax(xs >= np.random.rand() * xs[-1])


class HMM(object):
    """Docstring for HMM. """

    def __init__(self, n_obs, n_hid):
        """

        :param n_obs: int; Number of observable states
        :param n_hid: int; Number of hidden states

        """
        self._n_obs = n_obs
        self._n_hid = n_hid
        self.initialize()

    def initialize(self, mkh=None, mko=None, init=None):
        """@todo: Docstring for initialize.

        :param mkh: (n_hid, n_hid) array; Markov kernel for the hidden degrees
            of freedom
        :param mko: (n_obs, n_hid) array; Markov kernel for the observable
            degrees of freedom
        :param init: (n_hid) array; Initial pmf for the hidden degrees of
            freedom

        """
        self._mkh = _stoch_map(mkh if mkh is not None else
                               rand(self._n_hid, self._n_hid))
        self._mko = _stoch_map(mko if mko is not None else
                               rand(self._n_obs, self._n_hid))
        self._init = _stoch_map(init if init is not None else
                                rand(self._n_hid))

    @staticmethod
    def _get_forward(mkh, mko, init, obs):
        """Computes the (unscaled) forward variables of the HMM defined by

            alpha_0[i] = init[i] * mko[obs[0], i]
            alpha_t+1[i] = sum_j mkh[ij] * alpha_t[j] * mko[obs[t+1], i]

        where alpha is rescaled after each step alpha_t = alpha_t * c_t
        with the scaling variable

            c_t = 1 / (sum_i alpha_t[i])

        :returns: (len(obs), len(mkh)) array, (len(obs)) array;
            forward and scaling variables

        """
        steps, n_hid = len(obs), len(mkh)
        alpha = np.empty((steps, n_hid))
        alpha[0] = init * mko[obs[0]]

        # FIXME Make scaling adaptive
        scale = np.empty(steps)
        scale[0] = 1. / np.sum(alpha[0])
        alpha[0] *= scale[0]

        for t in xrange(steps - 1):
            alpha[t+1] = np.dot(mkh, alpha[t]) * mko[obs[t+1]]
            scale[t+1] = 1. / np.sum(alpha[t+1])
            alpha[t+1] *= scale[t+1]

        return alpha, scale

    @staticmethod
    def _get_backward(mkh, mko, init, obs, scale):
        """Computes the (unscaled) backward variables of the HMM defined by

            beta_T-1[i] = 1
            beta_t[i] = sum_j mko[obs[t+1], j] beta_t+1[j] mkh[ji]

        where T = len(obs)

        :returns: (len(obs), len(mkh)) array; backward variables

        """
        steps, n_hid = len(obs), len(mkh)
        beta = np.zeros((steps, n_hid))
        beta[-1] = scale[-1]
        for t in xrange(steps - 2, -1, -1):
            beta[t] = np.dot(mko[obs[t+1]] * beta[t+1], mkh) * scale[t]

        return beta

    @staticmethod
    def _iterate(mkh, mko, init, obs):
        """@todo: Docstring for _iterate.

        :returns: New values for mkh, mko, init

        """
        forward, scale = HMM._get_forward(mkh, mko, init, obs)
        backward = HMM._get_backward(mkh, mko, init, obs, scale)
        gamma = forward * backward
        gamma /= np.sum(gamma, axis=-1)[:, None]
        xi = forward[:-1, None, :] * mkh[None, :, :] \
            * mko[obs[1:], :, None] * backward[1:, :, None]
        xi /= np.sum(xi, axis=(1, 2))[:, None, None]

        init_n = gamma[0]
        mkh_n = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0, keepdims=True)
        mko_n = np.empty((len(mko), len(mkh)))
        gamma_sum = np.sum(gamma, axis=0)
        for k in range(len(mko)):
            mko_n[k, :] = np.sum(gamma[obs == k], axis=0) / gamma_sum

        return mkh_n, mko_n, init_n

    def fit(self, obs, rel=1e-3, max_iter=1000):
        """Fits the model to the observation sequence `obs`.

        :param obs: (steps) array of integers in range(n_obs); observation
            series to fit to

        """
        # TODO Addept to multiple chains
        norm = lambda x: np.max(np.abs(x))
        for i in Progress(xrange(max_iter)):
            mkh, mko, init = self._iterate(self._mkh, self._mko, self._init,
                                           obs)
            if norm(mkh - self._mkh) < rel and \
                    norm(mko - self._mko) < rel and \
                    norm(init - self._init) < rel:
                return

            self._mkh, self._mko, self._init = mkh, mko, init
        # FIXME
        print('Warning: Not converged!!!!!')

    def get_sequence(self, length):
        """Returns a sequence of given length of observable events according
        to the current model.

        :param length: int; length of sequence
        :returns: (length) array of int in range(n_obs)

        """
        obs = np.empty(length, dtype=int)
        state = self._init

        for t in range(length):
            obs[t] = _random_index(np.dot(self._mko, state))
            state = np.dot(self._mkh, state)

        return obs


if __name__ == '__main__':
    import matplotlib.pyplot as pl
    n_hid = 4
    n_obs = 3
    steps = 2500

    hmm_ref = HMM(n_obs, n_hid)
    hmm_test = HMM(n_obs, n_hid)
    norm = lambda x: np.sum(x * x) / np.size(x)

    for steps in [500, 1000, 2000, 5000]:
        obs = hmm_ref.get_sequence(steps)
        hmm_test.initialize(mkh=hmm_ref._mkh, mko=hmm_ref._mko,
                            init=hmm_ref._init, )
        hmm_test.fit(obs, rel=1e-4, max_iter=10000)
        norms = [norm(hmm_ref._mkh - hmm_test._mkh),
                 norm(hmm_ref._mko - hmm_test._mko),
                 norm(hmm_ref._init - hmm_test._init)]
        pl.scatter([steps] * 3, norms, color=('r', 'g', 'b'))
        print("")

    pl.show()
