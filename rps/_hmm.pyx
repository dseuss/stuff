#!python
#cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np


cdef inline double rescale(double[:] alpha) nogil:
    cdef double scale = 0.
    cdef long i
    for i in range(alpha.shape[0]):
        scale += alpha[i]

    for i in range(alpha.shape[0]):
        alpha[i] /= scale
    return 1. / scale


cdef inline double dot(double[:] a, double[:] b):
    cdef long i
    cdef double res = 0
    for i in range(a.shape[0]):
        res += a[i] * b[i]
    return res


cdef void get_forward(double[:,:] mkh, double[:,:] mko, double[:] init,
                 long[:] obs, double[:,:] alpha, double[:] scale):
    """Computes the (unscaled) forward variables of the HMM defined by

        alpha_0[i] = init[i] * mko[obs[0], i]
        alpha_t+1[i] = sum_j mkh[ij] * alpha_t[j] * mko[obs[t+1], i]

    where alpha is rescaled after each step alpha_t = alpha_t * c_t
    with the scaling variable

        c_t = 1 / (sum_i alpha_t[i])

    :returns: (len(obs), len(mkh)) array, (len(obs)) array;
        forward and scaling variables

    """
    cdef long steps = obs.shape[0]
    cdef long n_hid = mkh.shape[0]
    cdef long i, t

    for i in range(n_hid):
        alpha[0, i] = init[i] * mko[obs[0], i]
    scale[0] = rescale(alpha[0])

    for t in range(steps - 1):
        for i in range(n_hid):
            alpha[t+1, i] = dot(mkh[i], alpha[t]) * mko[obs[t+1], i]
        scale[t+1] = rescale(alpha[t+1, :])


cdef void get_backward(double[:,:] mkh, double[:,:] mko, double[:] init,
                       long[:] obs, double[:] scale, double[:,:] beta):
    """Computes the (unscaled) backward variables of the HMM defined by

        beta_T-1[i] = 1
        beta_t[i] = sum_j mko[obs[t+1], j] beta_t+1[j] mkh[ji]

    where T = len(obs)

    :returns: (len(obs), len(mkh)) array; backward variables

    """
    cdef long steps = obs.shape[0]
    cdef long n_hid = mkh.shape[0]
    cdef long i, j, t

    beta[:,:] = 0
    beta[steps, :] = scale[steps]
    for t in range(steps - 2, -1, -1):
        for i in range(n_hid):
            beta[t, i] = 0
            for j in range(n_hid):
                beta[t, i] += mko[obs[t+1], j] * beta[t+1, j] * mkh[j, i]
            beta[t, i] = beta[t, i] * scale[t]

def iterate(double[:,:] mkh, double[:,:] mko, double[:] init, long[:] obs):
    cdef long steps = obs.shape[0]
    cdef long n_hid = mkh.shape[0]
    cdef long n_obs = mko.shape[0]
    cdef double alpha[steps][n_hid]
    cdef double scale[steps]
    cdef double beta[steps][n_hid]

    get_forward(mkh, mko, init, obs, alpha, scale)
    get_backward(mkh, mko, init, obs, scale, beta)



