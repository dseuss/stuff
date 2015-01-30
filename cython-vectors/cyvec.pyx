"""
Testing of cython's parallel support by calculating the sum of eigenvectors
in parallel.
"""
from __future__ import division, print_function

import numpy as np
cimport numpy as np
cimport cython

from cython_gsl cimport *
from libcpp.vector cimport vector
from libc.string cimport memcpy



@cython.boundscheck(False)
cdef double get_evsum(double[:, :] A) nogil:
    """Calculates the sum of the eigenvalues of the symmetric matrix A"""
    if A.shape[0] != A.shape[1]:
        with gil:
            raise IndexError('Matrix not square.')

    cdef size_t dim = A.shape[0]
    cdef gsl_matrix_view mview
    cdef gsl_vector *evals
    cdef gsl_matrix *evecs
    cdef gsl_eigen_symmv_workspace *ws

    mview = gsl_matrix_view_array(&A[0, 0], dim, dim)
    evals = gsl_vector_alloc(dim)
    evecs = gsl_matrix_alloc(dim, dim)
    ws = gsl_eigen_symmv_alloc(dim)
    gsl_eigen_symmv (&(mview.matrix), evals, evecs, ws)

    cdef double result = 0.
    for i in range(dim):
        result += gsl_vector_get(evals, i)
    gsl_vector_free(evals)
    gsl_matrix_free(evecs)
    gsl_eigen_symmv_free(ws)

    return result

def eigenvalue_sums_preallocated(double[:, :, :] matrices):
    """Calculates the sums of the eigenvalues of symmetric matrices"""
    cdef size_t nr_matrices = matrices.shape[0]
    cdef double[:] evsums = np.empty(nr_matrices, dtype=np.float64)
    cdef int i

    for i in range(nr_matrices):
        evsums[i] = get_evsum(matrices[i])

    return evsums

def eigenvalue_sums_vector(double[:, :, :] matrices):
    """Calculates the sums of the eigenvalues of symmetric matrices"""
    cdef size_t nr_matrices = matrices.shape[0]
    cdef vector[double] evsums
    cdef int i

    for i in range(nr_matrices):
        evsums.push_back(get_evsum(matrices[i]))

    result = np.empty(evsums.size())
    cdef double[:] view = result
    # cdef double[:] evsums_view = <double[:evsums.size()]> &evsums[0]
    memcpy(&view[0], &evsums[0], sizeof(double) * evsums.size())
    # cdef double[:] result_view = result
    # result_view = evsums_view.copy()

    return result
