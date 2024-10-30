# cython_utils.pyx

import numpy as np
cimport numpy as np
from libc.math cimport exp

def update_sample_weights(np.ndarray[np.float64_t, ndim=1] y_train, 
                         np.ndarray[np.float64_t, ndim=1] predictions,
                         double learning_rate):
    cdef int n = y_train.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] weights = np.empty(n, dtype=np.float64)
    cdef double x
    cdef double sum_weights = 0.0
    cdef int i

    for i in range(n):
        x = learning_rate * y_train[i] * predictions[i]
        weights[i] = 1.0 / (1.0 + exp(x))
        sum_weights += weights[i]

    # Normalize weights
    for i in range(n):
        weights[i] /= sum_weights

    return weights

def compute_loss(np.ndarray[np.float64_t, ndim=1] y,
                np.ndarray[np.float64_t, ndim=1] F):
    cdef int n = y.shape[0]
    cdef double total_loss = 0.0
    cdef double x
    cdef int i

    for i in range(n):
        x = y[i] * F[i]
        total_loss += 1.0 / (1.0 + exp(x))

    return total_loss / n