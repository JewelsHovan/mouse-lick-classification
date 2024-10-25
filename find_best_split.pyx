# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np

# Define the dtype explicitly
np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

from libc.math cimport INFINITY
from cython.parallel cimport prange

cpdef find_best_split(np.ndarray[DTYPE_t, ndim=1] X_feature,
                     np.ndarray[DTYPE_t, ndim=1] y,
                     np.ndarray[DTYPE_t, ndim=1] sample_weight):
    """Find the best split point for a feature.

    Parameters
    ----------
    X_feature : ndarray of shape (n_samples,)
        Feature values
    y : ndarray of shape (n_samples,)
        Target values
    sample_weight : ndarray of shape (n_samples,)
        Sample weights

    Returns
    -------
    tuple of ndarrays
        Returns (thresholds, errors, left_values, right_values)
    """
    cdef:
        Py_ssize_t n_samples = X_feature.shape[0]
        Py_ssize_t i, split_idx = 0
        DTYPE_t threshold, error
        DTYPE_t left_weight_sum, right_weight_sum
        DTYPE_t left_sum_y, right_sum_y
        DTYPE_t left_sum_y2, right_sum_y2
        DTYPE_t left_value, right_value

    # Input validation
    if n_samples == 0:
        return (np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64))

    # Sort arrays
    sort_idx = np.argsort(X_feature)
    cdef:
        np.ndarray[DTYPE_t, ndim=1] sorted_feature = X_feature[sort_idx]
        np.ndarray[DTYPE_t, ndim=1] sorted_y = y[sort_idx]
        np.ndarray[DTYPE_t, ndim=1] sorted_weights = sample_weight[sort_idx]

    # Pre-compute weighted values using NumPy operations
    cdef:
        np.ndarray[DTYPE_t, ndim=1] weighted_y = sorted_y * sorted_weights
        np.ndarray[DTYPE_t, ndim=1] weighted_y2 = sorted_y * sorted_y * sorted_weights
        np.ndarray[DTYPE_t, ndim=1] cum_weights = np.cumsum(sorted_weights)
        np.ndarray[DTYPE_t, ndim=1] cum_weighted_y = np.cumsum(weighted_y)
        np.ndarray[DTYPE_t, ndim=1] cum_weighted_y2 = np.cumsum(weighted_y2)
        DTYPE_t total_weight = cum_weights[n_samples - 1]
        DTYPE_t total_weighted_y = cum_weighted_y[n_samples - 1]
        DTYPE_t total_weighted_y2 = cum_weighted_y2[n_samples - 1]

    # Find unique values and thresholds
    unique_vals = np.unique(sorted_feature)
    if len(unique_vals) <= 1:
        return (np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64))

    # Calculate midpoints between unique values
    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
    cdef Py_ssize_t n_thresholds = len(thresholds)
    
    # Initialize arrays with INFINITY
    errors = np.full(n_thresholds, INFINITY, dtype=np.float64)
    left_values = np.zeros(n_thresholds, dtype=np.float64)
    right_values = np.zeros(n_thresholds, dtype=np.float64)

    # Find split points
    for i in range(n_thresholds):
        threshold = thresholds[i]
        
        # Find split point
        while split_idx < n_samples and sorted_feature[split_idx] <= threshold:
            split_idx += 1
        
        if split_idx == 0 or split_idx == n_samples:
            continue

        # Calculate left and right statistics using cumulative sums
        left_weight_sum = cum_weights[split_idx - 1]
        right_weight_sum = total_weight - left_weight_sum
        
        if left_weight_sum > 0 and right_weight_sum > 0:
            left_sum_y = cum_weighted_y[split_idx - 1]
            right_sum_y = total_weighted_y - left_sum_y
            
            left_sum_y2 = cum_weighted_y2[split_idx - 1]
            right_sum_y2 = total_weighted_y2 - left_sum_y2
            
            left_value = left_sum_y / left_weight_sum
            right_value = right_sum_y / right_weight_sum
            
            # Calculate weighted MSE using pre-computed values
            left_error = left_sum_y2 - (left_sum_y * left_sum_y) / left_weight_sum
            right_error = right_sum_y2 - (right_sum_y * right_sum_y) / right_weight_sum
            
            error = left_error + right_error
            
            errors[i] = error
            left_values[i] = left_value
            right_values[i] = right_value

    # Filter out invalid splits (those with INFINITY error)
    valid_mask = errors < INFINITY
    return (thresholds[valid_mask],
            errors[valid_mask],
            left_values[valid_mask],
            right_values[valid_mask])
