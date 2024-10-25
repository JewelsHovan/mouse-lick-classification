import numpy as np
from find_best_split import find_best_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_array
from typing import Optional

class DecisionStump(BaseEstimator, RegressorMixin):
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize DecisionStump with parameters that can be set at initialization.
        Internal attributes should not be included here.
        """
        self.random_state = random_state
        # These are not initialization parameters, so they should be set with underscore prefix
        self._feature_index = None
        self._threshold = None
        self._left_value = None
        self._right_value = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit a decision stump to the data using optional sample weights.
        """
        # Input validation and conversion
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if sample_weight is None:
            sample_weight = np.ones_like(y, dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        # Validate shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if sample_weight.shape[0] != y.shape[0]:
            raise ValueError("sample_weight must have the same number of samples as y.")
        if X.size == 0:
            raise ValueError("X cannot be empty.")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        best_feature = None
        best_threshold = None
        best_error = float('inf')
        best_left_value = None
        best_right_value = None
        
        # Set random state if provided
        rng = np.random.default_rng(self.random_state)
        feature_order = rng.permutation(n_features) if self.random_state is not None else range(n_features)
        
        # Iterate over features in random order if random_state is set
        for feature in feature_order:
            X_feature = X[:, feature]
            thresholds, errors, left_values, right_values = find_best_split(X_feature, y, sample_weight)
            
            # Only proceed if valid splits were found
            if len(thresholds) > 0 and len(errors) > 0:
                min_error_idx = np.argmin(errors)
                if errors[min_error_idx] < best_error:
                    best_error = errors[min_error_idx]
                    best_feature = feature
                    best_threshold = thresholds[min_error_idx]
                    best_left_value = left_values[min_error_idx]
                    best_right_value = right_values[min_error_idx]

        # If no valid split was found, use the mean of y as a constant predictor
        if best_threshold is None:
            self._feature_index = 0
            self._threshold = 0
            self._left_value = self._right_value = np.average(y, weights=sample_weight)
        else:
            self._feature_index = best_feature
            self._threshold = best_threshold
            self._left_value = best_left_value
            self._right_value = best_right_value

        return self

    def predict(self, X):
        """
        Predict values for the input X based on the fitted stump.
        """
        check_is_fitted(self, ['_feature_index', '_threshold', '_left_value', '_right_value'])
        X = check_array(X, dtype=np.float32)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {X.shape[1]} features, but DecisionStump '
                           f'was trained with {self.n_features_in_} features.')
        
        feature_values = X[:, self._feature_index]
        predictions = np.where(feature_values <= self._threshold, 
                             self._left_value, self._right_value)
        return predictions

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Only return initialization parameters, not fitted attributes.
        """
        return {'random_state': self.random_state}

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _more_tags(self):
        return {
            'requires_fit': True,
            'allow_nan': False,
            'binary_only': False
        }
