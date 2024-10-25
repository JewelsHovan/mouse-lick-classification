import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from numba import njit
from cython_utils import update_sample_weights, compute_loss
import cProfile
import pstats

def map_to_range(predictions, lower=-1, upper=1):
    """
    In-place mapping of predictions to range [lower, upper], with zero division handling.
    """
    min_pred = np.min(predictions)
    max_pred = np.max(predictions)
    if max_pred == min_pred:
        predictions.fill((upper + lower) / 2)
        return predictions
    
    scale = (upper - lower) / (max_pred - min_pred)
    predictions -= min_pred
    predictions *= scale
    predictions += lower
    return predictions

class GentleBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0,
                 min_samples_split=2, min_samples_leaf=1, max_depth=1,
                 validation_fraction=0.1, n_iter_no_change=5, tol=1e-4, verbose=0, random_state=None,
                 sliding_window_size=3, learning_rate_decay=0.1):
        """
        GentleBoost constructor
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.sliding_window_size = sliding_window_size
        self.learning_rate_decay = learning_rate_decay  # New parameter for adaptive learning rate

        self.models = []  # To store weak learners
        self.train_scores = []  # Training losses
        self.val_scores = []    # Validation losses
        self.learning_rates_ = []  # Add this line to store precomputed rates

    def _get_adaptive_learning_rate(self, iteration):
        """
        Calculate adaptive learning rate that decays over iterations.
        """
        return self.learning_rate / (1 + self.learning_rate_decay * iteration)

    def _fit_single_estimator(self, m, X_train, y_train, sample_weights, X_val=None, y_val=None):
        """
        Fit a single estimator with batch processing for large datasets.
        """
        batch_size = 10000  # Adjust based on your memory constraints
        n_samples = X_train.shape[0]
        
        model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state)
        
        if n_samples > batch_size:
            # Process in batches for prediction only
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Batch predictions to save memory
            predictions_train = np.zeros(n_samples, dtype=np.float32)
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                predictions_train[i:batch_end] = model.predict(
                    X_train[i:batch_end]).astype(np.float32)
            
            predictions_val = None
            if X_val is not None:
                n_val_samples = X_val.shape[0]
                predictions_val = np.zeros(n_val_samples, dtype=np.float32)
                for i in range(0, n_val_samples, batch_size):
                    batch_end = min(i + batch_size, n_val_samples)
                    predictions_val[i:batch_end] = model.predict(
                        X_val[i:batch_end]).astype(np.float32)
        else:
            # Original code for smaller datasets
            model.fit(X_train, y_train, sample_weight=sample_weights)
            predictions_train = model.predict(X_train).astype(np.float32)
            predictions_val = model.predict(X_val).astype(np.float32) if X_val is not None else None

        map_to_range(predictions_train)
        if predictions_val is not None:
            map_to_range(predictions_val)
        
        return model, predictions_train, predictions_val

    def fit(self, X, y):
        """
        Memory-optimized fit method.
        """
        # Pre-allocate arrays with proper data types
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Use memory-efficient data structures
        self.models = []
        self.train_scores = np.zeros(self.n_estimators, dtype=np.float32)
        self.val_scores = np.zeros(self.n_estimators, dtype=np.float32)
        self.learning_rates_ = np.zeros(self.n_estimators, dtype=np.float32)
        
        # Pre-compute learning rates
        for m in range(self.n_estimators):
            self.learning_rates_[m] = self._get_adaptive_learning_rate(m)
        
        # Store the unique classes and their original labels
        self.classes_ = np.unique(y)
        
        # Convert y to -1 and 1 for boosting (ensure float32)
        y = np.where(y == self.classes_[0], -1.0, 1.0).astype(np.float32)
        
        n_samples = X.shape[0]

        # Efficient data splitting using numpy operations and random state
        rng = np.random.default_rng(self.random_state)
        if self.validation_fraction > 0.0:
            split_idx = int(n_samples * (1 - self.validation_fraction))
            indices = rng.permutation(n_samples)
            X_train, X_val = X[indices[:split_idx]], X[indices[split_idx:]]
            y_train, y_val = y[indices[:split_idx]], y[indices[split_idx:]]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Pre-allocate arrays
        sample_weights = np.ones(len(y_train), dtype=np.float32) / len(y_train)
        F_train = np.zeros(len(y_train), dtype=np.float32)
        if X_val is not None:
            F_val = np.zeros(len(y_val), dtype=np.float32)

        best_val_loss = np.inf
        no_improvement_count = 0
        val_loss_history = []

        for m in range(self.n_estimators):
            current_lr = self.learning_rates_[m]
            
            model, predictions_train, predictions_val = self._fit_single_estimator(
                m, X_train, y_train, sample_weights, X_val, y_val)

            F_train += current_lr * predictions_train
            sample_weights = update_sample_weights(y_train, predictions_train, current_lr)
            
            self.models.append(model)
            
            loss_train = compute_loss(y_train, F_train)
            self.train_scores[m] = loss_train

            if X_val is not None:
                F_val += current_lr * predictions_val
                loss_val = compute_loss(y_val, F_val)
                self.val_scores[m] = loss_val

                # Sliding window average for validation loss
                val_loss_history.append(loss_val)
                if len(val_loss_history) > self.sliding_window_size:
                    val_loss_avg = np.mean(val_loss_history[-self.sliding_window_size:])
                else:
                    val_loss_avg = np.mean(val_loss_history)

                # Early stopping logic
                if best_val_loss - val_loss_avg > self.tol:
                    best_val_loss = val_loss_avg
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if self.verbose > 0:
                    print(f"Iteration {m + 1}/{self.n_estimators}, "
                          f"Training Loss: {loss_train:.6f}, "
                          f"Validation Loss: {loss_val:.6f} (avg: {val_loss_avg:.6f})")

                if no_improvement_count >= self.n_iter_no_change // 2:
                    if self.verbose > 0:
                        print(f"Early stopping at iteration {m + 1}")
                    break
            else:
                if self.verbose > 0:
                    print(f"Iteration {m + 1}/{self.n_estimators}, "
                          f"Training Loss: {loss_train:.6f}")

        return self

    def _parallel_predict(self, i, model, X):
        """
        Helper function to predict and map to range for parallel execution.
        """
        pred = model.predict(X).astype(np.float32)
        map_to_range(pred)
        return i, pred

    def predict(self, X):
        """
        Parallelized prediction with adaptive parallelization based on problem size.
        """
        X = np.asarray(X, dtype=np.float32)
        F = np.zeros(X.shape[0], dtype=np.float32)

        # Only parallelize if we have enough models and data
        if len(self.models) > 10 and X.shape[0] > 1000:
            predictions = Parallel(n_jobs=-1)(
                delayed(self._parallel_predict)(i, model, X) 
                for i, model in enumerate(self.models)
            )
            
            for i, pred in predictions:
                F += self.learning_rates_[i] * pred
        else:
            # Use simple loop for small problems
            for i, model in enumerate(self.models):
                pred = model.predict(X).astype(np.float32)
                map_to_range(pred)
                F += self.learning_rates_[i] * pred

        return np.where(np.sign(F) == -1, self.classes_[0], self.classes_[1])

    def decision_function(self, X):
        """
        Parallelized decision function with precomputed learning rates.
        """
        X = np.asarray(X, dtype=np.float32)
        F = np.zeros(X.shape[0], dtype=np.float32)

        # Parallel computation of predictions across all models
        predictions = Parallel(n_jobs=-1)(
            delayed(self._parallel_predict)(i, model, X) 
            for i, model in enumerate(self.models)
        )
        
        # Accumulate results
        for i, pred in predictions:
            F += self.learning_rates_[i] * pred

        return F

    def _more_tags(self):
        """
        Used by scikit-learn to understand the capabilities of the estimator.
        """
        return {
            'binary_only': True,  # This estimator only handles binary classification
            'requires_y': True,   # Requires y for fitting
        }

    def _check_n_features(self, X, reset):
        """
        Set or check the n_features_in_ attribute.
        """
        if reset:
            self.n_features_in_ = X.shape[1]
        else:
            if not hasattr(self, 'n_features_in_'):
                self.n_features_in_ = X.shape[1]
            elif self.n_features_in_ != X.shape[1]:
                raise ValueError(f'X has {X.shape[1]} features, but GentleBoost is expecting '
                               f'{self.n_features_in_} features.')

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {
            'base_estimator': self.base_estimator,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_depth': self.max_depth,
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_iter_no_change,
            'tol': self.tol,
            'verbose': self.verbose,
            'random_state': self.random_state,
            'sliding_window_size': self.sliding_window_size,
            'learning_rate_decay': self.learning_rate_decay
        }
        
        if deep and hasattr(self.base_estimator, 'get_params'):
            deep_items = self.base_estimator.get_params().items()
            params.update((f'base_estimator__{key}', val) for key, val in deep_items)
            
        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Estimator instance.
        """
        valid_params = self.get_params(deep=True)
        
        base_estimator_params = {}
        for key, value in params.items():
            if key in valid_params:
                if key == 'base_estimator':
                    self.base_estimator = value
                else:
                    setattr(self, key, value)
            elif key.startswith('base_estimator__'):
                base_estimator_params[key.replace('base_estimator__', '')] = value
            else:
                raise ValueError(f'Invalid parameter {key} for estimator GentleBoost')
                
        if base_estimator_params and self.base_estimator is not None:
            self.base_estimator.set_params(**base_estimator_params)
            
        return self

    def profile_fit(self, X, y):
        """
        Profiled version of fit method.
        """
        profiler = cProfile.Profile()
        profiler.enable()
        
        self.fit(X, y)
        
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats()
        return self

# Example usage
if __name__ == "__main__":
    from utils import test_gentleboost
    model_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'learning_rate_decay': 0.1,
        'max_depth': 3,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'tol': 1e-4,
        'verbose': 1,
        'random_state': 42
    }
    
    results = test_gentleboost(**model_params, profile=True)
    
    print("\nPerformance Metrics:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    
    print("\nTiming Information:")
    print(f"Data Generation Time: {results['data_generation_time']:.2f} seconds")
    print(f"Training Time: {results['training_time']:.2f} seconds")
    print(f"Prediction Time: {results['prediction_time']:.2f} seconds")
    print(f"Total Time: {results['total_time']:.2f} seconds")
    print(f"Number of estimators used: {results['n_estimators_used']}")

