import time
from functools import wraps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from typing import List, Dict, Any
from sklearn.model_selection import GridSearchCV


class ClassifierTrainer:
    """
    A class to train multiple classifiers, evaluate their performance, and compare the results.
    """
    def __init__(self, results_csv_path: str = None):
        """
        Initializes the ClassifierTrainer with an empty results DataFrame.

        :param results_csv_path: Optional path to save the results DataFrame as a CSV file.
        """
        self.results_df = pd.DataFrame(columns=[
            'Model', 
            'Cross-Validation Accuracy', 
            'Training Accuracy', 
            'Test Accuracy', 
            'Precision', 
            'Recall',
            'Execution Time'
        ])
        self.classifiers = []
        self.results_csv_path = results_csv_path

    def log_execution_time(func):
        """
        Decorator to log the execution time of a function.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time for {func.__name__}: {execution_time:.4f} seconds")
            result['execution_time'] = execution_time
            return result
        return wrapper

    def save_results(func):
        """
        Decorator to save the results of training and evaluation.
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            estimator = args[4]  # Assuming the estimator is the 5th argument

            if isinstance(estimator, Pipeline):
                model_name = estimator.steps[-1][1].__class__.__name__
            else:
                model_name = estimator.__class__.__name__

            cv_accuracy = result['cv_accuracy']
            train_accuracy = result['train_accuracy']
            test_accuracy = result['test_accuracy']
            precision = result['precision']
            recall = result['recall']
            execution_time = result['execution_time']

            current_result_df = pd.DataFrame({
                'Model': [model_name],
                'Cross-Validation Accuracy': [cv_accuracy],
                'Training Accuracy': [train_accuracy],
                'Test Accuracy': [test_accuracy],
                'Precision': [precision],
                'Recall': [recall],
                'Execution Time': [execution_time]
            })

            self.results_df = pd.concat([self.results_df, current_result_df], ignore_index=True)

            if self.results_csv_path:
                self.results_df.to_csv(self.results_csv_path, index=False)

            return result
        return wrapper

    @save_results
    @log_execution_time
    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray, 
                           estimator: BaseEstimator, cv: int = 5) -> Dict[str, Any]:
        """
        Trains and evaluates a single classifier.

        :param X_train: Training features.
        :param y_train: Training labels.
        :param X_test: Testing features.
        :param y_test: Testing labels.
        :param estimator: The classifier to train.
        :param cv: Number of cross-validation folds.
        :return: A dictionary containing evaluation metrics.
        """
        cv_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='accuracy')
        cv_accuracy = cv_scores.mean()
        print(f"Cross-Validation Accuracy: {cv_accuracy:.4f} ± {cv_scores.std():.4f}")

        estimator.fit(X_train, y_train)

        train_accuracy = estimator.score(X_train, y_train)
        print(f"Training Accuracy: {train_accuracy:.4f}")

        test_accuracy = estimator.score(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy:.4f}")

        y_pred = estimator.predict(X_test)

        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        return {
            'cv_accuracy': cv_accuracy,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall
        }

    def add_classifier(self, estimator: BaseEstimator):
        """
        Adds a classifier to the list of classifiers to be trained.

        :param estimator: The classifier to add.
        """
        self.classifiers.append(estimator)

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_test: np.ndarray, y_test: np.ndarray):
        """
        Trains and evaluates all added classifiers.

        :param X_train: Training features.
        :param y_train: Training labels.
        :param X_test: Testing features.
        :param y_test: Testing labels.
        """
        for clf in self.classifiers:
            print(f"\nTraining and evaluating {clf.__class__.__name__}...")
            self.train_and_evaluate(X_train, y_train, X_test, y_test, clf)

    def plot_results(self, baseline_accuracy: float = None):
        """
        Plots the cross-validation accuracy of all classifiers.

        :param baseline_accuracy: Optional baseline accuracy to compare against (e.g., Dummy Classifier).
        """
        sorted_results = self.results_df.sort_values(by='Cross-Validation Accuracy', ascending=False)
        
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='Model', y='Cross-Validation Accuracy', data=sorted_results, palette="viridis")
        
        # Add text labels on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='baseline', 
                        fontsize=10, color='black', 
                        xytext=(0, 5), 
                        textcoords='offset points')
        
        if baseline_accuracy is not None:
            plt.axhline(y=baseline_accuracy, color='red', linestyle='dotted', label='Dummy Classifier Accuracy')
            plt.legend()
        
        plt.title('Cross-Validation Accuracy Comparison') 
        plt.xlabel('Model')
        plt.ylabel('Cross-Validation Accuracy')
        plt.ylim(min(sorted_results['Cross-Validation Accuracy']) - 0.05, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @save_results
    @log_execution_time
    def train_and_evaluate_with_gridsearch(self, X_train: np.ndarray, y_train: np.ndarray, 
                                           X_test: np.ndarray, y_test: np.ndarray, 
                                           estimator: BaseEstimator, param_grid: Dict[str, List[Any]], 
                                           cv: int = 5) -> Dict[str, Any]:
        """
        Trains and evaluates a classifier using GridSearchCV for hyperparameter tuning.

        :param X_train: Training features.
        :param y_train: Training labels.
        :param X_test: Testing features.
        :param y_test: Testing labels.
        :param estimator: The classifier to train.
        :param param_grid: Grid of hyperparameters to search.
        :param cv: Number of cross-validation folds.
        :return: A dictionary containing evaluation metrics.
        """
        grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")

        # Proceed with evaluation using the best estimator
        return self.train_and_evaluate(X_train, y_train, X_test, y_test, best_estimator, cv)