import os
import h5py
import numpy as np
from time import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compare_dimensions(train_data_element, train_folder, label_folder, verbose=False):
    """
    Compare the dimensions of the dataset and labels for a given training data element.

    Args:
        train_data_element (tuple): A tuple containing the file name of the training data and the file name of the label.
        train_folder (str): The path to the folder containing the training data.
        label_folder (str): The path to the folder containing the label data.
        verbose (bool, optional): If True, print the shape of the dataset and labels. Defaults to False.
    """
    # Extract file paths from the train_data element
    sample_train_file = os.path.join(train_folder, train_data_element[0])
    sample_label_file = os.path.join(label_folder, train_data_element[1])

    # Open the .h5 file for the dataset
    with h5py.File(sample_train_file, 'r') as f:
        # Access the group
        group = f['df_with_missing']

        # Access the dataset
        dataset = group['table']
        data_array = np.array([element[1] for element in dataset[:]])
        if verbose:
            print(f"Dataset shape: {data_array.shape}")

    # Open the .h5 file for the label
    with h5py.File(sample_label_file, 'r') as f:
        labels = f['labels']
        labels_array = labels[:].T
        if verbose:
            print(f"Labels shape: {labels_array.shape}")

    # Compare dimensions
    if data_array.shape[0] == labels_array.shape[0]:
        print("The number of samples in data and labels match.")
    else:
        print("Mismatch in the number of samples between data and labels.")


def test_gentleboost(
    n_samples=15000,
    n_features=150,
    n_informative=50,
    test_size=0.2,
    random_state=42,
    **model_params
):
    """
    Test GentleBoost classifier with timing and performance metrics.

    Args:
        n_samples (int): Number of samples in synthetic dataset
        n_features (int): Number of features in synthetic dataset
        n_informative (int): Number of informative features
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Random state for reproducibility
        **model_params: Parameters to pass to GentleBoost constructor

    Returns:
        dict: Dictionary containing timing and performance metrics
    """
    # Generate synthetic dataset
    start_time = time()
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=random_state
    )
    y = np.where(y == 0, -1, 1)
    data_gen_time = time() - start_time

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialize and train model
    from gentleboost import GentleBoost
    model = GentleBoost(**model_params)

    # Time the training
    train_start = time()
    model.fit(X_train, y_train)
    train_time = time() - train_start

    # Time the prediction
    predict_start = time()
    y_pred = model.predict(X_test)
    predict_time = time() - predict_start

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label=1),
        'recall': recall_score(y_test, y_pred, pos_label=1),
        'f1': f1_score(y_test, y_pred, pos_label=1),
        'data_generation_time': data_gen_time,
        'training_time': train_time,
        'prediction_time': predict_time,
        'total_time': data_gen_time + train_time + predict_time,
        # Actual number of estimators used before early stopping
        'n_estimators_used': len(model.models)
    }

    return metrics
