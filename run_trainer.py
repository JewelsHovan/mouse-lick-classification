from src.classifier_trainer import ClassifierTrainer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # load in the data
    data = pd.read_csv("Data/GentleBoost/train_data.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trainer = ClassifierTrainer(results_csv_path="classifier_results.csv")
    # Define your classifiers
    dummy_clf = DummyClassifier(strategy="most_frequent")
    gnb_clf = make_pipeline(PowerTransformer(), GaussianNB(var_smoothing=1e-8))
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    lr_clf = LogisticRegression(max_iter=2000, random_state=42)
    mlp_clf = MLPClassifier(max_iter=2000, random_state=42, 
                           early_stopping=True, validation_fraction=0.1)
    svc_clf = SVC(kernel='rbf', random_state=42)
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    xgb_clf = XGBClassifier(n_estimators=250, max_depth=4, random_state=42)

    # Add classifiers to the trainer
    trainer.add_classifier(dummy_clf)
    trainer.add_classifier(gnb_clf)
    trainer.add_classifier(knn_clf)
    trainer.add_classifier(lr_clf)
    trainer.add_classifier(mlp_clf)
    trainer.add_classifier(svc_clf)
    trainer.add_classifier(rf_clf)
    trainer.add_classifier(xgb_clf)

    # train the classifiers
    trainer.train_all(X_train, y_train, X_test, y_test)
    trainer.save_results()
