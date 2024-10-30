from src.classifier_trainer import ClassifierTrainer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from gentleboost import GentleBoost
import lightgbm as lgb
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PowerTransformer


if __name__ == "__main__":
    trainer = ClassifierTrainer()
    # Define your classifiers
    dummy_clf = DummyClassifier(strategy="most_frequent")
    gnb_clf = make_pipeline(PowerTransformer(), GaussianNB(var_smoothing=1e-8))
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    mlp_clf = MLPClassifier(max_iter=1000, random_state=42)
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
