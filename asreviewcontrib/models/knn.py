from asreview.models.classifiers.base import BaseTrainClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNN(BaseTrainClassifier):
    """KNN classifier

    This classifier is based on the KNeighborsClassifier from scikit-learn.

    """

    name = "knn"
    label = "KNN"

    def __init__(self):
        super().__init__()
        self._is_fit = False
        disable_logging()
    @ property
    def model(self):
        if not hasattr(self, "_model"):
            self._model = KNeighborsClassifier()
        return self._model

    def fit(self, X, y):
        if X.shape[0] > 5:
            self._is_fit = True
            return self.model.fit(X, y)

    def predict_proba(self, X):
        if self._is_fit:
            return self.model.predict_proba(X)
        else:
            return np.zeros((X.shape[0], 2))