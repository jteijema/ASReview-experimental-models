from asreview.models.classifiers.base import BaseTrainClassifier
from sklearn.neighbors import KNeighborsClassifier


class KNN(BaseTrainClassifier):
    """KNN classifier

    This classifier is based on the KNeighborsClassifier from scikit-learn.

    """

    name = "knn"
    label = "KNN"

    def __init__(self):
        super().__init__()
        self._model = KNeighborsClassifier()