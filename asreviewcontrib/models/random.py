import numpy as np
from asreview.models.classifiers.base import BaseTrainClassifier

class RandomClassifier(BaseTrainClassifier):
    """
    A random classifier that ignores input features and returns predictions randomly.

    This classifier serves as a simple baseline to compare against more sophisticated models.
    """

    name = "random"

    def __init__(self):
        super().__init__()
        self._model = self  # In this case, the model is self-referential since there's no underlying model.

    def fit(self, X, y):
        """Fit the model to the data. This model ignores the input data."""
        # No fitting process is needed as predictions are random
        pass

    def predict_proba(self, X):
        """Return random probabilities for each sample.

        Arguments
        ---------
        X: numpy.ndarray
            Feature matrix to predict. The shape of X is not used since predictions are random.

        Returns
        -------
        numpy.ndarray
            Random probabilities for two classes.
        """
        # Generate random probabilities for each sample
        random_probabilities = np.random.rand(X.shape[0], 2)
        # Ensure that the sum of probabilities for each sample is 1
        random_probabilities /= random_probabilities.sum(axis=1, keepdims=True)
        return random_probabilities