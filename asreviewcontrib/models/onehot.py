from sklearn.feature_extraction.text import CountVectorizer
from asreview.models.feature_extraction.base import BaseFeatureExtraction

class OneHot(BaseFeatureExtraction):
    name = "OneHot"
    label = "OneHot"

    def __init__(self):
        super().__init__()
        self._model = CountVectorizer(binary=True)

    def fit(self, texts):
        """Fit the model to the texts."""
        self._model.fit(texts)

    def transform(self, texts):
        """Transform texts into one-hot encoded features."""
        X = self._model.transform(texts)
        return X