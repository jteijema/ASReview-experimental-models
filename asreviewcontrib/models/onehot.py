from sklearn.feature_extraction.text import CountVectorizer
from asreview.models.feature_extraction.base import BaseFeatureExtraction

class OneHot(BaseFeatureExtraction):
    name = "onehot"
    label = "OneHot"

    def __init__(self):
        super().__init__()

    @property
    def _model(self):
        if not hasattr(self, "CV"):
            self.CV = CountVectorizer(binary=True, lowercase=True, ngram_range=(1, 3), max_df=0.9, min_df=5)
        return self.CV

    def fit(self, texts):
        """Fit the model to the texts."""
        self._model.fit(texts)

    def transform(self, texts):
        """Transform texts into one-hot encoded features."""
        X = self._model.transform(texts)
        return X