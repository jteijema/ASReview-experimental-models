import numpy as np
from asreview.models.feature_extraction.base import BaseFeatureExtraction
from tqdm import tqdm
import gensim.downloader as api

class Word2VecModel(BaseFeatureExtraction):
    name = "word2vec"
    label = "Word2Vec (Google News 300)"

    def __init__(self):
        super().__init__()

    @property
    def model(self):
        if not hasattr(self, "_model"):
            print("Loading embedding file...")
            self._model = api.load("word2vec-google-news-300")
            print("Embedding File loaded.")
        return self._model

    def transform(self, texts):
        """Transform texts into embeddings."""
        embeddings = []
        for text in tqdm(texts, desc="Transforming texts"):
            embeddings.append(self._get_embeddings(text))
        return np.array(embeddings)

    def _get_embeddings(self, text):
        """Extract embeddings from a text."""
        words = text.split()
        words = [word for word in words if word in self.model.key_to_index]
        if words:
            return np.mean([self.model[word] for word in words], axis=0)
        else:
            return np.zeros(self.model.vector_size)
