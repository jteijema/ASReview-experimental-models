import spacy
import numpy as np
from asreview.models.feature_extraction.base import BaseFeatureExtraction
from tqdm import tqdm

class SpacyEmbeddingExtractor(BaseFeatureExtraction):
    name = "spacy"
    label = "SpaCy (en_core_web_lg)"

    def __init__(self):
        super().__init__()
        self._model_name = "en_core_web_lg"

    @property
    def nlp(self):
        if not hasattr(self, "_nlp"):
            try:
                print("Loading spaCy model...")
                self._nlp = spacy.load(self._model_name)
            except OSError:
                print(f"Downloading language model for the spaCy POS tagger: {self._model_name}...")
                from spacy.cli import download
                download(self._model_name)
                self._nlp = spacy.load(self._model_name)
            print("Model loaded.")
        return self._nlp

    def _get_embeddings(self, doc):
        """Extract embeddings from a spacy document."""
        return doc.vector

    def transform(self, texts):
        """
        Transform texts into embeddings.

        The embeddings for each text are averaged across all tokens in the text.
        """
        embeddings = []
        for text in tqdm(texts):
            embeddings.append(self._get_embeddings(self.nlp(text)))
        embeddings = np.array(embeddings)
        return np.array(embeddings)
