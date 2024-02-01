import spacy
import numpy as np
from asreview.models.feature_extraction.base import BaseFeatureExtraction

class SpacyEmbeddingExtractor(BaseFeatureExtraction):
    name = "spacy"
    label = "SpaCy (en_core_web_lg)"

    def __init__(self):
        super().__init__()
        _model_name = "en_core_web_lg"
        try:
            print("Loading spaCy model...")
            self.nlp = spacy.load(_model_name)
        except OSError:
            print(f"Downloading language model for the spaCy POS tagger: {_model_name}...")
            from spacy.cli import download
            download(_model_name)
            self.nlp = spacy.load(_model_name)

    def _get_embeddings(self, doc):
        """Extract embeddings from a spacy document."""
        return doc.vector

    def transform(self, texts):
        """
        Transform texts into embeddings.

        The embeddings for each text are averaged across all tokens in the text.
        """
        embeddings = [self._get_embeddings(self.nlp(text)) for text in texts]
        return np.array(embeddings)
