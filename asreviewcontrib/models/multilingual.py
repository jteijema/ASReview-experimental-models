from sentence_transformers import SentenceTransformer
from asreview.models.feature_extraction.base import BaseFeatureExtraction


class MultilingualSentenceTransformer(BaseFeatureExtraction):
    name = "multilingual"
    label = "Multilingual Sentence Transformer (max_seq_length: 128)"

    def transform(self, texts):
        print("Max sequence length:", self.model.max_seq_length)
        print(
            "Encoding texts using the multilingual SentenceTransformer model, this may take a while..."
        )
        return self.model.encode(texts, show_progress_bar=True)
    
    @property
    def model(self):
        if not hasattr(self, "_model"):
            self._model =  SentenceTransformer(
                "sentence-transformers/distiluse-base-multilingual-cased-v2"
            )
        return self._model
