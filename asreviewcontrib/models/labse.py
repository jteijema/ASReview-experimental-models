from sentence_transformers import SentenceTransformer
from asreview.models.feature_extraction.base import BaseFeatureExtraction


class LaBSE(BaseFeatureExtraction):
    name = "LaBSE"
    label = "LaBSE Transformer (max_seq_length: 256)"

    def transform(self, texts):
        print("Max sequence length:", self.model.max_seq_length)
        return self.model.encode(texts, show_progress_bar=True)
    
    @property
    def model(self):
        if not hasattr(self, "_model"):
            self._model =  SentenceTransformer(
                "sentence-transformers/LaBSE"
            )
        return self._model

