from sentence_transformers import SentenceTransformer
from asreview.models.feature_extraction.base import BaseFeatureExtraction


class LaBSE(BaseFeatureExtraction):
    name = "LaBSE"
    label = "LaBSE Transformer (max_seq_length: 256)"

    model = SentenceTransformer(
        "sentence-transformers/LaBSE"
    )
    print("Max sequence length:", model.max_seq_length)

    def transform(self, texts):
        return self.model.encode(texts, show_progress_bar=True)
