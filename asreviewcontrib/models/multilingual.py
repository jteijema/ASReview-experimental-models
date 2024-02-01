from sentence_transformers import SentenceTransformer
from asreview.models.feature_extraction.base import BaseFeatureExtraction


class MultilingualSentenceTransformer(BaseFeatureExtraction):
    name = "multilingual"
    label = "Multilingual Sentence Transformer (max_seq_length: 128)"

    model = SentenceTransformer(
        "sentence-transformers/distiluse-base-multilingual-cased-v2"
    )
    print("Max sequence length:", model.max_seq_length)

    def transform(self, texts):
        print(
            "Encoding texts using the multilingual SentenceTransformer model, this may take a while..."
        )
        return self.model.encode(texts, show_progress_bar=True)
