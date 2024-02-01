from sentence_transformers import SentenceTransformer
from asreview.models.feature_extraction.base import BaseFeatureExtraction


class MultilingualSentenceTransformer(BaseFeatureExtraction):
    """
    Multilingual Sentence Transformer feature extraction technique using
    the 'sentence-transformers/distiluse-base-multilingual-cased-v2' model.

    This class inherits from the BaseFeatureExtraction class provided by the
    ASReview package and implements the transform method for encoding texts
    using the multilingual SentenceTransformer model.
    """

    name = "multilingual"
    label = "Multilingual Sentence Transformer"

    def transform(self, texts):
        """
        Encode the given texts using the SentenceTransformer model.

        Args:
        texts (List[str]): A list of text strings to be encoded.

        Returns:
        numpy.ndarray: A 2D array containing the encoded text embeddings.
        """

        model = SentenceTransformer(
            "sentence-transformers/distiluse-base-multilingual-cased-v2"
        )
        print("Max sequence length:", self.model.max_seq_length)
        print(
            "Encoding texts using the multilingual SentenceTransformer model, this may take a while..."
        )
        return model.encode(texts, show_progress_bar=True)
