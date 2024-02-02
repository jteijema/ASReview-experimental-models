from sentence_transformers import SentenceTransformer
from asreview.models.feature_extraction.base import BaseFeatureExtraction
import numpy as np
from tqdm import tqdm

class FullTextSBERTModel(BaseFeatureExtraction):
    """Full Text Sentence BERT model."""

    name = "ft_sbert"
    label = "Full Text Sentence BERT (max_seq_length: INF)"

    def __init__(
        self,
        *args,
        transformer_model="all-mpnet-base-v2",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.transformer_model = transformer_model
        self.model = SentenceTransformer(transformer_model)
        self.tokenizer = self.model.tokenizer
        print("Max sequence length:", self.model.max_seq_length)

    def split_text(self, text, token_limit):
        words = text.split()
        for i in range(0, len(words), token_limit):
            yield ' '.join(words[i:i + token_limit])

    def transform(self, texts):
        print("Encoding texts with sbert, this may take a while...")

        encoded_texts = []

        for text in tqdm(texts):
            segments = list(self.split_text(text, self.model.max_seq_length))
            segments = [segment for segment in segments if segment != ""]

            segment_embeddings = self.model.encode(segments, show_progress_bar=False)

            if len(segment_embeddings) == 0:
                encoded_texts.append(np.zeros(self.model.get_sentence_embedding_dimension()))
                continue

            encoded_texts.append(np.mean(segment_embeddings, axis=0))

            if encoded_texts[-1].shape != (768,):
                print("Encoded text shape:", encoded_texts[-1].shape)
                print("Encoded text:", encoded_texts[-1])
                print("Text:", text)
                print("Segments:", segments)

        return np.array(encoded_texts)