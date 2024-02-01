import os
from urllib.request import urlretrieve
from pathlib import Path
import numpy as np
from gensim.models import KeyedVectors
from asreview.models.feature_extraction.base import BaseFeatureExtraction
from asreview.utils import get_data_home

class FastTextFeatureExtractor(BaseFeatureExtraction):
    name = "fasttext"
    label = "FastText (wiki-news-300d-1M-subword.vec)"

    EMBEDDING_EN = {
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip",
        "name": "wiki-news-300d-1M-subword.vec",
    }

    @property
    def model(self):
        if not hasattr(self, "_model"):
            self._model = self.get_embedding_matrix()
            print("Embedding File loaded.")
        return self._model

    def get_embedding_matrix(self):
        data_home = Path(get_data_home())
        embedding_fp = data_home / self.EMBEDDING_EN["name"]
        print("\nLooking for embedding file in: ", embedding_fp)
        if not embedding_fp.exists():
            print("Embedding not found: Starting the download of the FastText embedding file.")
            self.download_embedding(data_home)
        else:
            print("Embedding file found.")
        print("Loading 2.2GB embedding file to memory.")
        return KeyedVectors.load_word2vec_format(embedding_fp, binary=False)

    def download_embedding(self, data_home):
        url = FastTextFeatureExtractor.EMBEDDING_EN["url"]
        file_name = FastTextFeatureExtractor.EMBEDDING_EN["name"]
        zip_path = data_home / (file_name + '.zip')
        print("Downloaded embedding file to: ", zip_path)
        urlretrieve(url, zip_path)
        print("Download complete.")

        print("Unzipping embedding file...")
        # Unzipping the file
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_home)
        print("Unzipping complete.")

        print("Removing zip file...")
        # Remove the zip file to save space
        os.remove(zip_path)
        print("Zip file removed.")

    def transform(self, texts):
        print("Encoding texts using FastText model...")
        transformed_texts = []
        for text in texts:
            transformed_texts.append(self.text_to_vector(text))
        print("Encoding complete.")
        self.clear_model()
        print("Unloading model.")
        return np.array(transformed_texts)

    def text_to_vector(self, text):
        words = text.split()
        word_vectors = [self.model[word] for word in words if word in self.model]
        if not word_vectors:
            return np.zeros(self.model.vector_size)
        return np.mean(word_vectors, axis=0)

    def clear_model(self):
        if hasattr(self, "_model"):
            del self._model