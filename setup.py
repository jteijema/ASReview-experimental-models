from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name='JTeijema-asreview-models',
    version='1.0',
    description='JTeijema-asreview-models',
    url='https://github.com/JTeijema/JTeijema-asreview-models',
    author='JTeijema',
    author_email='j.j.teijema@uu.nl',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='systematic review',
    packages=find_namespace_packages(include=['asreviewcontrib.*']),
    python_requires='~=3.6',
    install_requires=[
        'asreview>=1.0',
        'xgboost',
        'sentence_transformers',
        'gensim',
        'scikeras',
        'fasttext',
        'spacy',
        'scikit-learn'
    ],
    entry_points={
        'asreview.models.classifiers': [
            'xgboost = asreviewcontrib.models:XGBoost',
            'power_cnn = asreviewcontrib.models:Power_CNN',
            'dynamic_nn = asreviewcontrib.models:DynamicNNClassifier',
            'adaboost = asreviewcontrib.models:AdaBoost',
            'knn = asreviewcontrib.models:KNN',
        ],
        'asreview.models.feature_extraction': [
            'ft_sbert = asreviewcontrib.models:FullTextSBERTModel',
            'wide_doc2vec = asreviewcontrib.models:wide_doc2vec',
            "multilingual = asreviewcontrib.models:MultilingualSentenceTransformer",
            "labse = asreviewcontrib.models:LaBSE",
            "fasttext = asreviewcontrib.models:FastTextFeatureExtractor",
            "onehot = asreviewcontrib.models:OneHot",
            "spacy = asreviewcontrib.models:SpacyEmbeddingExtractor",
            "word2vec = asreviewcontrib.models:Word2VecModel",
        ],
        'asreview.models.balance': [
            # define balance strategy algorithms
        ],
        'asreview.models.query': [
            # define query strategy algorithms
        ]
    },
    project_urls={
        'Bug Reports': 'https://github.com/asreview/asreview/issues',
        'Source': 'https://github.com/asreview/asreview/',
    },
)
