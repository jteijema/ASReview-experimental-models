import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from math import log10, ceil
from asreview.models.classifiers.base import BaseTrainClassifier

def create_model(input_dim, num_layers):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    
    # Add additional layers based on the number of rows in the dataset
    for _ in range(num_layers - 1):
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

class DynamicNNClassifier(BaseTrainClassifier):
    name = "dynamic_nn"
    label = "Dynamic Neural Network"

    def __init__(self, verbose=0, patience=5, min_delta=0.01):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self._model = None

    def fit(self, X, y):
        max_features = 1000

        if X.shape[1] > max_features:
            raise ValueError(f"Feature size too large: {X.shape[1]} features. Maximum allowed is {max_features}.")

        if self.verbose == 1:
            print("\nNumber of features:", X.shape[1])
        # Determine the number of layers based on the number of rows
        num_layers = ceil(log10(max(10, X.shape[0])))

        self._model = KerasClassifier(model=create_model, 
                                      model__input_dim=X.shape[1], 
                                      model__num_layers=num_layers,
                                      verbose=self.verbose)

        callback = EarlyStopping(monitor='loss', patience=self.patience, 
                                 min_delta=self.min_delta, 
                                 restore_best_weights=True)

        if self.verbose == 1:
            print(f"\nFitting New Iteration with {num_layers} layers:\n")
        self._model.fit(X, y, epochs=100, batch_size=32, 
                        shuffle=True, callbacks=[callback])

    def predict_proba(self, X):
        return self._model.predict_proba(X)
