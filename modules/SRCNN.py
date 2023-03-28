"""SRCNN Module"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

from constants import MODEL_SRCNN_WEIGHTS

class SRCNN: 

    def __init__(self):

        self.model = keras.Sequential()

        self.model.add(layers.Conv2D(filters=128, kernel_size=(9,9), kernel_initializer='glorot_uniform', activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
        self.model.add(layers.Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='glorot_uniform', activation='relu', padding='valid', use_bias=True))
        self.model.add(layers.Conv2D(filters=1, kernel_size=(5,5),
                                kernel_initializer='glorot_uniform',
                activation='linear', padding='valid', use_bias=True))

        adam = optimizers.Adam(learning_rate=0.0003)

        self.model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

        self.model.load_weights(MODEL_SRCNN_WEIGHTS)

    def predict(self, img, batch_size=1):
        return self.model.predict(img, batch_size)