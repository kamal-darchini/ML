"""
This LSTM gets a difference time series with last column as the current value (or prediction, or label)
"""

from tensorflow.python.keras._impl.keras.layers import Dense, LSTM
from tensorflow.python.keras._impl.keras.models import Sequential
import tensorflow as tf


class LSTM_keras:
    def __init__(self, n_epochs, n_units, n_lstm_layers, batch_size, input_length):
        self.n_epochs = n_epochs
        self.n_units = n_units
        self.batch_size = batch_size
        self.input_length = input_length

        self.X = tf.placeholder(tf.float32, [None, 1, self.input_length])
        self.Y = tf.placeholder(tf.float32, [None, 1, 1])

        self.model = Sequential()
        for i in range(n_lstm_layers-1):
            self.model.add(LSTM(units=self.n_units,
                                batch_input_shape=(self.batch_size, self.X.shape[1], self.X.shape[2]),
                                return_sequences=True,
                                stateful=True))

        if n_lstm_layers > 1:
            self.model.add(LSTM(units=self.n_units,
                                batch_input_shape=(self.batch_size, self.X.shape[1], self.n_units),
                                stateful=True))
        else:
            self.model.add(LSTM(units=self.n_units,
                                batch_input_shape=(self.batch_size, self.X.shape[1], self.X.shape[2]),
                                stateful=True))

        self.model.add(Dense(5))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit(self, train_data, train_label):
        X = train_data
        y = train_label
        X = X.reshape(X.shape[0], 1, X.shape[1])
        self.model.fit(X, y, epochs=self.n_epochs, batch_size=self.batch_size, verbose=2, shuffle=True)
        # self.model.reset_states()

    def predict(self, data, batch_size):
        yhat = self.model.predict(data, batch_size=batch_size)
        return yhat[0, 0]