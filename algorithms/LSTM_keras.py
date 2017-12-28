"""
This LSTM gets a difference time series with last column as the current value (or prediction, or label)
"""

from tensorflow.python.keras._impl.keras.layers import Dense, LSTM, regularizers
from tensorflow.python.keras._impl.keras.models import Sequential
import tensorflow as tf


class LSTM_keras:
    def __init__(self,
                 n_epochs,
                 n_units,
                 batch_size,
                 input_length,
                 kernel_l1=0.1,
                 bias_l1=0.1,
                 recurrent_l1=0.1,
                 dropout=0.):
        self.n_epochs = n_epochs
        self.n_units = n_units
        self.batch_size = batch_size
        self.input_length = input_length
        self.kernel_l1 = kernel_l1
        self.bias_l1 = bias_l1
        self.recurrent_l1 = recurrent_l1
        self.dropout = dropout
        n_lstm_layers = len(self.n_units)

        self.X = tf.placeholder(tf.float32, [None, 1, self.input_length])
        self.Y = tf.placeholder(tf.float32, [None, 1, 1])

        self.model = Sequential()
        for i in range(n_lstm_layers-1):
            self.model.add(LSTM(units=self.n_units[i],
                                batch_input_shape=(self.batch_size, self.X.shape[1], self.X.shape[2]),
                                kernel_regularizer=regularizers.l1(self.kernel_l1),
                                bias_regularizer=regularizers.l1(self.bias_l1),
                                recurrent_regularizer=regularizers.l1(self.recurrent_l1),
                                return_sequences=True,
                                dropout=self.dropout,
                                stateful=True))

        if n_lstm_layers > 1:
            self.model.add(LSTM(units=self.n_units[-1],
                                batch_input_shape=(self.batch_size, self.X.shape[1], self.n_units),
                                kernel_regularizer=regularizers.l1(self.kernel_l1),
                                bias_regularizer=regularizers.l1(self.bias_l1),
                                recurrent_regularizer=regularizers.l1(self.recurrent_l1),
                                dropout=self.dropout,
                                stateful=True))
        else:
            self.model.add(LSTM(units=self.n_units[-1],
                                batch_input_shape=(self.batch_size, self.X.shape[1], self.X.shape[2]),
                                kernel_regularizer=regularizers.l1(self.kernel_l1),
                                bias_regularizer=regularizers.l1(self.bias_l1),
                                recurrent_regularizer=regularizers.l1(self.recurrent_l1),
                                dropout=self.dropout,
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