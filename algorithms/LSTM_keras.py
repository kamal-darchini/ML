"""
This LSTM gets a difference time series with last column as the current value (or prediction, or label)
"""

from tensorflow.python.keras._impl.keras.layers import Dense, LSTM
from tensorflow.python.keras._impl.keras.models import Sequential


class LSTM_keras:
    def __init__(self, n_epochs, n_units, batch_size):
        self.n_epochs = n_epochs
        self.n_units = n_units
        self.batch_size = batch_size

    def fit(self, train_data):
        X, y = train_data[:, 0:-1], train_data[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        self.model = Sequential()
        self.model.add(LSTM(units=self.n_units,
                            batch_input_shape=(self.batch_size, X.shape[1], X.shape[2]),
                            return_sequences=True,
                            stateful=True))
        self.model.add(LSTM(units=self.n_units,
                            batch_input_shape=(self.batch_size, X.shape[1], self.n_units),
                            return_sequences=True,
                            stateful=True))
        self.model.add(LSTM(units=self.n_units,
                            batch_input_shape=(self.batch_size, X.shape[1], self.n_units),
                            # return_sequences=True,
                            stateful=True))
        self.model.add(Dense(5))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(X, y, epochs=self.n_epochs, batch_size=self.batch_size, verbose=2, shuffle=False)
        # self.model.reset_states()

    def predict(self, data, batch_size):
        yhat = self.model.predict(data, batch_size=batch_size)
        return yhat[0, 0]