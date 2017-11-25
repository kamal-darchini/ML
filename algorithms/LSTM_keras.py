from tensorflow.python.keras._impl.keras.layers import LSTM, Dense
from tensorflow.python.keras._impl.keras.models import Sequential


class LSTM_keras:

    def fit(self, train, batch_size, nb_epoch, neurons):
        X, y = train[:, 0:-1], train[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        self.model = Sequential()
        self.model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        for i in range(nb_epoch):
            self.model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
            self.model.reset_states()

    def predict(self, batch_size, row):
        X = row[0:-1]
        X = X.reshape(1, 1, len(X))
        yhat = self.model.predict(X, batch_size=batch_size)
        return yhat[0, 0]