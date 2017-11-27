import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from algorithms.LSTM_keras import LSTM_keras
from utils.scale import scale, invert_scale
from utils.split import split
from applications.mc_kinsey_November_2017.read_data import read_data
from utils.timeseries_to_supervised import inverse_difference, timeseries_to_normalized_supervised

if __name__ == '__main__':

    data_1, data_2, data_3, data_4, moving_avg, std = read_data('/Users/kamaloddindarchinimaragheh/Desktop/mk_kinsey/mc_kinsey_data.txt')
    data_1 = [float(x['vehicles']) for x in data_1]
    data_2 = [float(x['vehicles']) for x in data_2]
    data_3 = [float(x['vehicles']) for x in data_3]
    data_4 = [float(x['vehicles']) for x in data_4]

    train_data, _, test_data, _ = split(np.array(data_1), None, validation_ratio=.3)

    # transform to be stationary and supervised learning
    train = timeseries_to_normalized_supervised(data=train_data, difference_interval=1, history=5).values
    test = timeseries_to_normalized_supervised(data=test_data, difference_interval=1, history=5).values

    scaler, train_scaled, test_scaled = scale(train, test)

    # train model
    lstm = LSTM_keras(n_epochs=10,
                      n_units=2,
                      batch_size=1)
    lstm.fit(train_scaled)

    # walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = lstm.predict(X.reshape(1, 1, len(X)), 1)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(test_data, yhat, len(test_scaled) + 1 - i)
        # store forecast
        predictions.append(yhat)
        expected = test_data[i]
        # print('Predicted=%f, Expected=%f' % (yhat, expected))

    # report performance
    rmse = np.sqrt(mean_squared_error(test_data[1:], predictions))
    print('Test RMSE: %.3f' % rmse)
    # line plot of observed vs predicted
    plt.plot(test_data, 'b')
    plt.plot(predictions, 'r')
    plt.show()