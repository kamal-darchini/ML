import numpy as np
from numpy.linalg import LinAlgError
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from algorithms.LSTM_keras import LSTM_keras
from applications.resturant_visitor_december_2017.arma import run_arma
from applications.resturant_visitor_december_2017.read_data import read_data
from applications.resturant_visitor_december_2017.read_data_air_only import read_data_air_only
from utils.scale import scale, invert_scale
from utils.split import split
from utils.timeseries_to_supervised import inverse_difference, timeseries_to_normalized_supervised


def join_dicts(air_dict, hpg_dict):
    for k in hpg_dict.keys():
        for date in hpg_dict[k].keys():
            try:
                air_dict[k][date] += hpg_dict[k][date]
            except KeyError:
                try:
                    air_dict[k].update({date: hpg_dict[k][date]})
                except KeyError:
                    air_dict.update({k: {date: hpg_dict[k][date]}})

    return air_dict


def run_one_resturant(train_data, test_data):

    print('length of train data is: ', len(train_data))
    print('length of test data is: ', len(test_data))

    # transform to be stationary and supervised learning
    history = 7
    train = timeseries_to_normalized_supervised(data=train_data, difference_interval=1, history=history).values
    test = timeseries_to_normalized_supervised(data=test_data, difference_interval=1, history=history).values

    scaler, train_scaled, test_scaled = scale(train, test)

    # train model
    lstm = LSTM_keras(n_epochs=20,
                      n_units=[7, 7],
                      batch_size=1,
                      input_length=train_scaled.shape[1] - 1,
                      kernel_l1=0.75,
                      bias_l1=0.75,
                      recurrent_l1=0.75)

    lstm.fit(train_scaled[:, :-1], train_scaled[:, -1])

    # walk-forward validation on the test data
    predictions = list()
    for i in range(len(test)):
        # make one-step forecast
        # X, y = test[i, 0:-1], test[i, -1]
        X = train_data[-1 * history:]
        y = test[i]
        if predictions:
            if len(predictions) >= len(X):
                X = predictions[-len(X):]
            else:
                X = [test[0][0]] * (len(X) - len(predictions)) + predictions
            try:
                tmp = list(X)
                tmp.append(predictions[-1])
                tmp = scaler.transform(tmp)
                X = tmp[:-1]
                y = tmp[-1]
                yhat = lstm.predict(X.reshape(1, 1, len(X)), 1)
            except (TypeError, AttributeError):
                X = np.array([X])
                tmp = list(X)
                tmp.append(predictions[-1])
                tmp = scaler.transform(tmp)
                X = tmp[:-1]
                y = tmp[-1]
                yhat = lstm.predict(X.reshape(1, 1, len(X)), 1)
        else:
            tmp = list(X)
            tmp.append(train_data[-1])
            tmp = scaler.transform(tmp)
            X = tmp[:-1]
            y = tmp[-1]
            yhat = lstm.predict(X.reshape(1, 1, len(X)), 1)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        ##### yhat = inverse_difference(test_data, yhat, len(test_scaled) + 1 - i)
        if predictions:
            yhat = yhat + predictions[-1]
        else:
            yhat = yhat + train_data[-1]
        # store forecast
        predictions.append(max(yhat, 0))
        expected = test_data[i]
        # print('Predicted=%f, Expected=%f' % (yhat, expected))

    # report performance
    msle = 0
    for i, pred in enumerate(predictions):
        msle += (np.log(pred + 1) - np.log(test_data[i + 1] + 1)) ** 2
    rmsle = np.sqrt(msle / len(predictions))

    print('Test RMSLE: %.3f' % rmsle)
    # line plot of observed vs predicted
    plt.plot(test_data, 'b')
    plt.plot(predictions, 'r')
    plt.show()

    return predictions


def run_one_resturant_output(train_data):
    print('length of train data is: ', len(train_data))

    # transform to be stationary and supervised learning
    history = 7
    train = timeseries_to_normalized_supervised(data=train_data, difference_interval=1, history=history).values

    scaler, train_scaled, test_scaled = scale(train)

    # train model
    lstm = LSTM_keras(n_epochs=10,
                      n_units=[7],
                      batch_size=1,
                      input_length=train_scaled.shape[1] - 1,
                      kernel_l1=0.8,
                      bias_l1=0.8,
                      recurrent_l1=0.8,
                      dropout=.6)

    lstm.fit(train_scaled[:, :-1], train_scaled[:, -1])

    # walk-forward validation on the test data
    predictions = list()
    for i in range(45):
        # make one-step forecast
        X = train_data[-1 * history:]
        if predictions:
            if len(predictions) >= len(X):
                X = predictions[-len(X):]
            else:
                X = list(train_data[-1 * (len(X) - len(predictions)):]) + predictions
            try:
                tmp = list(X)
                tmp.append(0)
                tmp = scaler.transform(tmp)
                X = tmp[:-1]
                y = tmp[-1]
                yhat = lstm.predict(X.reshape(1, 1, len(X)), 1)
            except (TypeError, AttributeError):
                X = np.array([X])
                tmp = list(X)
                tmp.append(0)
                tmp = scaler.transform(tmp)
                X = tmp[:-1]
                y = tmp[-1]
                yhat = lstm.predict(X.reshape(1, 1, len(X)), 1)
        else:
            tmp = list(X)
            tmp.append(0)
            tmp = scaler.transform(tmp)
            X = tmp[:-1]
            y = tmp[-1]
            yhat = lstm.predict(X.reshape(1, 1, len(X)), 1)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        if predictions:
            yhat = yhat + predictions[-1]
        else:
            yhat = yhat + train_data[-1]
        # store forecast
        predictions.append(max(yhat, 0))
        # print('Predicted=%f, Expected=%f' % (yhat, expected))

    # plot_data = [0] * len(train_data) + list(predictions)
    # plt.plot(train_data)
    # plt.plot(plot_data, 'r')
    # plt.show()

    return predictions


def fill_sample_submission(path, data):

    last_resturant_id = None
    with open(path, 'r') as output:
        for line_num, line in enumerate(output.readlines()):
            if line_num == 0:
                with open('./output_arima.csv', 'w') as file:
                    file.write(line)
            else:
                row_0 = line.split(',')[0]
                id_column = line.split(',')[0].split('_')
                resturant_id = id_column[0] + '_' + id_column[1]
                if resturant_id != last_resturant_id:
                    print('====================')
                    print('started filling csv file for ', resturant_id)
                    print('====================')
                    input_data = np.array(list(data[resturant_id].values()))
                    # predictions = run_one_resturant_output(input_data)

                    fitted = False
                    for i in range(-4, 0):
                        for j in range(-7, 0):
                            try:
                                predictions = run_arma(input_data, -1 * i, -1 * j)
                            except (ValueError, LinAlgError):
                                predictions = []
                            if len(predictions) > 0:
                                fitted = True
                                break
                        if fitted:
                            break

                    if len(predictions) == 0:
                        print('++++++++++++++++++++++++')
                        print('could not fit anything!!!')
                        print('++++++++++++++++++++++++')
                        predictions = [sum(input_data) / len(input_data)] * 40
                    new_line = row_0 + ',' + str(predictions[0]) + '\n'
                    with open('./output_arima.csv', 'a') as file:
                        file.write(new_line)
                    last_resturant_id = resturant_id
                    count = 0
                else:
                    count += 1
                    new_line = row_0 + ',' + str(predictions[count]) + '\n'
                    with open('./output_arima.csv', 'a') as file:
                        file.write(new_line)


if __name__ == '__main__':

    relation_path = '/Users/kamaloddindarchinimaragheh/git/ML/data/time_series/resturant_visitors/store_id_relation.csv'
    data_air = read_data('/Users/kamaloddindarchinimaragheh/git/ML/data/time_series/resturant_visitors/air_visit_data.csv')
    data_hpg = read_data('/Users/kamaloddindarchinimaragheh/git/ML/data/time_series/resturant_visitors/hpg_reserve.csv', relation_path)
    data = join_dicts(data_air, data_hpg)
    # data_air = read_data_air_only('/Users/kamaloddindarchinimaragheh/git/ML/data/time_series/resturant_visitors/air_visit_data.csv')

    # data = data_air
    #
    # # only air
    # # data_0 = np.array(data[list(data.keys())[0]].visitors)
    #
    # # air and hpg
    # resturant = list(data.keys())[0]
    # data_0 = np.array(list(data[resturant].values()))
    #
    # # train_data, _, test_data, _ = split(np.array(data_1), None, validation_ratio=.3)
    # data = data_0
    # train_data = data[:-39]
    # test_data = data[-39:]
    # run_one_resturant(data, test_data)

    path = '/Users/kamaloddindarchinimaragheh/git/ML/applications/resturant_visitor_december_2017/output/sample_submission.csv'
    fill_sample_submission(path, data)