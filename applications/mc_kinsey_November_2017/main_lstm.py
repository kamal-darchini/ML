import datetime
import random

from pandas import concat, Series
from sklearn.decomposition import PCA
from pandas import DataFrame

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from algorithms.LSTM import LSTM
from applications.mc_kinsey_November_2017.read_data import read_data
from applications.mc_kinsey_November_2017.read_test import read_test_data


def split(features, labels, validation_ratio):

    number_features = features.shape[0]
    train_features = []
    train_labels = []
    valid_features = []
    valid_labels = []
    for i in range(number_features):
        if random.random() >= validation_ratio:
            train_features.append(features[i])
            train_labels.append(labels[i])
        else:
            valid_features.append(features[i])
            valid_labels.append(labels[i])

    return np.array(train_features), np.array(train_labels), np.array(valid_features), np.array(valid_labels)


def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


if __name__ == '__main__':

    data_1, data_2, data_3, data_4, moving_avg, std = read_data('/Users/kamaloddindarchinimaragheh/Desktop/mk_kinsey/mc_kinsey_data.txt')
    data_1 = [float(x['vehicles']) for x in data_1]
    data_2 = [float(x['vehicles']) for x in data_2]
    data_3 = [float(x['vehicles']) for x in data_3]
    data_4 = [float(x['vehicles']) for x in data_4]

    supervised = timeseries_to_supervised(data_1, 1)
    print(supervised.head())

    # transform to be stationary
    differenced = difference(data_1, 1)
    print(differenced.head())
    # invert transform
    inverted = list()
    for i in range(len(differenced)):
        value = inverse_difference(data_1, differenced[i], len(data_1) - i)
        inverted.append(value)
    inverted = Series(inverted)
    print(inverted.head())

    # tranform scale
    X = np.array(data_1).reshape(len(data_1), 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(X)
    scaled_X = scaler.transform(X)

    # inverse transform
    inverted_X = scaler.inverse_transform(scaled_X)
    inverted_series = Series(inverted_X[:, 0])
    print(inverted_series.head())

