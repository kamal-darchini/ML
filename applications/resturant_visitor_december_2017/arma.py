# from applications.resturant_visitor_december_2017.main import join_dicts
from applications.resturant_visitor_december_2017.read_data import read_data
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt


def run_arma(train_data, test_data=None, ar=5, ma=5):
    train_data = [float(x) for x in train_data]

    # fig = plt.figure(figsize=(12, 8))
    # ax1 = fig.add_subplot(211)
    # fig = sm.graphics.tsa.plot_acf(train_data, lags=20, ax=ax1)
    # ax2 = fig.add_subplot(212)
    # fig = sm.graphics.tsa.plot_pacf(train_data, lags=20, ax=ax2)
    # plt.show()

    model = sm.tsa.ARMA(train_data, (ar, ma)).fit(disp=False)

    try:
        len_prediction = len(test_data) - 1
        predictions = [max(0, int(round(x))) for x in model.forecast(len_prediction)[0]]
        # report performance
        msle = 0
        for i, pred in enumerate(predictions):
            msle += (np.log(pred + 1) - np.log(test_data[i] + 1)) ** 2
        rmsle = np.sqrt(msle / len(predictions))

        print('Test RMSLE: %.3f' % rmsle)

    except TypeError:
        len_prediction = 40
        predictions = [max(0, int(round(x))) for x in model.forecast(len_prediction)[0]]

    # plt.plot(test_data)
    # plt.plot(predictions, 'r')
    # plt.show()

    return predictions

# if __name__ == '__main__':
#
#     relation_path = '/Users/kamaloddindarchinimaragheh/git/ML/data/time_series/resturant_visitors/store_id_relation.csv'
#     data_air = read_data('/Users/kamaloddindarchinimaragheh/git/ML/data/time_series/resturant_visitors/air_visit_data.csv')
#     data_hpg = read_data('/Users/kamaloddindarchinimaragheh/git/ML/data/time_series/resturant_visitors/hpg_reserve.csv', relation_path)
#     data = join_dicts(data_air, data_hpg)
#     # data_air = read_data_air_only('/Users/kamaloddindarchinimaragheh/git/ML/data/time_series/resturant_visitors/air_visit_data.csv')
#
#     data = data_air
#
#     # only air
#     # data_0 = np.array(data[list(data.keys())[0]].visitors)
#
#     # air and hpg
#     resturant = list(data.keys())[0]
#     data = np.array(list(data[resturant].values()))
#
#     # train_data, _, test_data, _ = split(np.array(data_1), None, validation_ratio=.3)
#     train_data = data[:-39]
#     test_data = data[-39:]
#     run_arma(data, test_data)
