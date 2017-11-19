import datetime
import random
from sklearn.decomposition import PCA
from MLP import MLP
from holt_winters import linear
from read_data import read_data
from read_test import read_test_data

import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt


def extract_features_and_label_test(data):
    features = []
    day_feature_dict = {'Monday': 0,
                        'Tuesday': 1,
                        'Wednesday': 2,
                        'Thursday': 3,
                        'Friday': 4,
                        'Saturday': 5,
                        'Sunday': 6}
    months = {'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December'}
    month_feature_dict = {k: v for v, k in enumerate(months)}

    visualize_label = []
    for datum in data:
        year, month, day = [int(x) for x in datum['date'].split('-')]

        month_of_year = datetime.date(year, month, day).strftime("%B")
        month_of_year_feature = [0] * 12
        month_of_year_feature[month_feature_dict[month_of_year]] = 1

        day_of_week = datetime.date(year, month, day).strftime("%A")
        day_of_week_feature = [0] * 7
        day_of_week_feature[day_feature_dict[day_of_week]] = 1
        # if day_of_week != 'Friday':
        #     continue

        hour_feature = [0] * 24
        hour, minute, second = datum['time'].split(':')
        hour_feature[int(hour) - 1] = 1

        junction_feature = [0] * 4
        junction_feature[int(datum['junction']) - 1] = 1

        year_feature = [0, 0, 0, 0]
        year_feature[year - 2015] = 1

        extra_feature = [np.sin(2 * np.pi / 7) * day_feature_dict[day_of_week]]

        features.append(
            # year_feature +
            month_of_year_feature +
            day_of_week_feature +
            hour_feature +
            junction_feature # +
            # extra_feature
        )

    # plt.plot(visualize_label)
    # plt.show()

    return np.array(features)


def extract_features_and_label(data):
    labels = []
    features = []
    day_feature_dict = {'Monday': 0,
                        'Tuesday': 1,
                        'Wednesday': 2,
                        'Thursday': 3,
                        'Friday': 4,
                        'Saturday': 5,
                        'Sunday': 6}
    months = {'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December'}
    month_feature_dict = {k: v for v, k in enumerate(months)}

    visualize_label = []
    for datum in data:
        year, month, day = [int(x) for x in datum['date'].split('-')]

        month_of_year = datetime.date(year, month, day).strftime("%B")
        month_of_year_feature = [0] * 12
        month_of_year_feature[month_feature_dict[month_of_year]] = 1

        day_of_week = datetime.date(year, month, day).strftime("%A")
        day_of_week_feature = [0] * 7
        day_of_week_feature[day_feature_dict[day_of_week]] = 1
        # if day_of_week != 'Friday':
        #     continue

        hour_feature = [0] * 24
        hour, minute, second = datum['time'].split(':')
        hour_feature[int(hour) - 1] = 1

        junction_feature = [0] * 4
        junction_feature[int(datum['junction']) - 1] = 1

        year_feature = [0, 0, 0, 0]
        year_feature[year - 2015] = 1

        extra_feature = [np.sin(2 * np.pi / 7) * day_feature_dict[day_of_week]]

        features.append(
            # year_feature +
            month_of_year_feature +
            day_of_week_feature +
            hour_feature +
            junction_feature # +
            # extra_feature
        )

        label = [0] * 180
        label[int(datum['vehicles']) - 1] = 1
        labels.append(label)

    # plt.plot(visualize_label)
    # plt.show()

    return np.array(features), np.array(labels)


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

if __name__ == '__main__':

    data_1, data_2, data_3, data_4, moving_avg, std, moving_avg_1 = read_data('/Users/kamaloddindarchinimaragheh/Desktop/mk_kinsey/mc_kinsey_data.txt')

    features_1, labels_1 = extract_features_and_label(data_1)
    features_2, labels_2 = extract_features_and_label(data_2)
    features_3, labels_3 = extract_features_and_label(data_3)
    features_4, labels_4 = extract_features_and_label(data_4)
    print('got features!')

    data_1, data_2, data_3, data_4 = read_test_data('/Users/kamaloddindarchinimaragheh/Desktop/mk_kinsey/mc_kinsey_data_test.txt')

    test_features_1 = extract_features_and_label_test(data_1)
    test_ids_1 = [x['id'] for x in data_1]
    test_features_2 = extract_features_and_label_test(data_2)
    test_ids_2 = [x['id'] for x in data_2]
    test_features_3 = extract_features_and_label_test(data_3)
    test_ids_3 = [x['id'] for x in data_3]
    test_features_4 = extract_features_and_label_test(data_4)
    test_ids_4 = [x['id'] for x in data_4]

    # pca = PCA(n_components=2)
    # features = pca.fit_transform(features)
    # np.random.seed(1234)
    # np.random.shuffle(features)
    # np.random.seed(1234)
    # np.random.shuffle(labels)

    idxs = list(range(len(features_1)))
    np.random.shuffle(idxs)
    new_features = np.array([features_1[i] for i in idxs])
    new_labels = np.array([labels_1[i] for i in idxs])
    features = new_features
    labels = new_labels
    # predictions, _, _, rmse = linear([int(x['vehicles']) for x in data], 10)
    train_features, train_labels, valid_features, valid_labels = split(features, labels, 0.3)
    predictor_1 = MLP()
    predictor_1.n_classes = len(labels[0])
    predictor_1.n_input = features.shape[1]
    test_predictions_1 = predictor_1.fit(train_features, train_labels, valid_features, valid_labels, test_features_1)
    with open('/Users/kamaloddindarchinimaragheh/Desktop/mk_kinsey/result.txt', 'w') as file:
        file.write('ID, Vehicle\n')
        for id, veh in zip(test_ids_1, test_predictions_1):
            file.write(str(id) + ',' + str(veh) + '\n')

    #############
    ############
    idxs = list(range(len(features_2)))
    np.random.shuffle(idxs)
    new_features = np.array([features_2[i] for i in idxs])
    new_labels = np.array([labels_2[i] for i in idxs])
    features = new_features
    labels = new_labels
    # predictions, _, _, rmse = linear([int(x['vehicles']) for x in data], 10)
    train_features, train_labels, valid_features, valid_labels = split(features, labels, 0.3)
    predictor_2 = MLP()
    predictor_2.n_classes = len(labels[0])
    predictor_2.n_input = features.shape[1]
    test_predictions_2 = predictor_2.fit(train_features, train_labels, valid_features, valid_labels, test_features_2)
    with open('/Users/kamaloddindarchinimaragheh/Desktop/mk_kinsey/result.txt', 'a') as file:
        for id, veh in zip(test_ids_2, test_predictions_2):
            file.write(str(id) + ',' + str(veh) + '\n')

    #############
    ############
    idxs = list(range(len(features_3)))
    np.random.shuffle(idxs)
    new_features = np.array([features_3[i] for i in idxs])
    new_labels = np.array([labels_3[i] for i in idxs])
    features = new_features
    labels = new_labels
    # predictions, _, _, rmse = linear([int(x['vehicles']) for x in data], 10)
    train_features, train_labels, valid_features, valid_labels = split(features, labels, 0.3)
    predictor_3 = MLP()
    predictor_3.n_classes = len(labels[0])
    predictor_3.n_input = features.shape[1]
    test_predictions_3 = predictor_3.fit(train_features, train_labels, valid_features, valid_labels, test_features_3)
    with open('/Users/kamaloddindarchinimaragheh/Desktop/mk_kinsey/result.txt', 'a') as file:
        for id, veh in zip(test_ids_3, test_predictions_3):
            file.write(str(id) + ',' + str(veh) + '\n')

    #############
    ############
    idxs = list(range(len(features_4)))
    np.random.shuffle(idxs)
    new_features = np.array([features_4[i] for i in idxs])
    new_labels = np.array([labels_4[i] for i in idxs])
    features = new_features
    labels = new_labels
    # predictions, _, _, rmse = linear([int(x['vehicles']) for x in data], 10)
    train_features, train_labels, valid_features, valid_labels = split(features, labels, 0.3)
    predictor_4 = MLP()
    predictor_4.n_classes = len(labels[0])
    predictor_4.n_input = features.shape[1]
    test_predictions_4 = predictor_4.fit(train_features, train_labels, valid_features, valid_labels, test_features_4)
    with open('/Users/kamaloddindarchinimaragheh/Desktop/mk_kinsey/result.txt', 'a') as file:
        for id, veh in zip(test_ids_4, test_predictions_4):
            file.write(str(id) + ',' + str(veh) + '\n')