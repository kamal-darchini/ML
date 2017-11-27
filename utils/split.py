import random
import numpy as np


def split(features, labels, validation_ratio=0.3):

    number_features = features.shape[0]
    train_features = []
    train_labels = []
    valid_features = []
    valid_labels = []
    if labels:
        for i in range(number_features):
            if random.random() >= validation_ratio:
                train_features.append(features[i])
                train_labels.append(labels[i])
            else:
                valid_features.append(features[i])
                valid_labels.append(labels[i])
    else:
        for i in range(number_features):
            if random.random() >= validation_ratio:
                train_features.append(features[i])
            else:
                valid_features.append(features[i])

    return np.array(train_features), np.array(train_labels), np.array(valid_features), np.array(valid_labels)
