from sklearn.preprocessing import MinMaxScaler
import numpy as np


# scale train and test data to [-1, 1]
def scale(train, test=None):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)

    # transform test
    try:
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = scaler.transform(test)
    except (TypeError, AttributeError):
        test_scaled = None
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
