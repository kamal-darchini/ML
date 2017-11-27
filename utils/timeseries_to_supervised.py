from pandas import DataFrame, Series
from pandas import concat


def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def difference(data, interval=1):
    diff = list()
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return Series(diff)


# invert differencing
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def timeseries_to_normalized_supervised(data, history, difference_interval):
    # transform to be stationary
    diff_data = difference(data, difference_interval)

    # transform data to be supervised learning
    output_df = timeseries_to_supervised(diff_data, history)
    return output_df

