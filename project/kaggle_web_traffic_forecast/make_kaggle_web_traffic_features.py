"""
This module is used to extract features for kaggle web traffic sequence features
@Author: DuFei
@Created Time: 2019/11/03 15:07
"""

from preprocess.features_engineering import *
from data.kaggle_wikipedia_traffic import get_kaggle_sample_data


def get_raw_data(input_df, raw_values, predict_length):
    """
    This method returns sequence value by basic process. It will compute log1p for raw values and
    then normalized it by mean and std. Here we use basic time features. see make_time_features() method
    :param input_df: input sample pandas.DataFrame, row is time_index, and columns is values for specific seq
    :param raw_values: sequence values
    :param predict_length: the date length for prediction
    :return X_train, X_test, y_train, y_test:
    """
    x = np.stack(make_time_features(input_df.index), axis=-1)
    y = raw_values

    return split_time_series_data(x, y, test_length=predict_length)


def get_data_with_feature_normalized(input_df, raw_values, predict_length):
    """
    This method returns sequence value by basic process. It will compute log1p for raw values and
    then normalized it by mean and std. Additionally, it will normalize time features.
    :param input_df: input sample pandas.DataFrame, row is time_index, and columns is values for specific seq
    :param raw_values: sequence values
    :param predict_length: the date length for prediction
    :return X_train, X_test, y_train, y_test:
    """
    x = normalize(np.stack(make_time_features(input_df.index), axis=-1))
    y = raw_values

    return split_time_series_data(x, y, test_length=predict_length)


def get_data_with_feature_encoding(input_df, raw_values, predict_length):
    """
    This method returns sequence value by basic process. It will compute log1p for raw values and then normalized it
    by mean and std. Additionally, it will normalize time features and then encoding each time feature by [cos, sin].
    You can also encoding time features by one-hot.
    :param input_df: input sample pandas.DataFrame, row is time_index, and columns is values for specific seq
    :param raw_values: sequence values
    :param predict_length: the date length for prediction
    :return X_train, X_test, y_train, y_test:
    """
    x = np.stack(make_time_features_with_encoding(input_df.index), axis=-1)
    y = raw_values

    return split_time_series_data(x, y, test_length=predict_length)


def get_data_with_lagged_feature(input_df, raw_values, predict_length, lagged_count=None):
    """
    This method will generate lagged values as features which will be concatenate with original time features.
    Lagged value means the sequence value that some days age. For example, if lagged_count=[7, 14], it means we will
    retrieve traffic number 7 days ago and 14 days ago as current point features. Note that not all time point have
    lagged values features. For example, if we set lagged_count=7, then the first 6 days of the sequence do not have
    any such lagged value feature. Additionally, if we predict next 50 days traffic, then the last 45 days have no any
    lagged values. So, it is proper to cut these days. For prediction, we can fill the predict value by time step.
    :param input_df:
    :param raw_values:
    :param predict_length:
    :param lagged_count:
    :return:
    """
    if lagged_count is None:
        lagged_count = [60, 90, 120, 150]
    cropped_values = np.ndarray.copy(raw_values)
    time_feature = np.stack(make_time_features_with_encoding(input_df.index), axis=-1)

    # we should remove real values that will be predicted
    cropped_values[-predict_length:] = 0

    start_date, end_date = input_df.index[0], input_df.index[-1]
    lagged_index = np.stack(lag_indexes(start_date, end_date, lagged_count), -1)
    lagged_values = get_lagged_time(cropped_values, lagged_index)

    all_features = np.concatenate((time_feature, lagged_values), axis=-1)

    return split_time_series_data(all_features, raw_values, predict_length)


if __name__ == '__main__':
    # test
    sample_df, index_name = get_kaggle_sample_data()
    input_values, mean, std = log1p_and_normalize_value(np.expand_dims(sample_df.values, -1))
    get_data_with_lagged_feature(sample_df, input_values, 100)
