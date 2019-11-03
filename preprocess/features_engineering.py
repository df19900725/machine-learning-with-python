"""
This module is used to make various features for data set
@Author: DuFei
@Created Time: 2019/11/02 19:31
"""

import numpy as np
from pandas import DatetimeIndex
from util.linear_algebra_util import normalize


def split_time_series_data(input_x, input_y, test_length: int):
    """
    This method is used to split time series data into train data set and test data set.
    It can not by split randomly since time series need an continuous data set
    :param input_x: features, shape [length, feature_number]
    :param input_y: label, shape [length, 1]
    :param test_length: how long will the test data be
    :return: X_train, X_test, y_train, y_test
    """
    X_train, X_test = np.split(input_x, [len(input_x) - test_length], axis=0)
    y_train, y_test = np.split(input_y, [len(input_y) - test_length], axis=0)

    return X_train, X_test, y_train, y_test


def log1p_and_normalize_value(input_values):
    """
    This method will firstly compute values by log1p and then normalize values
    :param input_values:
    :return result, mean ,std
    """
    return normalize(np.log1p(input_values))


def recover_log1p_and_normalize_value(res, mean, std):
    """
    This method will compute raw values, it's the reverse method of log1p_and_normalize_value()
    :param res: result
    :param mean: mean
    :param std: std
    :return true value:
    """
    return np.expm1((res * std) + mean)


def make_time_features(time_feature: DatetimeIndex):
    """
    Compute basic time features and return raw time features
    :param time_feature: input time feature, it should by pandas.DateTimeIndex list
    :return time features list
    """
    hour_of_day = time_feature.hour
    day_of_year = time_feature.dayofyear
    day_of_month = time_feature.day
    day_of_week = time_feature.dayofweek
    week_of_year = time_feature.weekofyear
    month_of_year = time_feature.month
    quarter_of_year = time_feature.quarter
    year = time_feature.year

    return [hour_of_day, day_of_year, day_of_month, day_of_week, week_of_year, month_of_year, quarter_of_year, year]


def make_time_features_normalized(time_feature: DatetimeIndex):
    """
    Compute basic time features and transform feature values by radian
    Link: https://github.com/Arturus/kaggle-web-traffic/blob/master/make_features.py
    :param time_feature: input time feature, it should by pandas.DateTimeIndex list
    :return time features list
    """
    hour_of_day = time_feature.hour / (24 / (2 * np.pi))
    day_of_year = time_feature.dayofyear / (366 / (2 * np.pi))
    day_of_month = time_feature.day / (31 / (2 * np.pi))
    day_of_week = time_feature.dayofweek / (7 / (2 * np.pi))
    week_of_year = time_feature.weekofyear / (52 / (2 * np.pi))
    month_of_year = time_feature.month / (12 / (2 * np.pi))
    quarter_of_year = time_feature.quarter / (4 / (2 * np.pi))

    return [hour_of_day, day_of_year, day_of_month, day_of_week, week_of_year, month_of_year, quarter_of_year]


def make_time_features_with_encoding(input_time: DatetimeIndex):
    encoding_res = []
    time_feature_list = make_time_features_normalized(input_time)
    for time_feature in time_feature_list:
        encoding_res.append(np.cos(time_feature))
        encoding_res.append(np.sin(time_feature))

    return encoding_res
