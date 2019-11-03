"""
This module provides several method that are used to process data set.
@Author: DuFei
@Created Time: 2019/11/02 19:31
"""

import numpy as np


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
