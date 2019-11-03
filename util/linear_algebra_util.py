"""
This module contains several method about linear algebra
@Author: DuFei
@Created Time: 2019/11/02 19:57
"""

import numpy as np


def normalize(input_array, return_mean_and_std=True):
    """
    Normalize data by mean and std: (x-mean)/std
    Link: https://en.wikipedia.org/wiki/Normalization_(statistics)
    :param input_array:
    :param return_mean_and_std: whether to return mean and std
    :return normalized data:
    """

    mean = np.mean(input_array, axis=0)
    std = np.std(input_array, ddof=1, axis=0)

    # if std equals zero, we keep raw values
    std[std == 0] = 1
    mean[std == 0] = 0

    res = (input_array - mean) / std

    print(f'{input_array.shape} {mean.shape} {std.shape} {res.shape}')

    if return_mean_and_std:
        return res, mean, std
    else:
        return res
