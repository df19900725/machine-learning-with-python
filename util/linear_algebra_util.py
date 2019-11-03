"""
This module contains several method about linear algebra
@Author: DuFei
@Created Time: 2019/11/02 19:57
"""

import numpy as np
from util.logger_util import get_logger

logger = get_logger()


def normalize(input_array, return_mean_and_std=True):
    """
    Normalize data by mean and std: (x-mean)/std
    Link: https://en.wikipedia.org/wiki/Normalization_(statistics)
    :param input_array:
    :param return_mean_and_std: whether to return mean and std
    :return normalized data:
    """

    if len(input_array) == 1:
        logger.warning(f'dims of input data should be 2. currently is {input_array.shape}')
        input_array = np.expand_dims(input_array, -1)

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
