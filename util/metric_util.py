"""
This module will provide some evaluation method
@Author: DuFei
@Created Time: 2019/11/02 21:33
"""

import numpy as np


def smape(y_true, y_predict):
    """
    Symmetric mean absolute percentage error(https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)
    :param y_true: true y values, shape:[samples,1]
    :param y_predict: prediction values, shape:[samples,1]
    :return smape:
    """
    assert y_true.shape == y_predict.shape
    return 100 / len(y_true) * np.sum(2 * np.abs(y_predict - y_true) / (np.abs(y_true) + np.abs(y_predict)))
