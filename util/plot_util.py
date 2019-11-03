"""
This module is used to plot
@Author: DuFei
@Created Time: 2019/11/02 21:05
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_regress_predict(y_test, predict_list, predict_label_list=None, y_train=None, title=''):
    """
    This method is used to plot real prediction with true values
    :param y_train: training values
    :param y_test: real test values
    :param predict_list: predict result list, you can provide different prediction result by different methods here to
                         compare
    :param predict_label_list: predict label list, this is prediction result name list corresponds to predict_List
                               parameter. its length should equals to predict_list
    :param title: title of the plot
    """
    if predict_list is not None:
        assert len(predict_list) == len(predict_label_list)

    if y_train is not None:
        train_x_axis = np.arange(0, len(y_train))
        test_x_axis = np.arange(len(y_train), len(y_train) + len(y_test))
        plt.plot(train_x_axis, y_train, label='training', color='blue')
    else:
        test_x_axis = np.arange(len(y_test))

    for predict_index, predict_res in enumerate(predict_list):
        if predict_label_list is not None:
            label = predict_label_list[predict_index]
        else:
            label = f'predict {predict_index}'
        plt.plot(test_x_axis, predict_res, label=label)

    plt.plot(test_x_axis, y_test, label='true value')
    plt.title(title)
    plt.legend()
    plt.show()
