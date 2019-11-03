"""
This module uses simple deep learning regression to forecast web traffic.
@Author: DuFei
@Created Time: 2019/11/02 22:04
"""

import numpy as np
from sklearn.neural_network import MLPRegressor

from data.kaggle_wikipedia_traffic import get_kaggle_sample_data
from util.logger_util import get_logger
from util.metric_util import smape
from util.plot_util import plot_regress_predict
from preprocess.features_engineering import log1p_and_normalize_value, recover_log1p_and_normalize_value, \
    make_time_features, split_time_series_data

logger = get_logger()


def evaluate_by_mlp(input_df):
    raw_values, mean, std = log1p_and_normalize_value(input_df.values)

    X = np.stack(make_time_features(input_df.index), axis=-1)
    y = np.expand_dims(raw_values, -1)

    X_train, X_test, y_train, y_test = split_time_series_data(X, y, test_length=100)

    logger.info(f'raw X shape:{X.shape} raw y shape:{y.shape}')
    logger.info(f'X_train shape:{X_train.shape} y_train shape:{y_train.shape} '
                f'X_test shape:{X_test.shape} y_test shape:{y_test.shape}')

    mlp_regression = MLPRegressor(hidden_layer_sizes=(10, 50,), activation='logistic', solver='sgd', early_stopping=True)
    mlp_regression.fit(X_train, y_train)

    y_predict = mlp_regression.predict(X_test)
    real_predict = recover_log1p_and_normalize_value(y_predict, mean, std)
    real_y = recover_log1p_and_normalize_value(y_test, mean, std)

    return real_predict, real_y, recover_log1p_and_normalize_value(y_train, mean, std)


if __name__ == '__main__':
    sample_df, index_name = get_kaggle_sample_data()

    real_normalized_predict, real_y_test, real_y_train = evaluate_by_mlp(sample_df)

    normalized_smape = smape(real_y_test, real_normalized_predict)
    print(f'normalized smape result:{normalized_smape}')

    # plot the result
    plot_regress_predict(real_y_test,
                         [real_normalized_predict],
                         ['normalized prediction'],
                         y_train=real_y_train,
                         title=index_name)