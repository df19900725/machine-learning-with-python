"""
This module uses XGBoost Regression to forecast web traffic.
Link: https://www.kaggle.com/robikscube/tutorial-time-series-forecasting-with-xgboost
@Author: DuFei
@Created Time: 2019/11/02 19:31
"""

import numpy as np
import xgboost as xgb

from util.logger_util import get_logger
from util.data_util import split_time_series_data
from util.plot_util import plot_regress_predict
from util.metric_util import smape
from preprocess.features_engineering import make_time_features, log1p_and_normalize_value, \
    recover_log1p_and_normalize_value
from data.kaggle_wikipedia_traffic import get_kaggle_sample_data

logger = get_logger()


def evaluate_by_raw_data(input_df):
    raw_values = input_df.values

    X = np.stack(make_time_features(input_df.index), axis=-1)
    y = raw_values

    X_train, X_test, y_train, y_test = split_time_series_data(X, y, test_length=100)

    logger.info(f'raw X shape:{X.shape} raw y shape:{y.shape}')
    logger.info(f'X_train shape:{X_train.shape} y_train shape:{y_train.shape} '
                f'X_test shape:{X_test.shape} y_test shape:{y_test.shape}')

    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train)],
            early_stopping_rounds=50,
            verbose=False)

    return reg.predict(X_test), y_test, y_train


def evaluate_by_normalized_data(input_df):
    raw_values, mean, std = log1p_and_normalize_value(input_df.values)

    X = np.stack(make_time_features(input_df.index), axis=-1)
    y = np.expand_dims(raw_values, -1)

    X_train, X_test, y_train, y_test = split_time_series_data(X, y, test_length=100)

    logger.info(f'raw X shape:{X.shape} raw y shape:{y.shape}')
    logger.info(f'X_train shape:{X_train.shape} y_train shape:{y_train.shape} '
                f'X_test shape:{X_test.shape} y_test shape:{y_test.shape}')

    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train)],
            early_stopping_rounds=50,
            verbose=False)

    y_predict = reg.predict(X_test)
    real_predict = recover_log1p_and_normalize_value(y_predict, mean, std)
    real_y = recover_log1p_and_normalize_value(y_test, mean, std)

    return real_predict, real_y, recover_log1p_and_normalize_value(y_train, mean, std)


if __name__ == '__main__':
    sample_df, index_name = get_kaggle_sample_data()

    real_raw_predict, real_y_test, real_y_train = evaluate_by_raw_data(sample_df)
    real_normalized_predict, real_y_test2, real_y_train2 = evaluate_by_normalized_data(sample_df)

    raw_smape = smape(real_y_test, real_raw_predict)
    normalized_smape = smape(real_y_test, real_normalized_predict)
    print(f'raw smape result:{raw_smape}')
    print(f'normalized smape result:{normalized_smape}')

    # plot the result
    plot_regress_predict(real_y_test,
                         [real_raw_predict, real_normalized_predict],
                         ['raw prediction', 'normalized prediction'],
                         y_train=real_y_train,
                         title=index_name)
