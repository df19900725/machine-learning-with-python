"""
This module uses Seq 2 Seq model to forecast web traffic.
@Author: DuFei
@Created Time: 2019/11/02 19:31
"""

from preprocess.make_rnn_feature import InputPipe
from project.kaggle_web_traffic_forecast.make_kaggle_web_traffic_features import *

if __name__ == '__main__':
    sample_df, index_name = get_kaggle_sample_data()
    raw_values, mean, std = log1p_and_normalize_value(np.expand_dims(sample_df.values, -1))
    print(f'raw values shape:{raw_values.shape}')
    predict_length = 50

    X_train, X_test, y_train, y_test = get_raw_data(sample_df, raw_values, predict_length)
    X_train_simple_norm, X_test_simple_norm, y_train_simple_norm, y_test_simple_norm = get_data_with_feature_normalized(
        sample_df, raw_values, predict_length)
    X_train_encoding, X_test_encoding, y_train_encoding, y_test_encoding = get_data_with_feature_encoding(
        sample_df, raw_values, predict_length)
    X_train_with_lagged, X_test_with_lagged, y_train_with_lagged, y_test_with_lagged = get_data_with_feature_encoding(
        sample_df, raw_values, predict_length)

    print(f'{X_train_with_lagged.shape} {X_test_with_lagged.shape}')

    x = np.zeros((1, 100, 10))
    y = np.zeros((1, 100, 1))
    sample_num = 100
    for i in range(sample_num):
        value = np.zeros(10)
        value.fill(i)
        x[0][i] = value
        y[0][i] = i * 2

    print(x)
    print(y)

    input_x = np.expand_dims(X_train_with_lagged, 0)
    input_y = np.expand_dims(y_train_with_lagged, 0)

    print(f'{input_x.shape} {input_y.shape}')

    input_pipe = InputPipe(x, y, 20, 10)

    print('------------------------')

    print(input_pipe.encoder_features_all[1])
    print(input_pipe.encoder_features_all.shape)

