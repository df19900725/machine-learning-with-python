"""
This module will evaluate all methods together for kaggle web traffic forecast
@Author: DuFei
@Created Time: 2019/11/02 22:01
"""

import numpy as np
import xgboost as xgb
import tensorflow as tf
from sklearn.neural_network import MLPRegressor

from util.logger_util import get_logger
from util.linear_algebra_util import normalize
from data.kaggle_wikipedia_traffic import get_kaggle_sample_data
from preprocess.features_engineering import log1p_and_normalize_value, make_time_features, \
    make_time_features_with_encoding, split_time_series_data, recover_log1p_and_normalize_value
from util.metric_util import smape
from util.plot_util import plot_regress_predict

logger = get_logger()


def get_test_data(input_df, values):
    X = np.stack(make_time_features(input_df.index), axis=-1)
    y = values

    return split_time_series_data(X, y, test_length=100)


def get_test_data_with_simple_normalize(input_df, values):
    X = normalize(np.stack(make_time_features(input_df.index), axis=-1))
    y = values

    return split_time_series_data(X, y, test_length=100)


def get_test_data_with_encoding_feature(input_df, values):
    X = np.stack(make_time_features_with_encoding(input_df.index), axis=-1)
    y = values

    return split_time_series_data(X, y, test_length=100)


if __name__ == '__main__':
    sample_df, index_name = get_kaggle_sample_data()
    raw_values, mean, std = log1p_and_normalize_value(np.expand_dims(sample_df.values, -1))
    print(f'raw values shape:{raw_values.shape}')

    X_train, X_test, y_train, y_test = get_test_data(sample_df, raw_values)
    X_train_simple_norm, X_test_simple_norm, y_train_simple_norm, y_test_simple_norm = get_test_data_with_simple_normalize(
        sample_df, raw_values)
    X_train_encoding, X_test_encoding, y_train_encoding, y_test_encoding = get_test_data_with_encoding_feature(
        sample_df, raw_values)

    logger.info(f'X_train shape:{X_train.shape} y_train shape:{y_train.shape} '
                f'X_test shape:{X_test.shape} y_test shape:{y_test.shape}')
    logger.info(f'X_train encoding shape:{X_train_encoding.shape} y_train encoding shape:{y_train_encoding.shape} '
                f'X_test encoding shape:{X_test_encoding.shape} y_test encoding shape:{y_test_encoding.shape}')

    models = []
    res = {}
    res_values = []

    # ------------------------XGBoost Regression------------------------
    logger.info(f'start to train XGBoost model...')
    xgb_reg = xgb.XGBRegressor(n_estimators=1000, objective='reg:squarederror')
    xgb_reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train)],
                early_stopping_rounds=50,
                verbose=False)
    xgb_res = np.expand_dims(recover_log1p_and_normalize_value(xgb_reg.predict(X_test), mean, std), -1)
    res['XGBoost_Predict'] = xgb_res
    res_values.append(xgb_res)
    models.append(xgb_reg)

    # ------------------------Multi-layer Perceptron------------------------
    logger.info(f'start to train sklearn mlp model...')
    mlp_regression = MLPRegressor(hidden_layer_sizes=(10, 50,), activation='tanh', solver='sgd',
                                  verbose=False, max_iter=1000, shuffle=True)
    mlp_regression.fit(X_train, np.squeeze(y_train))
    mlp_res = np.expand_dims(recover_log1p_and_normalize_value(mlp_regression.predict(X_test), mean, std), -1)
    # res['SciKitLearn_MLP_Predict'] = mlp_res
    res_values.append(mlp_res)
    models.append(mlp_regression)

    # ------------------------Deep Learning------------------------
    logger.info(f'start to train customer keras deep learning model...')
    keras_deep_mlp = tf.keras.Sequential()
    keras_deep_mlp.add(tf.keras.layers.Dense(10, input_shape=(X_train.shape[1],), activation=tf.keras.activations.relu))
    keras_deep_mlp.add(tf.keras.layers.Dense(20, input_shape=(10,), activation=tf.keras.activations.relu))
    keras_deep_mlp.add(tf.keras.layers.Dense(1, input_shape=(20,)))

    keras_deep_mlp.compile(optimizer=tf.keras.optimizers.SGD(),  # Optimizer
                           # Loss function to minimize
                           loss=tf.keras.losses.mean_absolute_error,
                           # List of metrics to monitor
                           metrics=[tf.keras.metrics.mean_absolute_error])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min')
    keras_deep_mlp.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.1, verbose=False,
                       callbacks=[es])
    keras_deep_mlp_res = recover_log1p_and_normalize_value(keras_deep_mlp.predict(X_test), mean, std)
    # res['Keras_MLP'] = keras_deep_mlp_res
    res_values.append(keras_deep_mlp_res)
    models.append(keras_deep_mlp)

    # ------------------------Deep Learning Simple norm------------------------
    logger.info(f'start to train customer keras deep learning model with simple norm...')
    keras_deep_mlp_simple_norm = tf.keras.Sequential()
    keras_deep_mlp_simple_norm.add(
        tf.keras.layers.Dense(10, input_shape=(X_train.shape[1],), activation=tf.keras.activations.relu))
    keras_deep_mlp_simple_norm.add(tf.keras.layers.Dense(20, input_shape=(10,), activation=tf.keras.activations.relu))
    keras_deep_mlp_simple_norm.add(tf.keras.layers.Dense(1, input_shape=(20,)))

    keras_deep_mlp_simple_norm.compile(optimizer=tf.keras.optimizers.SGD(),  # Optimizer
                                       # Loss function to minimize
                                       loss=tf.keras.losses.mean_absolute_error,
                                       # List of metrics to monitor
                                       metrics=[tf.keras.metrics.mean_absolute_error])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min')
    keras_deep_mlp_simple_norm.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.1, verbose=False,
                                   callbacks=[es])
    keras_deep_mlp_simple_norm_res = recover_log1p_and_normalize_value(keras_deep_mlp_simple_norm.predict(X_test), mean,
                                                                       std)
    # res['Keras_MLP_Simple_Norm'] = keras_deep_mlp_simple_norm_res
    res_values.append(keras_deep_mlp_simple_norm_res)
    models.append(keras_deep_mlp_simple_norm)

    # ------------------------Deep Learning with encoding feature------------------------
    logger.info(f'start to train customer keras deep learning model with feature encoding...')
    keras_encoding = tf.keras.Sequential()
    keras_encoding.add(
        tf.keras.layers.Dense(10, input_shape=(X_train_encoding.shape[1],), activation=tf.keras.activations.relu))
    keras_encoding.add(tf.keras.layers.Dense(20, input_shape=(10,), activation=tf.keras.activations.relu))
    keras_encoding.add(tf.keras.layers.Dense(20, input_shape=(20,), activation=tf.keras.activations.relu))
    keras_encoding.add(tf.keras.layers.Dense(1, input_shape=(20,)))

    keras_encoding.compile(optimizer=tf.keras.optimizers.SGD(),  # Optimizer
                           # Loss function to minimize
                           loss=tf.keras.losses.mean_squared_error,
                           # List of metrics to monitor
                           metrics=[tf.keras.metrics.mean_squared_error])

    keras_encoding.fit(X_train_encoding, y_train_encoding, epochs=500, batch_size=32, validation_split=0.1,
                       verbose=False)
    keras_encoding_res = recover_log1p_and_normalize_value(keras_encoding.predict(X_test_encoding), mean, std)
    models.append(keras_encoding)
    res_values.append(keras_encoding_res)
    res['Keras_MLP_Feature_Encoding'] = keras_encoding_res

    # ------------------------Model fusion------------------------
    res_values = np.stack(res_values, axis=-1)
    fusion_res = np.mean(res_values, axis=-1)
    res['Fusion_Predict'] = fusion_res

    y_test = recover_log1p_and_normalize_value(y_test, mean, std)

    plot_labels = []
    plot_result = []
    for predict_name in res:
        plot_labels.append(predict_name)
        plot_result.append(res[predict_name])
        print(f"{predict_name:<30} {smape(y_test, res[predict_name]):>10}")

    # plot the result
    plot_regress_predict(y_test, plot_result, plot_labels,
                         y_train=recover_log1p_and_normalize_value(y_train, mean, std),
                         title=f'Kaggle Web Traffic Forecast - {index_name}')
