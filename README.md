# machine-learning-with-python
This is a repository that use python to do machine learning tasks. It aims to provide kaggle competition solutions in one project. 

For each kaggle competition, the repository will create a project package to provide several solutions with different models and strategies. It will cover traditional algorithms such as regression, svm and so on. Deep learning models will be also provided by using TensorFlow. Since I have no any GPUs, all deep learning models are written for CPUs.

Currentyly, it contains several packages:

data - The package that contains load different data set methods.

model - This pakcage provides some customer models

preprocess - This package contains several modules including feature engineering

project - Including kaggle competitions. Includings:

Kaggle Web Traffic Prediction (https://www.kaggle.com/robikscube/tutorial-time-series-forecasting-with-xgboost): Since my personal computer can not run all data set. I just pick up serveral traffic sequences to test our models.

util - This package contains several utilize modules that will be used by other modules. Includings:

liearn_algebra_util.py - some useful method for math
logger_util.py - logger util module
metric_util.py - some methods for evaluation metric algorithms
plot_util.py - some methods that will be used to plot result

