"""
This class is used to extract and explore kaggle traffic data set
Link:https://www.kaggle.com/c/web-traffic-time-series-forecasting/data
@Author: DuFei
@Created Time: 2019/11/02 19:31
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util.logger_util import get_logger
from util.linear_algebra_util import normalize

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.interactive(False)

logger = get_logger()

# data input file
train_file = 'D:/OneDrive/工作目录/27 kaggle/web-traffic-time-series-forecasting/raw data/train_1.csv'


def get_kaggle_raw_data():
    raw_df = pd.read_csv(train_file, index_col=0, header=0, sep=',')
    raw_df.columns = pd.to_datetime(raw_df.columns, format='%Y-%m-%d')
    return raw_df


def get_kaggle_sample_data(sample_index=None, return_index=True):
    raw_df = get_kaggle_raw_data()
    if sample_index is None:
        sample_index = f'王丹_zh.wikipedia.org_all-access_spider'

    if return_index:
        return raw_df.loc[sample_index], sample_index
    else:
        return raw_df.loc[sample_index]


def explore_data(input_df):
    """
    This method is used to explore the data set
    :param input_df:
    :return:
    """

    logger.info(f'input data shape:{input_df.shape}')

    start_time, end_time = input_df.columns[0], input_df.columns[-1]
    logger.info(f'time range:{start_time} - {end_time}')

    sample_index = u'王丹_zh.wikipedia.org_all-access_spider'

    raw_value = input_df.loc[sample_index].values
    log1p_value = np.log1p(raw_value)
    normal_value = normalize(log1p_value)

    plt.figure(1)
    plt.plot(raw_value)
    plt.title(f'{sample_index} - raw value')

    plt.figure(2)
    plt.plot(log1p_value)
    plt.title(f'{sample_index} - raw value with log1p')

    plt.figure(3)
    plt.plot(normal_value)
    plt.title(f'{sample_index} - raw value with normalize')
    plt.show()
    # for index in input_df.index:
    #     print(input_df.loc[index])


if __name__ == '__main__':
    kaggle_df = get_kaggle_raw_data()

    explore_data(kaggle_df)
