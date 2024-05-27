import pandas as pd
import matplotlib.pyplot as plt
from pyecharts.charts import Bar, Line, Pie
from pyecharts import options as opts
import xgboost as xgb
import os

# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


def dispose_offline(dataset):
    """
    预处理内容：
    1、折扣处理
    2、距离处理
    3、时间处理
    """
    offline = dataset.copy()
    # 填充Distance中的空值
    offline['Distance'].fillna(-1, downcast='infer', inplace=True)
    offline['null_distance'] = offline['Distance'].map(lambda x: 1 if x == -1 else 0)

    # 创建新的date_received，date，转化时间显示模式
    offline['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')
    # offline['date'] = pd.to_datetime(offline['Date'], format='%Y%m%d')

    # 找到折扣率
    offline['discount_rate'] = offline['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))

    # 添加满减列：
    offline['is_manjian'] = offline['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    return offline


def get_time_change(data_train):
    """
    单独对Date列进行转换
    """
    offline = data_train.copy()
    offline['date'] = pd.to_datetime(offline['Date'], format='%Y%m%d')
    return offline


def label(data_train):
    # 打标
    data_train['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0,
                                   data_train['date'],
                                   data_train['date_received']))
    return data_train


# 卖家和商家相关特征提取
def Get_User_And_Merchant_Related_Feature(label_field):
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)

    data['cnt'] = 1  # 方便特征提取
    # 返回的特征数据集
    feature = data.copy()

    ## 用户领券数
    keys = ['User_id']
    prefix = 'simple_' + '_'.join(keys) + '_'
    pivot = data.pivot_table(index=keys, values='cnt', aggfunc='count').reset_index()
    pivot.columns = keys + [prefix + 'receive_cnt']
    feature = pd.merge(feature, pivot, on=keys, how='left')

    ## 领券并消费人数：


def get_dataset(history_field, middle_field, label_field):
    """

    :param history_field:提取特征
    :param middle_field:空窗期
    :param label_field:考试测验相当于
    :return:
    """


if __name__ == '__main__':
    ##   读取数据
    data_train = pd.read_csv('D:\\Data\\opodata\\tabel3\\ccf_offline_stage1_train.csv')
    data_test = pd.read_csv('D:\\Data\\opodata\\tabel1\\ccf_offline_stage1_test_revised.csv')

    # 对数据预处理
    data_train = dispose_offline(data_train)
    data_train = get_time_change(data_train)
    # print('测试训练数据')
    print(data_train.head())

    data_test = dispose_offline(data_test)
    # print('测试实验数据')
    # print(data_test.head())

    ## 数据打标，领券后15天内有消费记录的打标为1
    data_train = label(data_train)
    # print(data_train)

    # 划分区间
    # 训练集特征
    # 提取特征
    train_history_field = data_train[
        data_train['date_received'].isin(pd.date_range('2016/3/2', periods=60))]
    # 空窗期
    train_middle_field = data_train[data_train['date'].isin(pd.date_range('2016/5/1', periods=15))]
    # 考察期
    train_label_field = data_train[
        data_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]

    # 验证集特征
    validate_history_field = data_train[(data_train.date_received >= '2016/1/16')
                     & (data_train.date_received <= '2016/3/16')]
    validate_middle_field = data_train[(data_train.date_received >= '2016/3/16')
                     & (data_train.date_received <= '2016/3/31')]
    validate_label_field = data_train[(data_train.date_received >= '2016/3/31')
                     & (data_train.date_received <= '2016/5/1')]

    # 测试集特征
    test_history_field = data_train[
        data_train['date_received'].isin(pd.date_range('2016/4/17', periods=60))]
    test_middle_field = data_train[(data_train.date_received >= '2016/6/16')
                     & (data_train.date_received <= '2016/7/1')]
    # 验证test文件
    test_label_field = data_test.copy()  # test作为测试集

    # 构造训练集、验证集、测试集
    train = get_dataset(train_history_field, train_middle_field, train_label_field)
    print('训练集构造完成')
    validate = get_dataset(validate_history_field, validate_middle_field, validate_label_field)
    print('验证集构造完成')
    test = get_dataset(test_history_field, test_middle_field, test_label_field)
    print('测试集构造完成')


