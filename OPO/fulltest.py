import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyecharts import options as opts
import os
from datetime import date
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score, auc,roc_curve
from  sklearn.model_selection import train_test_split
# 使用GridSearchCV进行参数搜索
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
# 绘制特征得分图
import matplotlib.pyplot as plt
from xgboost import plot_importance

"""
train数据：
列 'Date_received' 中有 418751 个缺失值。
列 'Coupon_id' 中有 418751 个缺失值。
列 'Merchant_id' 中没有缺失值。
列 'User_id' 中没有缺失值。
列 'Date' 中有 584858 个缺失值。
列 'Distance' 中有 62986 个缺失值。
列 'Discount_rate' 中有 418751 个缺失值。
"""
def prepare_data(data):
    # 找到折扣率
    data['discount_rate'] = data['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    # 添加满减列：
    data['is_manjian'] = data['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)

    return data

def getLabel(row): # 打标
    row = row.values
    a = str(row[0])
    b = str(row[1])
    if a=='null' or b=='null':
        return 0
    elif (date(int(b[0:4]),int(b[4:6]),int(b[6:8])) - date(int(a[0:4]),int(a[4:6]),int(a[6:8]))).days <= 15:
        return 1
    else:
        return 0

def time_change(data):
    data = data_train.copy()
    # 创建新的date_received，date，转化时间显示模式
    data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
    data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    return data



def getReceivedUseGap(dates):
    dates = dates.values
    # print(dates)
    receive,use = dates[0],dates[1]
    return (date(int(use[0:4]),int(use[4:6]),int(use[6:8])) -
            date(int(receive[0:4]),int(receive[4:6]),int(receive[6:8]))).days


# 用户特征分析
def get_User_Related_Feature(feature):
    """
    ##########提取的特征：
    User_receive_from_diff_Merchant:用户在不同商家领取
    User_buy_merchant_count:每个用户购买商品的不同商户数量
    User_max_distance：用户距离用消费券的店铺的最大值
    User_min_distance：用户距离用消费券的店铺的最小值
    User_mean_distance：用户距离用消费券的店铺的平均值
    User_median_distance：用户距离用消费券的店铺的中位数
    User_buy_use_coupon_count：用户使用优惠券消费次数
    User_buy_count：用户总体消费次数
    User_received_count：用户总共领取的消费券次数
    User_received_use_gap：用户领取了隔了几天才使用的次数
    User_received_use_max_gap：gap天数的最大值
    User_received_use_min_gap：gap天数的最小值
    User_received_use_mean_gap：gap天数的平均值
    User_browser_count：用户总数
    Discount_rate_mean：用户领券并消费部分的平均数
    User_no_buy_use_coupon_count：用户领券但是没有消费的数量
    ##概率：
    User_buy_use_coupon_goumai_rate：用户消费中使用优惠券率
    user_buy_use_coupon_hexiao_rate：用户领券中使用优惠券概率（核销率）


    :param feature:
    :return:
    """

    t = feature['User_id'].copy()
    t.drop_duplicates(inplace=True)

    # 特征：用户在不同商家领取
    t12=feature[feature['Date_received']!='null'][['User_id','Merchant_id']].copy()
    t12['User_receive_from_diff_Merchant']=1
    t12.groupby(['User_id']).agg('sum').reset_index()

    # 特征：用户在不同商家领券消费类数
    t1 = feature[(feature['Date']!='null')&(feature['Date_received']!='null')][['User_id','Merchant_id']].copy()
    # t1.drop_duplicates(inplace=True)
    t1['User_buy_from_diff_Merchant'] = 1
    t1 = t1.groupby('User_id').agg('sum').reset_index()
    # t1.rename(columns={'Merchant_id':'User_buy_from_diff_Merchant'},inplace=True)


    t2 = feature[(feature['Date']!='null') & (feature['Coupon_id']!='null')][['User_id','Distance']].copy()
    t2.replace('null',-1,inplace=True)
    t2['Distance'] = t2['Distance'].astype(float)
    t2.replace(-1,np.nan,inplace=True)

    # 特征：用户距离已用消费券消费店铺的最大、最小、平均、中位距离
    t2_1 = t2.groupby('User_id').agg('max').reset_index()
    t2_1.rename(columns={'Distance':'User_max_distance'},inplace=True)
    t2_2 = t2.groupby('User_id').agg('min').reset_index()
    t2_2.rename(columns={'Distance':'User_min_distance'},inplace=True)
    t2_3 = t2.groupby('User_id').agg('mean').reset_index()
    t2_3.rename(columns={'Distance':'User_mean_distance'},inplace=True)
    t2_4 = t2.groupby('User_id').agg('median').reset_index()
    t2_4.rename(columns={'Distance':'User_median_distance'},inplace=True)

    #特征：用户使用优惠券并消费次数
    t3 = feature[(feature['Coupon_id']!='null') & (feature['Date']!='null')][['User_id']].copy()
    t3['User_buy_use_coupon_count'] = 1
    t3 = t3.groupby('User_id').agg('sum').reset_index()

    # 特征：用户消费次数
    t4 = feature[(feature['Date']!='null')][['User_id']].copy()
    t4['User_buy_count'] = 1
    t4 = t4.groupby('User_id').agg('sum').reset_index()

    # 特征：用户领券优惠券数目（总数）
    t5 = feature[(feature['Coupon_id'] != 'null')][['User_id']].copy()
    t5['User_received_count'] = 1
    t5 = t5.groupby('User_id').agg('sum').reset_index()

    # 特征：用户领券并使用消费券间隔天数
    t6 = feature[(feature['Coupon_id'] != 'null') & (feature['Date'] != 'null')][['User_id', 'Date_received', 'Date']].copy()
    t6['User_received_use_gap'] = t6[['Date_received', 'Date']].apply(getReceivedUseGap, axis=1)
    t6 = t6[['User_id', 'User_received_use_gap']]

    # 特征：用户领券并使用优惠券的最大/最小/平均间隔天数
    t7 = t6.copy()
    t7_1 = t7.groupby('User_id').agg('max').reset_index()
    t7_1.rename(columns={'User_received_use_gap':'User_received_use_max_gap'},inplace=True)
    t7_2 = t7.groupby('User_id').agg('min').reset_index()
    t7_2.rename(columns={'User_received_use_gap':'User_received_use_min_gap'},inplace=True)
    t7_3 = t7.groupby('User_id').agg('mean').reset_index()
    t7_3.rename(columns={'User_received_use_gap':'User_received_use_mean_gap'},inplace=True)

    # 特征：用户总数
    t8 = feature[['User_id']].copy()
    t8['User_browser_count'] = 1
    t8 = t8.groupby('User_id').agg('sum').reset_index()



    #特征：用户领取优惠券但没有消费次数
    t10 = feature[(feature['Coupon_id']!='null') & (feature['Date']=='null')][['User_id']].copy()
    t10['User_no_buy_use_coupon_count'] = 1
    t10 = t10.groupby('User_id').agg('sum').reset_index()

    #特征：用户领取并消费的平均折扣率：
    #找到折扣率
    t11=feature[(feature['Coupon_id']!='null')&(feature['discount_rate']!='null')][['User_id','discount_rate']]

    t11=t11.groupby(['User_id']).agg('mean').reset_index()
    t11=t11.rename(columns={'discount_rate':'discount_rate_mean'},inplace=True)



    userFeature = pd.merge(t,t1,on='User_id',how='left')
    # userFeature = pd.merge(userFeature, t1, on='User_id', how='left')
    userFeature = pd.merge(userFeature,t2_1,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t2_2,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t2_3,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t2_4,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t3,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t4,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t5,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t6,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t7_1,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t7_2,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t7_3,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t8,on='User_id',how='left')
    userFeature = pd.merge(userFeature, t10, on='User_id', how='left')
    userFeature = pd.merge(userFeature, t12, on='User_id', how='left')


    # 特征：客户使用优惠券率：用户消费总量中使用优惠券占比
    userFeature['User_buy_use_coupon_goumai_rate'] = (userFeature['User_buy_use_coupon_count']
                                                      /userFeature['User_buy_count'])
    # 特征：优惠券的核销率：用户领券并消费的数目/用户领券数目
    userFeature['user_buy_use_coupon_hexiao_rate'] = (userFeature['User_buy_use_coupon_count']
                                                      /userFeature['User_received_count'])
    # 特征：用户在不同商家领取消费/在商家领取种数：定义为商家选取率：
    userFeature['seller_been_chosen']=(userFeature['User_buy_from_diff_Merchant']
                                       /userFeature['User_receive_from_diff_Merchant'])


    # 对于次数或者数目或占比，将Nan转换为0
    userFeature['User_buy_from_diff_Merchant'].replace(np.nan,0,inplace=True)
    userFeature['User_buy_use_coupon_count'].replace(np.nan,0,inplace=True)
    userFeature['User_buy_count'].replace(np.nan,0,inplace=True)
    userFeature['User_received_count'].replace(np.nan,0,inplace=True)
    userFeature['User_buy_use_coupon_goumai_rate'].replace(np.nan,0,inplace=True)
    userFeature['user_buy_use_coupon_hexiao_rate'].replace(np.nan, 0, inplace=True)
    userFeature['seller_been_chosen'].replace(np.nan,0,inplace=True)
    userFeature['User_no_buy_use_coupon_count'].replace(np.nan,0,inplace=True)

    return userFeature

# 商家特征分析
def get_Merchant_Related_Feature(feature):

    t = feature['Merchant_id'].copy()
    t.drop_duplicates(inplace=True)

    # 特征：商家卖出数目
    t1 = feature[(feature['Date']!='null')][['Merchant_id']].copy()
    t1['Merchant_sale_count'] = 1
    t1 = t1.groupby('Merchant_id').agg('sum').reset_index()

    # 特征：商家核销数目
    t2 = feature[(feature['Coupon_id']!='null') & (feature['Date']!='null')][['Merchant_id']].copy()
    t2['Merchant_sale_use_coupon_count'] = 1
    t2 = t2.groupby('Merchant_id').agg('sum').reset_index()

    # 特征：商家优惠券的总数量
    t3 = feature[(feature['Coupon_id']!='null')][['Merchant_id']].copy()
    t3['Merchant_give_count'] = 1
    t3 = t3.groupby('Merchant_id').agg('sum').reset_index()

    t4 = feature[(feature['Coupon_id']!='null') & (feature['Date']!='null')][['Merchant_id','Distance']].copy()
    t4['Distance'].replace('null',-1,inplace=True)
    t4['Distance'] = t4['Distance'].astype(float)
    t4['Distance'].replace(-1,np.nan,inplace=True)
    # 特征：商家已核销优惠券中距离的最小\最大\平均\中值
    t4_1 = t4.groupby('Merchant_id').agg('max').reset_index()
    t4_1.rename(columns={'Distance':'Merchant_max_distance'},inplace=True)
    t4_2 = t4.groupby('Merchant_id').agg('min').reset_index()
    t4_2.rename(columns={'Distance':'Merchant_min_distance'},inplace=True)
    t4_3 = t4.groupby('Merchant_id').agg('mean').reset_index()
    t4_3.rename(columns={'Distance':'Merchant_mean_distance'},inplace=True)

    merchantFeature = pd.merge(t,t1,on='Merchant_id',how='left')
    merchantFeature = pd.merge(merchantFeature,t2,on='Merchant_id',how='left')
    merchantFeature = pd.merge(merchantFeature,t3,on='Merchant_id',how='left')
    merchantFeature = pd.merge(merchantFeature,t4_1,on='Merchant_id',how='left')
    merchantFeature = pd.merge(merchantFeature,t4_2,on='Merchant_id',how='left')
    merchantFeature = pd.merge(merchantFeature,t4_3,on='Merchant_id',how='left')

    # 特征：商家卖出总量中优惠券的核销比
    merchantFeature['Merchant_sale_use_coupon_rate'] = (merchantFeature['Merchant_sale_use_coupon_count']
                                                        /merchantFeature['Merchant_sale_count'])
    # 特征：商家发放总量中优惠券的核销比
    merchantFeature['Merhcant_give_coupon_use_rate'] = (merchantFeature['Merchant_sale_use_coupon_count']
                                                        /merchantFeature['Merchant_give_count'])

    # 次数项目和占比类型数据，Nan用0替代(之所以最后转化，是防止上两个特征提取时出现分母为零溢出)（另外，上两个特征值的计算，只要分子分母一个为pd.nan结果就为nd.nan）
    merchantFeature['Merchant_sale_use_coupon_count'].replace(np.nan,0,inplace=True)
    merchantFeature['Merchant_sale_count'].replace(np.nan,0,inplace=True)
    merchantFeature['Merchant_give_count'].replace(np.nan,0,inplace=True)
    merchantFeature['Merchant_sale_use_coupon_rate'].replace(np.nan,0,inplace=True)
    merchantFeature['Merhcant_give_coupon_use_rate'].replace(np.nan,0,inplace=True)


    return merchantFeature

# 优惠券特征分析
def Coupon_Related_future(dataset):
    """
    Coupon_give_weekday:消费券发放的星期几
    Coupon_give_monthday：消费券发放的月份几号数目
    Coupon_discount_type：是否满减
    Coupon_discount_man：满多少触发
    Coupon_discount_jian：减多少触发
    Coupon_discount_rate：打折率
    Coupon_count：数目
    :param dataset:
    :return:
    """

    t = dataset.copy()
    # 这里dataset无重复值，不用drop_duplicates()

    # 特征：消费券发放的周或月份
    #20160101  对时间数字进行切片处理
    t['Coupon_give_weekday'] = t['Date_received'].astype(str).apply(
        lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)
    t['Coupon_give_monthday'] = t['Date_received'].astype(str).apply(lambda x: int(x[6:8]))

    t['Discount_rate'] = t['Discount_rate'].astype(str)
    # 特征：消费券是否是满减类型   ：表示满减多少
    t['Coupon_discount_type'] = t['Discount_rate'].apply(lambda s: 1 if ':' in s else 0)
    # 特征：消费券满减的满
    t['Coupon_discount_man'] = t['Discount_rate'].apply(lambda s: int(s.split(':')[0]) if ':' in s else 0)
    # 特征：消费券减
    t['Coupon_discount_jian'] = t['Discount_rate'].apply(lambda s: int(s.split(':')[1]) if ':' in s else 0)
    # 特征：优惠券打折率（在写一次）
    t['Coupon_discount_rate'] = t['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))

    # 特征：每种优惠券的数目
    t1 = dataset[['Coupon_id']].copy()
    t1['Coupon_count'] = 1
    t1 = t1.groupby('Coupon_id').agg('sum').reset_index()

    couponFeature = pd.merge(t, t1, on='Coupon_id', how='left')


    return couponFeature

def isWeekend(day):
    if day>=1 and day<=5:
        return 0
    else:
        return 1

def featureProcess(dataset,feature,processFlag):
    user = get_User_Related_Feature(feature)
    merchant = get_Merchant_Related_Feature(feature)
    coupon = Coupon_Related_future(dataset)


    allFeature = pd.merge(coupon,user,on='User_id',how='left')
    allFeature = pd.merge(allFeature,merchant,on='Merchant_id',how='left')


    allFeature['Coupon_give_weekday_is_weekend'] = allFeature['Coupon_give_weekday'].apply(isWeekend)
    weekday_dummies = pd.get_dummies(allFeature['Coupon_give_weekday'])
    weekday_dummies.columns = ['Coupon_give_weekday_' + str(i) for i in range(1,weekday_dummies.shape[1]+1)]
    allFeature = pd.concat([allFeature,weekday_dummies],axis=1)
    allFeature.drop('Coupon_give_weekday',axis=1,inplace=True)

    if processFlag:
        allFeature['Label'] = allFeature[['Date_received','Date']].apply(getLabel,axis=1)
        allFeature.drop(['User_id','Date_received','Coupon_id','Merchant_id','Discount_rate','Date'],axis=1,inplace=True)
    else:
        # 'User_id','Date_received','Coupon_id'字段需要在提交文档中,先留下
        allFeature.drop(['Merchant_id','Discount_rate'],axis=1,inplace=True)
    allFeature.replace('null',np.nan,inplace=True)

    return allFeature




if __name__ == '__main__':

    data_train = pd.read_csv('D:\\Data\\opodata\\tabel3\\ccf_offline_stage1_train.csv',
                             header=0, keep_default_na=False)
    data_test = pd.read_csv('D:\\Data\\opodata\\tabel1\\ccf_offline_stage1_test_revised.csv',
                            header=0, keep_default_na=False)
    # 找到折扣率（data_train)
    data_train['Discount_rate'] = data_train['Discount_rate'].replace('null', np.nan)
    data_train['discount_rate'] = data_train['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    data_train['discount_rate'] = data_train['discount_rate'].replace(np.nan,'null')

    data_test['discount_rate'] = data_test['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    print(data_test.head(5))
    print(data_train.head(5))

    Path = r'D:\Data\opodata'

    # 划分区间
    # 训练集特征
    # 提取特征
    train_history_field = data_train[(data_train.Date_received >= '20160302')
                     & (data_train.Date_received <= '20160501')]
    train_middle_field = data_train[(data_train.Date_received >= '20160501')
                     & (data_train.Date_received <= '20160516')]
    train_label_field = data_train[(data_train.Date_received >= '20160516')
                     & (data_train.Date_received <= '20160616')]

    # 验证集特征
    validate_history_field = data_train[(data_train.Date_received >= '20160116')
                     & (data_train.Date_received <= '20160316')]
    validate_middle_field = data_train[(data_train.Date_received >= '20160316')
                     & (data_train.Date_received <= '20160331')]
    validate_label_field = data_train[(data_train.Date_received >= '20160331')
                     & (data_train.Date_received <= '20160501')]

    # 测试集特征
    test_history_field = data_train[(data_train.Date_received >= '20160417')
                     & (data_train.Date_received <= '20160616')]
    test_middle_field = data_train[(data_train.Date_received >= '20160616')
                     & (data_train.Date_received <= '20160701')]
    test_label_field = data_train[(data_train.Date_received >= '20160701')
                     & (data_train.Date_received <= '20160801')]

    # 验证test文件
    test_label_field = data_test.copy()  # test作为测试集
    test_label_field['Date_received'] = test_label_field['Date_received'].astype(str)


    df1 = featureProcess(train_label_field, train_history_field, True)  # train
    df1.to_csv(Path + r'\df1.csv')
    print('df1 write over')

    df2 = featureProcess(validate_label_field, validate_history_field, True)  #  validate
    df2.to_csv(Path + r'\df2.csv')
    print('df2 write over')

    df3 = featureProcess(test_label_field, test_history_field, False)  # test
    df3.to_csv(Path + r'\df3.csv')
    print('df3 write over')

    ##########################       训练
    ######### 新数据
    Path = r'D:\Data\opodata'

    train = pd.read_csv(Path + r'\df1.csv', index_col=0)
    validate = pd.read_csv(Path + r'\df2.csv', index_col=0)
    test = pd.read_csv(Path + r'\df3.csv', index_col=0)
    # 输出保留三列
    print(train.columns)
    test_preds = test[['User_id', 'Coupon_id', 'Date_received']].copy()
    test_x = test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1)

    dataset_12 = pd.concat([train, validate], axis=0)
    dataset_12_y = dataset_12.Label
    dataset_12_x = dataset_12.drop(['Label'], axis=1)

    dataTrain = xgb.DMatrix(dataset_12_x, label=dataset_12_y)
    dataTest = xgb.DMatrix(test_x)
    print('---data prepare over---')

    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'gamma': 0,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.9,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.05,
              'tree_method': 'exact',
              'seed': 0,
              }

    watchlist = [(dataTrain, 'train')]
    model = xgb.train(params, dataTrain, num_boost_round=50, evals=watchlist)
    # 然后进行预测

    print('start predict')
    test_preds1 = test_preds
    test_preds1['Label'] = model.predict(dataTest)
    print(type(test_preds1.Label))
    test_preds1['Label'] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(
        test_preds1['Label'].values.reshape(-1, 1))
    test_preds1.to_csv(Path + r'\sample_submission.csv', index=None, header=True)
    print('write over')