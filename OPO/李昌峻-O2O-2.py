import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

params={'booster':'gbtree',
	    'objective': 'binary:logistic',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact'
	    }

watchlist = [(dataTrain, 'train')]
evals_result = {}
model = xgb.train(params, dataTrain, num_boost_round=50, evals=watchlist, evals_result=evals_result)

# Plotting the AUC curve for the training data
train_auc = evals_result['train']['auc']
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_auc) + 1), train_auc, label='Train AUC')
plt.xlabel('Number of boosting rounds')
plt.ylabel('AUC')
plt.title('Training AUC Curve')
plt.legend()
plt.grid(True)
plt.show()

# 然后进行预测
print('start predict')
test_preds1 = test_preds
test_preds1['Label'] = model.predict(dataTest)
print(type(test_preds1.Label))
test_preds1['Label'] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(
    test_preds1['Label'].values.reshape(-1, 1))
test_preds1.to_csv(Path + r'\sample_submission.csv', index=None, header=True)
print('write over')
