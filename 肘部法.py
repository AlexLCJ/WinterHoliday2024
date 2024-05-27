from 数据集2 import get_distance
from 数据集2 import  KMeans_train
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_csv('C:\\Users\\李昌峻\\Desktop\\NPHA-doctor-visits.csv')
print(data.shape)

# (568, 32) 32个特征值 需要进行降维到2纬
# 数据中带有字符串类型，对这一列进行删除，只保留数字

# 读取CSV文件
df = pd.read_csv('C:\\Users\\李昌峻\\Desktop\\NPHA-doctor-visits.csv')

# 删除带有字母的列
df = df.select_dtypes(exclude=['M'])

# 保存修改后的数据到新的CSV文件
df.to_csv('C:\\Users\\李昌峻\\Desktop\\NPHA-doctor-visits.csv', index=False)

from sklearn.manifold import TSNE
tsne=TSNE()
data=tsne.fit_transform(data)
print(data.shape)

# (568, 2) 此时降维到2纬
max_train=500
res=[]
for K in range(1,15):
    cluster, initial_centroids=KMeans_train(data, K, max_train)
    sum=0
    for i,cluster in enumerate(cluster):
        _distance=get_distance(data,cluster[i],K)
        mean=np.mean(_distance)
        sum+=mean
    mean=sum/K
    res.append(mean)
print(res)