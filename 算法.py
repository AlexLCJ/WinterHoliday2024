import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
    步骤：

"""


# 随机初始化数据中心
def random_init_centroids(data, K):
    num=data.shape[0]
    parts = np.random.permutation(num)
    # 取到所有的中心
    centroids = data[parts[:K], :]
    return centroids


# 获得欧氏距离
def get_distance(data, centroids, K):
    # 运用np库进行欧氏距离的计算
    for i in range(len(data)):
        distance = np.sqrt(np.sum(np.square(np.tile(data[i], (K, 1)) - centroids)))
        # 找到最进的距离点
        min_distance = np.where(distance == np.min(distance))
        # 提出数字
        min_distance = min_distance[0][0]
    return distance, min_distance


# 样本分类
def classify_cluster(data, centroids, K):
    # 运用cluster来存储各个类别
    cluster = []
    # 传入两个数值
    distance = get_distance(data, centroids, K)[0]
    min_distance = get_distance(data, centroids, K)[1]
    # 分类
    for i in range(len(data)):
        cluster[i] = min_distance.astype(int)
    return cluster


# 重新计算中心
def new_centroids(data, K, cluster, centroids):
    # 对每一个聚类中心进行遍历
    for j in range(len(K)):
        index = (np.where(cluster == j))[0]
        # 总和除以个数得到均值（样本中心）
        centroids[j] = np.sum(data, axis=0) / len(data)
    return centroids


# 定义kmeans算法，开始训练！
def KMeans_train(data, K, max_train):  # max_train 最大迭代训练次数
    initial_centroids = random_init_centroids(data, K)  # 随机初始化质心坐标
    for i in range(max_train):  # 迭代训练次数
        cluster = classify_cluster(data, initial_centroids, K)  # 分类和计算距离
        initial_centroids = new_centroids(data, K, cluster, initial_centroids)  # 重新计算质心坐标
    return cluster, initial_centroids, K


# 数据传输以及使用算法计算

# 数据没有列名，我自己定义
column_names=['X','Y','class']
data = pd.read_csv('C:\\Users\\李昌峻\\Desktop\\iris.csv',names=column_names)
iris_name=['Iris-setosa','Iris-versicolor','Iris-virginica']
# 定义x，y轴
x_axis ='X'
y_axis ='Y'

# 画图

# 数据对比，通过已知的标签和kmeans计算无标签的数据比较
# 有标签：
for i in iris_name:
    plt.scatter(data[x_axis][data['class']==i],data[y_axis][data['class']==i],label=iris_name)
plt.title('Name known')
plt.show()

# 无标签
for i in iris_name:
    plt.scatter(data[x_axis][:],data[y_axis][:])
plt.title('Unknown')
plt.show()

# 运用kmeans算法在进行聚类
data = data[[x_axis, y_axis]] # 只保留x和y列作为特征数据集（假设只使用两个特征）
data = data.values # 将Pandas DataFrame转换为NumPy数组以便进行数学运算
K = 3 # 设置聚类数量（这里设置为3个簇）
max_train = 500 # 设置最大迭代次数（这里设置为500次）
cluster, centroids, K = KMeans_train(data, K, max_train)
color=['r','b','g']
for i in range(K):
    index=np.where(cluster==i)[0]
    plt.scatter((data[index])[:,0],(data[index])[:,1],c=color[i])
plt.show()
