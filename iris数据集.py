import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 随机初始化数据中心
def random_init_centroids(data, K):
    num = data.shape[0]
    parts = np.random.permutation(num)
    centroids = data[parts[:K], :]
    return centroids


# 获得欧氏距离
def get_distance(data, centroids, K):
    distance = np.zeros((len(data), K))  # 初始化距离矩阵
    for i in range(K):
        distance[:, i] = np.sqrt(np.sum(np.square(data - centroids[i]), axis=1))  # 计算每个点到每个质心的距离
    min_distance = np.argmin(distance, axis=1)  # 找到最小距离的质心索引
    return distance, min_distance


# 样本分类
def classify_cluster(data, centroids, K):
    cluster = np.zeros(len(data))  # 初始化聚类标签数组
    distance, min_distance = get_distance(data, centroids, K)  # 计算距离和最小距离
    cluster = min_distance  # 将最小距离的质心索引赋值给聚类标签数组
    return cluster


# 重新计算中心
def new_centroids(data, K, cluster, centroids):
    for j in range(K):
        index = (np.where(cluster == j))[0]  # 获取簇中所有数据点的索引
        # 总和除以个数得到均值（样本中心）
        centroids[j] = np.sum(data[index], axis=0) / len(index)  # 计算新的质心坐标
    return centroids


# K-means算法主要函数
def KMeans_train(data, K, max_train):  # max_train 最大迭代训练次数
    # 随机初始化质心坐标
    initial_centroids = random_init_centroids(data, K)
    for i in range(max_train):
        cluster = classify_cluster(data, initial_centroids, K)  # 分类和计算距离
        initial_centroids = new_centroids(data, K, cluster, initial_centroids)  # 重新计算质心坐标
    return cluster, initial_centroids, K


# 数据处理
column_names=['X','Y','class']
data = pd.read_csv('C:\\Users\\李昌峻\\Desktop\\iris.csv',names=column_names)
# 引入名字列表
iris_name=['Iris-setosa','Iris-versicolor','Iris-virginica']
x_axis = 'X'  # x轴
y_axis = 'Y'  # y轴

# 数据对比，通过已知的标签和kmeans计算无标签的数据比较
# 有标签：
for i in iris_name:
    plt.scatter(data[x_axis][data['class']==i],data[y_axis][data['class']==i],label=iris_name)
plt.title('label known')
plt.show()

# 无标签
for i in iris_name:
    plt.scatter(data[x_axis][:],data[y_axis][:])
plt.title('label Unknown')
plt.show()

# Kmeans计算
data = data[[x_axis, y_axis]] # 只保留x和y列作为特征数据集（假设只使用两个特征）
data = data.values # 将Pandas DataFrame转换为NumPy数组以便进行数学运算
# 参数设置
K = 3 # 设置聚类数量（单独的算法，iris中已知标签数目，定义位3）
max_train = 500 # 设置最大迭代次数

# 训练k-means
cluster, centroids, K = KMeans_train(data, K, max_train)
color=['r','b','g'] # 绘制不同簇的颜色
for i in range(K):
    index=np.where(cluster==i)[0]
    #
    plt.scatter((data[index])[:,0],(data[index])[:,1],c=color[i])
    # 画出centroids的位置
    plt.scatter(centroids[:, 0], centroids[:, 1], c='purple', marker='x')
    plt.title('KMeans')
plt.show()