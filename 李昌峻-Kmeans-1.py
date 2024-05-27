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