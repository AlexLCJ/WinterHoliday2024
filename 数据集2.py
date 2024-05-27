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


"""# 数据处理
column_names=['X','Y','class']
data = pd.read_csv('C:\\Users\\李昌峻\\Desktop\\wdbc.csv')
# 引入名字列表

x_axis = 'X'  # x轴
y_axis = 'Y'  # y轴

# 数据对比，通过已知的标签和kmeans计算无标签的数据比较

# 无标签
for i in column_names:
    plt.scatter(data[x_axis][:],data[y_axis][:])
plt.title('label Unknown')
plt.show()

# Kmeans计算
data = data[[x_axis, y_axis]] # 只保留x和y列作为特征数据集（假设只使用两个特征）
data = data.values # 将Pandas DataFrame转换为NumPy数组以便进行数学运算
# 参数设置
K = 3 # 设置聚类数量
max_train = 900 # 设置最大迭代次数

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
plt.show()"""


data=pd.read_csv('C:\\Users\\李昌峻\\Desktop\\abalone.csv')
print(data.shape)

# (568, 32) 32个特征值 需要进行降维到2纬
# 数据中带有字符串类型，对这一列进行删除，只保留数字

# 读取CSV文件
df = pd.read_csv('C:\\Users\\李昌峻\\Desktop\\abalone.csv')

# 删除带有字母的列
# 保存修改后的数据到新的CSV文件
df.to_csv('C:\\Users\\李昌峻\\Desktop\\abalone.csv', index=False)


# 删除非数值型特征
data = data.select_dtypes(include=['float64', 'int64'])

from sklearn.manifold import TSNE
tsne=TSNE()
data=tsne.fit_transform(data)
print(data.shape)
# 执行 t-SNE
data = tsne.fit_transform(data)
# (568, 2) 此时降维到2纬
max_train=500



def KMeans_train_with_K(data, max_K, max_train):
    distortions = []  # 代替SSE，用于存储每个 K 对应的畸变程度

    for K in range(1, max_K + 1):
        initial_centroids = random_init_centroids(data, K)
        for i in range(max_train):
            cluster = classify_cluster(data, initial_centroids, K)
            initial_centroids = new_centroids(data, K, cluster, initial_centroids)

        # 计算畸变程度并存储
        distortion = calculate_distortion(data, cluster, initial_centroids)
        distortions.append(distortion)

    # 使用肘部法找到最佳 K 值
    optimal_K = find_optimal_K(distortions)

    # 返回最佳 K 对应的结果
    return cluster, initial_centroids, optimal_K,distortions



def calculate_distortion(data, cluster, centroids):
    distortion = 0
    for i in range(len(data)):
        distortion += np.linalg.norm(data[i] - centroids[cluster[i]])**2
    return distortion

#传入一组的数
def find_optimal_K(distortions):
    # 使用肘部法找到最佳 K 值
    # 计算每相邻两个 K 对应的畸变程度变化率
    # distortions_changes作为一个数组储存
    distortions_changes = [distortions[i] - distortions[i + 1] for i in range(len(distortions) - 1)]

    # 找到肘部，即畸变程度变化率开始减缓的位置
    optimal_K_index = distortions_changes.index(max(distortions_changes))

    # 最佳 K 值为肘部对应的 K 值加1
    optimal_K = optimal_K_index + 2  # 加1是因为索引从0开始，K从1开始

    return optimal_K


# 调用 KMeans_train_with_K 函数
cluster_result, centroids_result, optimal_K_value, SEE = KMeans_train_with_K(data, max_K=10, max_train=100)

# 打印最优的 K 值
print("Best K:", optimal_K_value)
K_values = range(1, len(SEE) + 1)
plt.plot(K_values, SEE, marker='o',color='green')
plt.title('K')
plt.xlabel('Number of K')
plt.ylabel('SSE')

plt.show()

