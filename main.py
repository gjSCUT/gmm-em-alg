# -*- coding: utf-8 -*-
# ----------------------------------------------------
# Copyright (c) 2017, Wray Zheng. All Rights Reserved.
# Distributed under the BSD License.
# ----------------------------------------------------

import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from gmm import *

# 设置调试模式
DEBUG = True

# 载入数据
data1 = sio.loadmat("data/1")
data2 = sio.loadmat("data/2")
data3 = sio.loadmat("data/3")
Y = np.hstack((data1['x'], data1['y'])) + np.hstack((data2['x'], data2['y'])) + np.hstack((data3['x'], data3['y']))
print(Y)
matY = np.matrix(Y, copy=True)

# 模型个数，即聚类的类别个数
K = 2

# 计算 GMM 模型参数
mu, cov, alpha = GMM_EM(matY, K, 100)

# 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
N = Y.shape[0]
# 求当前模型参数下，各模型对样本的响应度矩阵
prob = getExpectation(matY, mu, cov, alpha)
gamma = getGamma(matY, prob, alpha)
# 对每个样本，求响应度最大的模型下标，作为其类别标识
category = gamma.argmax(axis=1).flatten().tolist()[0]
# 将每个样本放入对应类别的列表中
class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
class2 = np.array([Y[i] for i in range(N) if category[i] == 1])
prob1 = np.array([np.asarray(prob[i])[0, 0] for i in range(N) if category[i] == 0])
prob2 = np.array([np.asarray(prob[i])[0, 1] for i in range(N) if category[i] == 1])

# 绘制聚类结果
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(class1[:, 0], class1[:, 1], prob1, marker='o', c='red')
ax.scatter(class2[:, 0], class2[:, 1], prob2, marker='o', c='blue')

plt.title("GMM Clustering By EM Algorithm")
plt.show()
