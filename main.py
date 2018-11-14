import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

TIME = 100
K = 2
paths = ["data/1", "data/2", "data/3"]


def main():
    data = init_data(paths[index])
    mu, cov, alpha = init_params(data)
    p, w = gmm_em(data, mu, cov, alpha)
    for i in range(TIME):
        p, w = gmm_em(data, mu, cov, alpha)
        mu, cov, alpha = maximize(data, w)
    y = w.argmax(axis=1).flatten().tolist()[0]
    display(data, p, y)


def init_data(path):
    mat = sio.loadmat(path)
    data = np.matrix(np.hstack((mat['x'], mat['y'])), copy=True)
    for i in range(data.shape[1]):
        d_max, d_min = data[:, i].max(), data[:, i].min()
        data[:, i] = (data[:, i] - d_min) / (d_max - d_min)
    return data


def init_params(data):
    v = data.shape[1]
    mu = np.random.rand(K, v)
    cov = np.array([np.eye(v)] * K)
    alpha = np.array([1.0 / K] * K)
    return mu, cov, alpha


def display(data, p, y):
    n = data.shape[0]
    result1 = np.array([[data[i, 0], data[i, 1], np.asarray(p[i])[0, 0]] for i in range(n) if y[i] == 0])
    result2 = np.array([[data[i, 0], data[i, 1], np.asarray(p[i])[0, 1]] for i in range(n) if y[i] == 1])

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(result1[:, 0], result1[:, 1], result1[:, 2], marker='o', c='blue')
    ax.scatter(result2[:, 0], result2[:, 1], result2[:, 2], marker='o', c='red')
    plt.title("GMM EM Algorithm")
    plt.show()


def gmm_em(data, mu, cov, alpha):
    n = data.shape[0]

    p = np.zeros((n, K))
    for k in range(K):
        p[:, k] = multivariate_normal(mean=mu[k], cov=cov[k]).pdf(data)
    p = np.mat(p)

    w = np.mat(np.zeros((n, K)))
    for k in range(K):
        w[:, k] = alpha[k] * p[:, k]
    for i in range(n):
        w[i, :] /= np.sum(w[i, :])

    return p, w


def maximize(data, w):
    n, v = data.shape
    mu = np.zeros((K, v))
    cov = []
    alpha = np.zeros(K)
    for k in range(K):
        nk = np.sum(w[:, k])
        for d in range(v):
            mu[k, d] = np.sum(np.multiply(w[:, k], data[:, d])) / nk
        cov_k = np.mat(np.zeros((v, v)))
        for i in range(n):
            cov_k += w[i, k] * (data[i] - mu[k]).T * (data[i] - mu[k]) / nk
        cov.append(cov_k)
        alpha[k] = nk / v
    cov = np.array(cov)
    return mu, cov, alpha


for index in range(len(paths)):
    main()


