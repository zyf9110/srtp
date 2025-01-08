from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
import math
import numpy as np
import pandas as pd
import time

from .base import calc_similars, generate_extend_rules2
from .base import calc_active_weights
from .base import evidence_reasoning
from .base import adjust_theta
from .base import transform_to_belief
from .base import generate_extend_rules
from .base import adjust_theta2
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# =============================================================================
# Base variable
# =============================================================================
EPS = 1e-6


# =============================================================================
# Base function
# =============================================================================
def similarity_func(belief_i, belief_k, band=1.0):
    """
    Parameters
    belief_i: list(float)
        置信分布

    belief_k: list(float)
        置信分布

    band:
        兼容带参数的计算相似度公式
    Returns
    -------
    similar : float
        返回两个置信分布的相似度.
    """
    dis = 0
    for i in range(len(belief_i)):
        dis += (belief_i[i] - belief_k[i]) ** 2

    similar = max(0.0, 1 - math.sqrt(dis))
    return similar


def entropy_weight(X):
    X_copy = np.copy(X)
    n_samples = X_copy.shape[0]
    n_features = X_copy.shape[1]
    # step 1:将各属性标准化
    for j in range(n_features):
        min = np.nanmin(X_copy[:, j])
        max = np.nanmax(X_copy[:, j])
        for i in range(n_samples):
            X_copy[i][j] = (X_copy[i][j] - min) / (max - min)
    # step 2：计算第i个样本的第j个属性在该属性值中占的比重
    for j in range(n_features):
        sum = np.sum(X_copy[:, j])
        for i in range(n_samples):
            X_copy[i][j] = X_copy[i][j] / sum
    # step 3:计算熵值
    e = []
    for j in range(n_features):
        sum = 0
        for i in range(n_samples):
            if X[i][j] == 0:
                continue
            sum += X[i][j] * np.log(X_copy[i][j])
        e_j = -1 * 1 / np.log(n_samples) * sum
        e.append(e_j)
    # step 4:计算权重
    tmp = n_features - np.sum(e)
    w = [(1 - value) / tmp for value in e]
    return w


# =============================================================================
# estimator
# =============================================================================
class EDBRBBase(BaseEstimator):
    def __init__(self, A, D, sigma=None):
        """
        Parameters
        A: list(float),二维
            属性参考值

        D: list(float),一维
            结果评价等级

        sigma: list(float),一维
            属性权重
        """
        self.A = A
        self.D = D
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = [1.0] * len(A)
        self.rules = None
        self.average_process_time = None

    def fit(self, X, y, is_class):
        """Build a decision tree regressor from the training set (X, y).
        Parameters
        ----------
        X : array-like
            输入数据, X[i][j]表示第i个数据第j个特征值

        y : array-like, shape = [n_samples]
            数据对应的标签

        Returns
        -------
        self : object
            Returns self.
        """
        self.sigma = entropy_weight(X)  # 获取前提属性权重

        # 划分数据集
        cpl_X = []  # 完整数据
        cpl_y = []
        incpl_X = []  # 不完整数据
        incpl_y = []
        for i in range(len(X)):
            flag = False
            for c in X[i]:
                if pd.isnull(c):
                    incpl_X.append(X[i])
                    incpl_y.append(y[i])
                    flag = True
                    break
            if not flag:
                cpl_X.append(X[i])
                cpl_y.append(y[i])
        # 构建完整规则
        rules = generate_extend_rules(cpl_X, cpl_y, self.A, self.D, is_class)
        rules = adjust_theta(similarity_func, rules, len(self.A))

        # 构建不完整规则
        rules2 = generate_extend_rules(incpl_X, incpl_y, self.A, self.D, is_class)
        rules2 = adjust_theta2(similarity_func, rules2, rules, len(self.A), self.sigma)

        self.rules = np.concatenate((rules, rules2))
        return self

    def predict(self, X, is_class, real_y=None):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like
            输入数据, X[i][j]表示第i个数据第j个特征值

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        n_samples = np.shape(X)[0]
        n_features = np.shape(X)[1]
        y = np.zeros(n_samples)

        time_start = time.time()
        for i in range(n_samples):
            alpha = transform_to_belief(X[i], self.A)
            similars = calc_similars(similarity_func, self.rules, n_features, alpha)
            active_weights = calc_active_weights(similars, self.rules, n_features, self.sigma)
            if active_weights is None:
                print("出现零激活！")
                # y[i] = real_y[i]
                continue
            beta = evidence_reasoning(active_weights, self.rules, self.D)
            if not is_class:
                y[i] = 0
                for j in range(len(self.D)):
                    y[i] += beta[j] * self.D[j]
            else:
                y[i] = np.argmax(beta)
        time_end = time.time()
        if n_samples > 0:
            self.average_process_time = (time_end - time_start) / n_samples
        return y


class EDBRBRegressor(EDBRBBase, RegressorMixin):
    def __init__(self, A, D, sigma=None):
        super().__init__(A, D, sigma)

    def fit(self, X, y):
        super().fit(X, y, is_class=False)
        return self

    def predict(self, X, real_y=None):
        return super().predict(X, is_class=False, real_y=real_y)


class EDBRBClassifier(EDBRBBase, ClassifierMixin):
    def __init__(self, A, D, sigma=None):
        super().__init__(A, D, sigma)

    def fit(self, X, y):
        super().fit(X, y, is_class=True)
        return self

    def predict(self, X, real_y=None):
        return super().predict(X, is_class=True, real_y=real_y)
