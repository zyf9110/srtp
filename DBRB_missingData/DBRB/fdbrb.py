import time

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from DBRB.base import *
from DBRB.rule import Rule
from DBRB.f_r_differential_evolution import run_DE_algorithm
import numpy as np


def similar_function(A, rule, alpha):
    """
    第j个输入对第k条规则的相似度
    :param A: 前提属性参考值
    :param rule:
    :param alpha: 输入数据的置信分布
    :return:
    """
    similarities = []
    for i in range(len(A)):
        for j in range(len(A[i])):
            if rule.condition[i] == A[i][j]:
                similarities.append(alpha[i][j])
                break
            # if j == len(A[i])-1:
            #     print(j)
            #     print(len(A[i]))
            #     raise RuntimeError('value don\'t match any references (#属性值没有匹配任何的属性候选值)')

    return similarities


def train_params_func(dbrb_model, X, y):
    """
    参数训练方法
    :param dbrb_model:
    :param X:
    :param y:
    :return:
    """
    return run_DE_algorithm(dbrb_model, X, y, 200, 100)


def transform_to_belief(X, A, delta):
    """
    将输入数据转化为置信表示
    :param delta: 一维数据，各个属性的平均值
    :param X: 一维数据
    :param A:
    :return:
    """
    match_set = []
    for i in range(np.shape(A)[0]):
        match_degree = [0] * np.shape(A)[1]
        if X[i] is None:
            match_set.append([np.nan] * np.shape(A)[1])  # 生成参考属性匹配度都是nan的list
            continue
        sum = 0
        for j in range(np.shape(A)[1]):
            match_degree[j] = np.exp(-1 * (X[i] - A[i][j])**2 / (2 * delta[i]**2))  # 高斯隶属函数
            sum += match_degree[j]

        match_set.append([value / sum for value in match_degree])
    return match_set


class DBRBBase(BaseEstimator):
    def __init__(self, A, D, sigma=None, is_classify=False):
        """
        :param A: 属性参考值
        :param D: 评价结果等级
        :param sigma: 属性权重
        """
        self.A = A
        self.D = D
        if sigma is None:
            self.sigma = [1.0] * len(A)
        else:
            self.sigma = sigma
        self.rules = None
        self.average_process_time = None
        self.is_classify = is_classify
        self.delta = None  # 用于高斯隶属函数中的delta--样本的均值

    def set_params(self, params):
        """
        设置参数
        :param params: list: n_sigma属性权重 + k_theta规则权重 + m_beta规则后件置信度 + t_utility规则前件效用
        :return:
        """
        if len(params) < 1:
            raise RuntimeError("没有传入参数到BRB中")
        else:
            params = np.array(params)
            index = 0

            # 属性权重
            self.sigma = []
            for N in range(len(self.A)):
                self.sigma.append(params[index])
                index += 1

            # 规则权重
            for k in range(len(self.rules)):
                rule = self.rules[k]
                rule.theta = params[index]
                index += 1

            # 规则效用
            utility = np.zeros((np.shape(self.A)[0], np.shape(self.A)[1]))
            for t in range(np.shape(self.A)[1]):
                for i in range(np.shape(self.A)[0]):
                    utility[i][t] = params[index]
                    index += 1
            for t in range(len(utility)):
                self.A[t] = np.sort(utility[t])
            # 更新规则库
            rules = []
            for k in range(len(self.rules)):
                rule = Rule(self.A, self.D)
                rule.condition = utility[:, k]
                rules.append(rule)
            self.rules = rules

            # beta
            for M in range(len(self.rules)):
                rule = self.rules[M]
                rule.beta = params[index: index + len(self.D)]
                index += len(self.D)

    def get_params_num(self):
        """
        获取参数个数：属性权重 +  规则权重 + 规则后件置信度 + 规则前件效用
        :return:
        """
        return len(self.A) + len(self.rules) + len(self.D) * len(self.rules) + len(self.A) * len(self.rules)

    def fit(self, X, y):
        """
        :param X: array-like: [n_simples, n_features]
        :param y: list: [n_simples]
        :return:
        """
        pre_rules = generate_antecedent(self.A)
        rules = []
        for k in range(len(pre_rules)):
            rule = Rule(self.A, self.D)
            rule.condition = pre_rules[k]
            rules.append(rule)
        self.rules = rules
        self.delta = np.mean(X, 0)  # 计算样本各个属性的均值

        train_params_func(self, X, y)  # 参数训练
        return self

    def predict(self, X):
        """
        训练后调用该方法进行预测
        :param X:
        :param is_classify: true:分类； False:回归
        :return:
        """
        n_samples = np.shape(X)[0]
        n_features = np.shape(X)[1]
        y = np.zeros(n_samples)

        time_start = time.time()

        for i in range(n_samples):
            alpha = transform_to_belief(X[i], self.A, self.delta)
            similar = calc_rule_match_degree(similar_function, self.A, self.rules, alpha)
            active_weight = calc_active_weights(similar, self.rules, n_features, self.sigma)
            if active_weight is None:
                print("出现零激活")
                continue
            beta = evidence_reasoning(active_weight, self.rules, self.D)

            if not self.is_classify:
                y[i] = 0
                for j in range(len(beta)):
                    y[i] += self.D[j] * beta[j]

            else:
                y[i] = self.D[np.argmax(beta)]

        time_end = time.time()
        if n_samples > 0:
            self.average_process_time = (time_end - time_start) / n_samples
        return y


class FDBRBRegressor(DBRBBase, RegressorMixin):
    def __init__(self, A, D, sigma=None):
        super().__init__(A, D, sigma)

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def predict(self, X):
        return super().predict(X)


class FDBRBClassifier(DBRBBase, ClassifierMixin):
    def __init__(self, A, D, sigma=None):
        super().__init__(A, D, sigma, True)

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def classify(self, X):
        return super().predict(X)
