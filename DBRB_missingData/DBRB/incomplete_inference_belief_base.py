import numpy as np
import math
import pandas as pd
from DBRB.base import EPS
from DBRB.iidbrb import transform_to_belief


def calc_active_weight(X, missing_idx):
    """
    计算激活权重
    :return:
    """
    incons = []
    incons_sum = 0
    for i in range(len(X)):
        x = X[i]
        incons_j = 0
        for j in range(len(X)):
            if j == i:
                continue
            sra = 1
            src = 1
            x2 = X[j]
            for k in range(len(x)):
                dic = 1 / (1 + np.sqrt(np.sum((x[k] - x2[k]) ** 2)))
                if k in missing_idx:
                    src = src if src < dic else dic
                else:
                    sra = sra if sra < dic else dic
            incons_j += 1 - np.exp(-1 * ((1+sra) / (1+src) - 1) ** 2 * src ** 2)
        incons.append(incons_j)
        incons_sum += incons_j
    if incons_sum == 0:
        res = [1] * len(incons)
    else:
        res = [1 - value / incons_sum for value in incons]
    return np.array(res)


class IIBB():
    def __init__(self, X, A):
        """
        :param A: [前提属性，参考值]
        """
        self.base = None
        self.A = A
        self.generate_belief(X)

    def generate_belief(self, X):
        """
        :param data: 完整的数据集
        :return:
        """
        res = []
        for k in range(np.shape(X)[0]):
            match_set = transform_to_belief(X[k], self.A)
            res.append(match_set)
        self.base = np.array(res)
        return self.base

    def resolve_incomplete(self, incomplete_data, missing_antecedent, delta=0.1):
        """
        处理缺失数据
        :param delta:
        :param missing_antecedent: 缺失数据的索引
        :param incomplete_data:
        :return:
        """
        # 检索具有相同结构的置信度
        similar = []  # 存放具有相似结构的数据
        for j in range(np.shape(self.base)[0]):
            flag = True
            for k in range(np.shape(self.base)[1]):
                if k in missing_antecedent:
                    continue
                for n in range(np.shape(self.base)[2]):
                    if incomplete_data[k][n] != 0 and self.base[j][k][n] == 0 or incomplete_data[k][n] \
                            == 0 and self.base[j][k][n] != 0:
                        flag = False
                        break
                if not flag:
                    break
            if flag:
                similar.append(self.base[j])
        similar = np.array(similar)
        incomplete_data = np.array(incomplete_data)
        #计算权重
        weights = calc_active_weight(similar, missing_antecedent)
        # 计算激活程度
        pro_sum = 0
        res = []
        for k in range(len(similar)):
            pro = 1
            for n in range(len(similar[k])):
                if n in missing_antecedent:
                    continue
                pro *= np.exp(-1 * np.sum((incomplete_data[n] - similar[k][n]) ** 2) / (2 * delta ** 2))
            res.append(weights[k] * pro)
            pro_sum += pro
        res = np.array(res) / pro_sum
        # 填充缺失值
        for j in missing_antecedent:
            if len(similar) == 0:
                incomplete_data[j] = np.zeros(np.shape(incomplete_data)[1])
                continue
            incomplete_data[j] = np.sum(similar[:, j] * np.array([[value] * np.shape(similar)[2] for value in res]), 0)
        return incomplete_data

    def resolve(self, X):
        res = []
        for i in range(len(X)):
            incomplete_data = np.array(transform_to_belief(X[i], self.A))
            idx = []
            tmp = incomplete_data[:, 0]
            for j in range(len(tmp)):
                if pd.isnull(tmp[j]):
                    idx.append(j)
            res.append(self.resolve_incomplete(incomplete_data, idx))
        return res
