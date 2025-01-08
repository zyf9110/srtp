import numpy as np
import math
import pandas as pd
from DBRB.base import EPS
from DBRB.iidbrb import transform_to_belief


class IIBB2():
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

    def calc_weight(self, missing_idx):
        """
        计算权重
        :return:
        """
        incons = []
        incons_sum = 0
        for i in range(len(self.base)):
            x = self.base[i]
            incons_j = 0
            for j in range(len(self.base)):
                if j == i:
                    continue
                x2 = self.base[j]
                sra = 1
                src = 1
                for k in range(len(x)):
                    dic = 1 - np.sqrt(np.sum((x[k] - x2[k]) ** 2))
                    if k in missing_idx:
                        src = src if src < dic else dic
                    else:
                        sra = sra if sra < dic else dic
                incons_j += 1 - np.exp(-1 * (sra / src - 1) ** 2 / (1 / sra ** 2))
            incons.append(incons_j)
            incons_sum += incons_j
        res = [1 - value / incons_sum for value in incons]
        return np.array(res)

    def fusion(self, active_weights, missing_idx, incomplete):
        """
        合成缺失数据信息
        :param active_weights: 激活权重
        :param missing_idx: 一维
        :param incomplete: 一维
        :return:
        """
        for idx in missing_idx:
            fir_para = [1.0] * np.shape(self.base)[2]  # w(k)*beta(n,k)+1-w(k)
            fir_para_sum = 0.0
            for j in range(np.shape(self.base)[2]):
                for k in range(len(self.base)):
                    fir_para[j] *= (active_weights[k] * self.base[k][idx][j] + 1 - active_weights[k])  # 信息完备
                fir_para_sum += fir_para[j]

            sec_para = 1.0  # sigma(1-w(k))
            for k in range(len(self.base)):
                sec_para *= (1 - active_weights[k])

            beta = [0] * np.shape(self.base)[2]
            for j in range(np.shape(self.base)[2]):
                beta[j] = (fir_para[j] - sec_para) / (fir_para_sum - np.shape(self.base)[2] * sec_para)
            incomplete[idx] = beta
        return incomplete

    def resolve_incomplete(self, incomplete_data, missing_antecedent):
        """
        处理缺失数据
        :param missing_antecedent: 缺失数据的索引
        :param incomplete_data: 一维数据
        :return:
        """
        #计算权重
        weights = self.calc_weight(missing_antecedent)
        # 计算激活程度
        pro_sum = 0
        active_weight = []
        for k in range(len(self.base)):
            pro = 1
            for n in range(len(self.base[k])):
                if n in missing_antecedent:
                    continue
                pro *= (1 - np.sqrt(np.sum((incomplete_data[n] - self.base[k][n]) ** 2)))
            active_weight.append(weights[k] * pro)
            pro_sum += pro
        active_weight = np.array(active_weight) / pro_sum
        # 填充缺失值
        incomplete_data = self.fusion(active_weight, missing_antecedent, incomplete_data)
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
