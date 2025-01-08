import numpy as np
import pandas as pd
import copy
import math


# =============================================================================
# Base variable
# =============================================================================
EPS = 1e-6


# =============================================================================
# Base structure
# =============================================================================
# 置信规则
class Rule:
    def __init__(self, A, D):
        # 初始化
        self.theta = 1  # 规则权重
        self.alpha = []
        for i in range(len(A)):
            self.alpha.append([0]*len(A[i]))  # 前置属性置信度
        self.beta = [0]*len(D)  # 结果置信度


# =============================================================================
# Base function
# =============================================================================


def calc_belief(references, numeric):
    """
    Parameters
    ----------
    references : list
        attributes' reference value

    numeric : float
        numeric input's value

    Returns
    -------
    result : list
       believes.
    """
    result = [0] * len(references)
    for j in range(len(references) - 1):
        if references[j + 1] >= numeric:
            result[j] = (references[j + 1] - numeric) / (references[j + 1] - references[j])
            result[j + 1] = 1 - result[j]
            break
    return result


def transform_to_belief(x, A):
    alpha = []
    for j in range(len(A)):
        alpha.append(calc_belief(A[j], x[j]))
    return alpha


def adjust_theta(simi_func, rules, n_features, band=1):
    incons_sum = 0
    incons = [0] * len(rules)
    for i in range(len(rules)):
        for j in range(len(rules)):
            if i == j:
                continue
            sra = 1
            for k in range(n_features):
                sra_temp = simi_func(rules[i].alpha[k], rules[j].alpha[k], band)
                sra = min(sra, sra_temp)
            src = simi_func(rules[i].beta, rules[j].beta, band)
            if sra <= EPS or src <= EPS:  # 这里没考虑到src<=EPS, 后期需要调整
                cons = 1
            else:
                cons = math.exp(-((sra / src - 1) ** 2) / ((1 / sra) ** 2))
            incons[i] += 1 - cons
        incons_sum += incons[i]

    for i in range(len(rules)):
        if incons_sum > EPS:
            rules[i].theta = 1 - incons[i] / incons_sum
        else:
            rules[i].theta = 1

    return rules


def generate_extend_rules(X, y, A, D, is_class=True):
    """
    Parameters
    ----------
    X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

    y : array-like, shape = [n_samples] or [n_samples, n_outputs]
        The target values (real numbers). Use ``dtype=np.float64`` and
        ``order='C'`` for maximum efficiency.

    is_class: bool
        classification task or regression task
        Returns

    -------
    result : rules.
        规则库
    """
    rules = []
    for i in range(np.shape(X)[0]):
        rule = Rule(A,D)
        for j in range(len(A)):
            rule.alpha[j] = calc_belief(A[j], X[i][j])
        if is_class:
            rule.beta[y[i]] = 1.0
        else:
            rule.beta = calc_belief(D, y[i])
        rules.append(rule)
    return rules


def calc_similars(simi_func, rules, n_features, alpha, band=1.0):
    similars = []
    for k in range(len(rules)):
        similars.append([])
        for j in range(n_features):
            similars[k].append(simi_func(rules[k].alpha[j], alpha[j], band=band))
    return similars


def calc_active_weights(similarities, rules, n_features, sigma):
    """
    Parameters
    ----------
    similarities : list(float)，
        similarities[k][j]表示第k条规则第j个属性与输入的相似度

    rules : list(Rule)
        置信规则库

    attr_num : int
        属性个数

    sigma : list(float)

    Returns
    -------
    active_weights : list(float)
       规则激活权重.
       如果零激活，则返回None
    """
    active_weights = [0] * len(rules)
    active_weight_sum = 0.0
    for k in range(len(rules)):
        active_weights[k] = rules[k].theta
        for j in range(n_features):
            active_weights[k] *= (similarities[k][j] ** sigma[j])
        active_weight_sum += active_weights[k]

    if active_weight_sum < EPS:
        return None
    for k in range(len(rules)):
        active_weights[k] /= active_weight_sum

    return active_weights


def evidence_reasoning(active_weights, rules, D):
    """
    Parameters
    ----------
    active_weights : list数组，
        规则激活权重

    rules : list数组
        置信规则库

    Returns
    -------
    beta : list数组
       结果置信度.
    param D:
    """
    rules_copy = []
    active_weights_copy = []
    for i in range(len(active_weights)):
        if active_weights[i] > EPS:
            rules_copy.append(rules[i])
            active_weights_copy.append(active_weights[i])
    rules = rules_copy
    active_weights = active_weights_copy

    # 计算beta
    fir_para = [1.0] * len(D)  # w(k)*beta(n,k)+1-w(k)
    fir_para_sum = 0.0
    for j in range(len(D)):
        for k in range(len(rules)):
            fir_para[j] *= (active_weights[k] * rules[k].beta[j] + 1 - active_weights[k])
        fir_para_sum += fir_para[j]

    sec_para = 1.0  # sigma(1-w(k))
    for k in range(len(rules)):
        sec_para *= (1 - active_weights[k])

    beta = [0] * len(D)
    for j in range(len(D)):
        beta[j] = (fir_para[j] - sec_para) / (fir_para_sum - len(D) * sec_para)

    return beta
