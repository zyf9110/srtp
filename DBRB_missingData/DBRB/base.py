import numpy as np
# =============================================================================
# Base variable
# =============================================================================


EPS = 1e-6


# =============================================================================
# Base function
# =============================================================================


def calc_references_match_degree(x, A):
    """
    :param x: 输入一维数据
    :param A: 条件参数集，[n_antecedents, n_references]
    :return: alpha：输出对所有规则条件属性的匹配度
    """
    similar = []
    for j in range(len(A)):
        similar.append(calc_match_degree(A[j], x[j]))
    return similar


def calc_match_degree(references, numeric):
    """
    Parameters
    ----------
    references : list
        attributes' reference value，order by asc

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


def calc_rule_match_degree(cal_method, A, rules, alpha):
    """

    :param cal_method: 计算前件匹配度的方法
    :param A: 前提属性参考值
    :param rules:
    :param alpha:
    :return: match_degree: array-like:
    """
    match_degree = []
    for k in range(len(rules)):
        match_degree.append(cal_method(A, rules[k], alpha))  # 计算输入数据x与第k条规则的匹配度
    return match_degree


def calc_active_weights(similarities, rules, n_features, sigma):
    """
    Parameters
    ----------
    similarities : list(float)，
        similarities[k][j]表示第k条规则第j个属性与输入的相似度（匹配度）

    rules : list(Rule)
        置信规则库

    n_features : int
        属性个数

    sigma : list(float) 条件属性的权重

    Returns
    -------
    active_weights : list(float)
       规则激活权重.
       如果零激活，则终止程序
    """
    active_weights = [0] * len(rules)
    active_weight_sum = 0.0
    for k in range(len(rules)):
        alpha = 0
        for j in range(n_features):  # alpha_i = alpha_{i-1} + (1 - alpha_{i-1}) * sigma_i * matchDegree_i
            if j == 0:
                alpha = (sigma[j] / np.max(sigma)) * similarities[k][j]
            else:
                alpha = alpha + (1 - alpha) * (sigma[j] / np.max(sigma)) * similarities[k][j]
        active_weights[k] = alpha * rules[k].theta
        active_weight_sum += active_weights[k]

    if active_weight_sum < EPS:
        return None
    for k in range(len(rules)):
        active_weights[k] /= active_weight_sum

    return active_weights


def generate_antecedent(A):
    """
    生成规则库前件
    :param A:
    :return:
    """
    return line_combination(A)


def line_combination(A, j_=0):
    """
    线性组合属性值
    :param j_:
    :param A:
    :return:
    """
    antecedent = []
    flag = False
    for i in range(len(A)):
        if j_ >= len(A[i]):
            antecedent.append(A[i][-1])
        else:
            flag = True
            antecedent.append(A[i][j_])
    if flag:
        pre_rules = line_combination(A, j_ + 1)
        pre_rules.append(antecedent)
        return pre_rules
    else:
        return []


def evidence_reasoning(active_weights, rules, D):
    """
    Parameters
    ----------
    :param active_weights:
    :param rules:
    :param D:

    Returns
    -------
    beta : list数组
       结果置信度.

    """
    rules_copy = []
    active_weights_copy = []
    for i in range(len(active_weights)):
        if active_weights[i] > EPS:
            rules_copy.append(rules[i])
            active_weights_copy.append(active_weights[i])
    rules = rules_copy
    active_weights = active_weights_copy

    beta = [0] * len(D)
    if len(active_weights) == 0:
        return beta  # 没有激活规则
    # 计算beta
    fir_para = [1.0] * len(D)  # w(k)*beta(n,k)+1-w(k)
    fir_para_sum = 0.0

    beta_sum = []
    weight_pro = 1
    for k in range(len(rules)):
        beta_sum.append(np.sum(rules[k].beta))
        weight_pro *= (1 - active_weights[k])

    for j in range(len(D)):
        for k in range(len(rules)):
            fir_para[j] *= (active_weights[k] * rules[k].beta[j] + 1 - active_weights[k] * beta_sum[k])
        fir_para_sum += fir_para[j]

    sec_para = 1.0  # sigma(1-w(k))
    for k in range(len(rules)):
        sec_para *= (1 - active_weights[k] * beta_sum[k])

    for j in range(len(D)):
        beta[j] = (fir_para[j] - sec_para) / (fir_para_sum - (len(D) - 1) * sec_para - weight_pro)

    return beta
