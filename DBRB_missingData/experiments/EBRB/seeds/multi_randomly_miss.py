import time
import math
import numpy
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import operator
import copy
import csv
from datasets.load_data import load_seeds
from miss_data_method.random_process import random_array
import numpy as np


# writer = csv.writer(open('paper data/thyroid_rule.csv', 'w', newline=''))


class RuleBase:
    def __init__(self):
        self.A = []  # antecedent attribute
        self.D = []  # consequent
        self.rules = []


class AntecedentAttribute:
    def __init__(self, referential_values, delta):
        self.r = referential_values  # referential value
        self.delta = delta


class Rule:
    def __init__(self, rule_base, input_data, output_data, theta=1.0):
        self.alpha = self.get_alpha(rule_base.A, input_data)  # # belief in antecedent
        self.beta = self.get_beta(rule_base.D, output_data)  # belief in consequent
        self.theta = theta # weight of rule
        self.active_weight = 0

    @staticmethod
    def get_alpha(A, input_data):
        t_alpha = []
        for i in range(len(A)):
            beliefs = Rule.calculate_belief(A[i].r, input_data[i])
            t_alpha.append(beliefs)
        return t_alpha

    @staticmethod
    def get_beta(D, output_data):
        t_beta = []
        for D_i in D:
            if D_i == output_data:
                t_beta.append(1.0)
            else:
                t_beta.append(0.0)
        return t_beta

    @staticmethod
    def calculate_belief(A_i, input_i):
        beliefs = [0.0] * len(A_i)
        for j in range(1, len(A_i)):
            if A_i[j - 1] <= input_i <= A_i[j]:
                beliefs[j - 1] = (A_i[j] - input_i) / (A_i[j] - A_i[j - 1])
                beliefs[j] = 1 - beliefs[j - 1]
                break
        return beliefs


def generate_antecedent_attributes(rule_base, x):
    x_split = 5
    for i in range(len(x[0])):
        a = list(numpy.linspace(x[:, i].min(), x[:, i].max(), x_split))
        delta = 1.0
        rule_base.A.append(AntecedentAttribute(a, delta))


def generate_consequent(rule_base, y):
    rule_base.D = list(numpy.unique(y))


def init_rule_base(x, y):
    rule_base = RuleBase()
    generate_antecedent_attributes(rule_base, x)
    generate_consequent(rule_base, y)
    return rule_base


def get_data(get_way=2):
    x, y = [], []
    if get_way == 1:
        x, y = load_seeds()

        # x, y = sklearn.datasets.load_wine(return_X_y=True)
    if get_way == 2:
        data = pd.read_csv("thyroid.csv", sep=',')
        data_values = data.values
        x, y = data_values[:, :-1], data_values[:, -1]
        x = numpy.array(x)
        y = numpy.array(y)
        print(y)
    for i in range(len(x[0]) - 1, -1, -1):
        if x[:, i].min() == x[:, i].max():
            x = numpy.delete(x, i, axis=1)

    # print(x)
    # print(y)
    cnt = 0
    for i in range(len(y) - 1, -1, -1):
        if cnt >= 0:
            break
        if y[i] == 1:
            cnt += 1
            y = numpy.delete(y, i)
            x = numpy.delete(x, i, axis=0)

    # print(len(x[0]))
    # print(y)
    uni_y = numpy.unique(y)
    # print(uni_y)
    count_y = [0] * len(uni_y)
    for yy in y:
        for i in range(len(uni_y)):
            if yy == uni_y[i]:
                count_y[i] += 1
    # print(count_y)
    return x, y


def k_fold(x, y, rule_base, way, k, miss_percent):  # miss_percent 缺失数据率
    skf = sklearn.model_selection.KFold(n_splits=k, shuffle=True)
    skf.get_n_splits(x, y)
    ave_accuracy = 0.0

    for train, test in skf.split(x, y):
        index_list = list(numpy.linspace(0, len(x[0]) - 1, len(x[0])))
        index_list = list(map(int, index_list))

        x_train, x_test = pd.DataFrame(x[train], columns=index_list), pd.DataFrame(x[test], columns=index_list)
        y_train, y_test = pd.DataFrame(y[train], columns=['y']), pd.DataFrame(y[test], columns=['y'])

        df_train = pd.concat([x_train, y_train], axis=1)
        df_test = pd.concat([x_test, y_test], axis=1)

        # 设置缺失数据，并进行填补
        X_copy = numpy.copy(df_train.values[:, :-1])
        X_copy = random_array(X_copy, miss_percent)
        # 均值
        # for c in range(numpy.shape(X_copy)[1]):
        #     mi = numpy.nanmean(X_copy[:, c])
        #     for r in range(numpy.shape(X_copy)[0]):
        #         if pd.isnull(X_copy[r][c]):
        #             X_copy[r][c] = mi
        # 众数、中位数
        for c in range(np.shape(X_copy)[1]):
            tmp = []
            for t in X_copy[:, c]:
                if not pd.isnull(t):
                    tmp.append(t)
        # number = np.median(tmp)
        number = np.argmax(np.bincount(tmp))
        for r in range(np.shape(X_copy)[0]):
            if pd.isnull(X_copy[r][c]):
                X_copy[r][c] = number
        X_copy = pd.DataFrame(X_copy)
        X_copy['y'] = df_train['y']
        # 论文数据，十折训练条目
        # print()
        # cnt = [0] * 3
        # print(df_train)
        # for yy in df_train['y']:
        #     print(yy)
        #     cnt[int(yy) - 1] += 1
        # print(cnt)

        ave_accuracy += run_EBRB(X_copy, df_test, rule_base, way)

    # ave_accuracy /= 10
    # print("ave_acc = " + str(ave_accuracy))
    return ave_accuracy


def run_EBRB(df_train, df_test, rule_base, way):
    rule_base.rules = []
    construct_rule_base(df_train, rule_base)
    return get_accuracy(df_test, rule_base, way)


def get_data_pair(df_data):
    data = []
    for row in range(len(df_data.index)):
        t_data = []
        for column in df_data.columns:
            t_data.append(df_data[column][row])
        data.append([t_data[:-1], t_data[-1]])
    return data


def construct_rule_base(df_train, rule_base):
    data = get_data_pair(df_train)
    for t_data in data:
        input_data, output_data = t_data[0], t_data[1]
        rule_base.rules.append(Rule(rule_base, input_data, output_data))


def get_accuracy(df_test, rule_base, way):
    outputs = []
    v = []
    data = get_data_pair(df_test)
    accuracy = 0.0
    ac = [0] * len(rule_base.D)
    al = [0] * len(rule_base.D)
    for t_data in data:
        input_data, output_data = t_data[0], t_data[1]
        input_alpha = Rule.get_alpha(rule_base.A, input_data)
        calculate_active_weight(input_alpha, rule_base, way[0], way[2])
        isNone = normalization_active_weight(rule_base.rules)
        if isNone:
            print("出现零激活")
            result = -1
        else:
            result = calculate_output(rule_base, way)
        v.append(t_data[1])
        outputs.append(result)

        # 论文数据，少数类样本的各规则权重和合成结果
        '''
        if t_data[1] == rule_base.D[2] and t_data[1] != result:
            print(t_data[1])
            writer.writerow([t_data[1]])
            print(normalization_active_weight2(rule_base.rules))
            print(calculate_output2(rule_base, way))
        '''

        for i in range(len(rule_base.D)):
            if output_data == rule_base.D[i]:
                al[i] += 1
                if result == output_data:
                    accuracy += 1
                    ac[i] += 1

        # if result != 1:
        #     print(result, t_data[1])

    # print(outputs)
    # print(v)
    s_accuracy = sklearn.metrics.accuracy_score(outputs, v)
    f12 = sklearn.metrics.f1_score(v, outputs, average='macro')
    f13 = sklearn.metrics.f1_score(v, outputs, average='weighted')
    accuracy = []
    for i in range(len(rule_base.D)):
        if al[i] == 0:
            accuracy.append(1e5)
        else:
            accuracy.append(ac[i] / al[i])
    accuracy.append(s_accuracy)
    accuracy.append(f12)
    accuracy.append(f13)
    # print(accuracy)
    accuracy = numpy.array(accuracy)
    # accuracy /= len(data)
    # print(accuracy)
    return accuracy


def calculate_output(rule_base, way):
    output = [0.0] * len(rule_base.D)
    if way[1] == 1:
        # output = numpy.array(output)
        acw = 0
        for rule in rule_base.rules:
            if rule.active_weight > acw:
                acw = rule.active_weight
                output = rule.beta
        rule_base.rules.sort(key=operator.attrgetter('active_weight'))
        t_rules = rule_base.rules[(-1 * way[2]):]
        t_rule_base = copy.copy(rule_base)
        t_rule_base.rules = t_rules
        output = er_aggregation(t_rule_base)
    if way[1] == 2:
        output = er_aggregation(rule_base)
    output = numpy.array(output)
    # print(output)
    for i in range(len(output)):
        if output[i] == output.max():
            return rule_base.D[i]

    print(output)


def calculate_output2(rule_base, way):
    output = [0.0] * len(rule_base.D)
    if way[1] == 1:
        # output = numpy.array(output)
        acw = 0
        for rule in rule_base.rules:
            if rule.active_weight > acw:
                acw = rule.active_weight
                output = rule.beta
        rule_base.rules.sort(key=operator.attrgetter('active_weight'))
        t_rules = rule_base.rules[(-1 * way[2]):]
        t_rule_base = copy.copy(rule_base)
        t_rule_base.rules = t_rules
        output = er_aggregation(t_rule_base)
    if way[1] == 2:
        output = er_aggregation(rule_base)
    writer.writerow(output)
    return output


def er_aggregation(rule_base):
    _wkeb = []
    dr = 1
    for k in range(len(rule_base.rules)):
        eb = 0
        for t_beta in rule_base.rules[k].beta:
            eb += t_beta
        _wkeb.append(1 - rule_base.rules[k].active_weight * eb)
        dr *= _wkeb[k]

    dl = 0
    wb_wkebs = []
    for i in range(len(rule_base.D)):
        wb_wkeb = 1.0
        for k in range(len(rule_base.rules)):
            wb_wkeb *= rule_base.rules[k].active_weight * rule_base.rules[k].beta[i] + _wkeb[k]
        wb_wkebs.append(wb_wkeb)
        dl += wb_wkeb

    d = 1 / (dl - (len(rule_base.D) - 1) * dr)

    bd = 1
    for k in range(len(rule_base.rules)):
        bd *= 1 - rule_base.rules[k].active_weight
    bd = 1 - d * bd

    output = []
    for i in range(len(rule_base.D)):
        output.append(d * (wb_wkebs[i] - dr) / bd)
    return numpy.array(output)


def calculate_active_weight(input_alpha, rule_base, way, p):
    for k in range(len(rule_base.rules)):
        matching_degree = calculate_matching_degree(input_alpha, rule_base.rules[k].alpha)
        if way == 1:

            min_matching_degree = numpy.array(matching_degree).min()

            rule_base.rules[k].active_weight = 1.0
            # this operation is temporarily useless
            max_delta = 0.0
            for A_i in rule_base.A:
                max_delta = max(max_delta, A_i.delta)
            # -------------------------------------
            flag = True
            for i in range(len(matching_degree)):
                if matching_degree[i] == 0:
                    break
                if matching_degree[i] == min_matching_degree and flag is True:
                    flag = False
                    continue
                rule_base.rules[k].active_weight *= matching_degree[i] ** (rule_base.A[i].delta / max_delta)
            rule_base.rules[k].active_weight *= rule_base.rules[k].theta

            # print(min_matching_degree)
            # print(rule_base.rules[k].active_weight) 
            # a = 1.0 / rule_base.rules[k].active_weight
            # print(a)
            # print(min_matching_degree)
            # rule_base.rules[k].active_weight = min_matching_degree ** a
            # print(rule_base.rules[k].active_weight)
            # print()
            if rule_base.rules[k].active_weight != 0:
                a = 1.0 / rule_base.rules[k].active_weight
                rule_base.rules[k].active_weight = min_matching_degree ** a
            # if rule_base.rules[k].active_weight == 0:
            #　   rule_base.rules[k].active_weight = 1e-11


        if way == 2:
            rule_base.rules[k].active_weight = 1.0
            # this operation is temporarily useless
            max_delta = 0.0
            for A_i in rule_base.A:
                max_delta = max(max_delta, A_i.delta)
            # -------------------------------------
            for i in range(len(matching_degree)):
                rule_base.rules[k].active_weight *= matching_degree[i] ** (rule_base.A[i].delta / max_delta)
            rule_base.rules[k].active_weight *= rule_base.rules[k].theta
            # if rule_base.rules[k].active_weight == 0:
            #     rule_base.rules[k].active_weight = 1e-11
            rule_base.rules[k].active_weight **= p


def normalization_active_weight(rules):
    sum_weight = 0.0
    for rule in rules:
        sum_weight += rule.active_weight
    if sum_weight == 0:
        return True
    # print(sum_weight)
    for k in range(len(rules)):
        rules[k].active_weight /= sum_weight
    return False


def normalization_active_weight2(rules):
    ret = [0] * 3
    p = 0
    acw = -1
    sum_weight = 0.0
    for rule in rules:
        row = []
        row.append(rule.beta)
        row.append(rule.active_weight)
        writer.writerow(row)
        print(row)
        sum_weight += rule.active_weight
    for k in range(len(rules)):
        rules[k].active_weight /= sum_weight
        t = numpy.array(rules[k].beta)
        for i in range(len(t)):
            if t[i] == t.max():
                ret[i] += rules[k].active_weight
        if rules[k].active_weight > acw:
            acw = rules[k].active_weight
            p = k
    # print(rules[p].beta)
    writer.writerow(ret)
    return ret


def calculate_matching_degree(input_alpha, rule_alpha):
    matching_degree = []
    for i in range(len(input_alpha)):
        distance = calculate_distance(input_alpha[i], rule_alpha[i])
        matching_degree.append(1 - distance)
    return matching_degree


def calculate_distance(input_alpha_i, rule_alpha_i):
    distance = 0.0
    for j in range(len(input_alpha_i)):
        distance += (input_alpha_i[j] - rule_alpha_i[j]) ** 2
    distance = math.sqrt(distance / 2)
    return distance


def main(miss_percent):
    x, y = get_data(1)
    rule_base = init_rule_base(x, y)

    title = ['min + max', 'min + er', 'mul + max', 'mul + er']
    # writer = csv.writer(open('paper data/bupa f1.csv', 'w', newline=''))
    numpy.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    for way1 in range(1, 2):
        for way2 in range(1, 2):
            way = [way1 + 1, way2 + 1, 0]
            if way2 == 1:
                for p in range(1, 2):
                    way[2] = p
                    ave = 0
                    times = 10
                    for i in range(times):
                        tmp = k_fold(x, y, rule_base, way, 20, miss_percent)
                        ave += tmp
                    ave = (ave % 1e5) / (times * 20 - ave // 1e5)
                    ave *= 100
                    # writer.writerow(list(ave))
                    print('miss percent = %d%% ' % (miss_percent * 100))
                    print("total ave_acc = " + str(ave) + "  " + title[way1 * 2 + way2] + ' ' + str(p))
            if way2 == 0:
                for k in range(1, 2, 2):
                    way[2] = k
                    ave = 0
                    times = 10
                    for i in range(times):
                        ave += k_fold(x, y, rule_base, way, k=10)
                    print("total ave_acc = " + str(ave / times) + "  " + title[way1 * 2 + way2] + ' ' + str(k))


if __name__ == '__main__':
    for i in range(9):
        main((i+1) / 10)

