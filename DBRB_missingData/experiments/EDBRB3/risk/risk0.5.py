import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from EDBRB.ebrb import EDBRBClassifier

from datasets.load_data import load_risk4
from datasets.process_data import process_to_pieces
import random


def unify_single_miss(X, y, miss_percent):
    """
    统一缺失单个属性：
        数据集X中只缺失某一个属性值，缺失的位置是随机的
    :param X: 
    :param y: 
    :param miss_percent: 训练数据集的缺失率
    :return: 返回list格式，其中第i个位置代表缺失第i个属性的平均数据
    """
    N_SPLITS = 2
    A, D = process_to_pieces(X, y, 2, 4)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    ebrb = EDBRBClassifier(A, D)

    total_maes = []
    total_std = []
    total_times = []
    for i in range(np.shape(X)[1]):  # 设置缺失第i个属性值
        maes = []
        times = []
        for train_index, test_index in kf.split(X):
            train_X, train_y = X[train_index, :], y[train_index]
            test_X, test_y = X[test_index], y[test_index]
            X_copy = np.copy(train_X)
            for j in random.sample([value for value in range(len(train_X))],
                                   int(miss_percent * len(train_X))):  # 随机获取空值索引
                X_copy[j][i] = None  # 将获取的索引位置设置为空
            ebrb = ebrb.fit(X_copy, train_y)
            y_predict = ebrb.predict(test_X)
            maes.append(accuracy_score(y_predict, test_y))
            times.append(ebrb.average_process_time)
        total_maes.append(np.mean(maes))
        total_std.append(np.std(maes))
        total_times.append(np.mean(times))
    return total_maes, total_std, total_times


def random_miss(X, y, miss_percent, miss_feature_nums=1):
    """
    随机缺失多个属性值
    :param X:
    :param y:
    :param miss_percent: 训练数据集的缺失率
    :param miss_feature_nums: 缺失属性个数
    :return: 返回十折实验后的平均数据
    """
    N_SPLITS = 10

    temp_ = [[], []]
    t = 200
    x_in = []
    while t:
        a = random.randint(0, 1197)
        if a not in x_in:
            x_in.append(a)
            t -= 1
    x_in.sort()
    X_, y_ = [], []
    # for i in range(len(X)):
    #     if i < 600:
    #         X_.append(X[i])
    #         y_.append(y[i])
    #     else:
    #         break
    for i in range(len(X)):
        if i in x_in:
            X_.append(X[i])
            y_.append(y[i])
        else:
            temp_[0].append(X[i])
            temp_[1].append(y[i])
    X_ = np.array(X_)
    y_ = np.array(y_)

    A, D = process_to_pieces(X, y, 2, 9)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    sigma = [0.027878511017689324, 0.027788421967163867, 0.04718774026895699, 0.0533445592106889, 0.028920547710856108, 0.02949066699603518, 0.04652229419771956, 0.04320452388736006, 0.04170455739960938, 0.04310415542188969, 0.028864225536482168, 0.02696809515970698, 0.03161985041298906, 0.026574054105181013, 0.02724070697971776, 0.04117530036750375, 0.04084266637550822, 0.0268196835392445, 0.03981538689954374, 0.026273165361417196, 0.04339109681613088, 0.05018627052104866, 0.037251589877245014, 0.03864236935674763, 0.03448743061235808, 0.0383708883317531, 0.025488447193220375, 0.02684279447623289]

    ebrb = EDBRBClassifier(A, D, sigma=sigma)

    maes = []
    times = []
    my_ = [[0, 0], [0, 0]]
    for train_index, test_index in kf.split(X_):
        train_X, train_y = X_[train_index, :], y_[train_index]
        test_X, test_y = X_[test_index], y_[test_index]
        X_copy = np.copy(train_X).astype(float)
        for r in random.sample([value for value in range(len(train_X))],
                               int(miss_percent * len(train_X))):  # 随机获取行索引
            for c in random.sample([value for value in range(np.shape(train_X)[1])],
                                   miss_feature_nums):  # 随机获取列索引
                X_copy[r][c] = np.nan  # 将获取的索引位置设置为空
        ebrb = ebrb.fit(X_copy, train_y)
        y_predict = ebrb.predict(temp_[0])

        temp_a = [[0, 0], [0, 0]]
        for i in range(len(y_predict)):
            temp_a[int(y_predict[i])][int(temp_[1][i])] += 1
        print(temp_a, '  ', temp_a[1][1] / (temp_a[1][1] + temp_a[1][0]))
        my_.append(temp_a[1][1] / (temp_a[1][1] + temp_a[1][0]))
        maes.append(accuracy_score(y_predict, temp_[1]))
        times.append(ebrb.average_process_time)
    print('平均精确度', np.mean(my_))
    return np.mean(maes), np.std(maes), np.mean(times)


def unify_miss_CV(miss_percent):
    if miss_percent >= 1:
        raise RuntimeError("数据缺失率应该小于1")
    X, y = load_mammographic()
    total_acc = []
    total_std = []
    total_times = []
    for tc in range(5):
        print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
        acc, std, times = unify_single_miss(X, y, miss_percent)
        total_acc.append(acc)
        total_std.append(std)
        total_times.append(times)
        print("acc: %s " % acc)
        print("std: %s" % std)
        print("avg_process_time: %s" % times)
        print("")
        best_acc = np.max(total_acc, 0)
        best_idx = np.argmax(total_acc, 0)
        idx = [i for i in range(len(acc))]
        print("miss data percent: %f%%" % int(miss_percent * 100))
        print("best_acc: %s(avg: %s)" % (best_acc, np.mean(best_acc)))
        print("std: %s" % np.array(total_std)[best_idx, idx])
        print("time: %s" % np.array(total_times)[best_idx, idx])
        mean = np.mean(total_acc, 0)
        print("avg_acc: %s(avg: %s)" % (mean, np.mean(mean)))
        print("")
    return np.mean(best_acc), np.mean(mean)


def random_miss_CV(miss_percent, miss_attr_num):
    if miss_percent >= 1:
        raise RuntimeError("数据缺失率应该小于1")
    X, y = load_risk4()
    if miss_attr_num > np.shape(X)[1]:
        raise RuntimeError("缺失属性数大于数据集特征值数")

    best_acc, best_std, best_time = 0, 0, 0
    avg_acc = 0
    for tc in range(5):
        start_time = time.time()
        print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
        acc, std, times = random_miss(X, y, miss_percent, miss_attr_num)
        avg_acc = (tc * avg_acc + acc) / (tc + 1)
        if acc > best_acc:
            best_acc, best_std, best_time = acc, std, times
        print("acc:%f(std:%f), avg_process_time:%f" % (acc, std, times))
        print("best_acc:%f(std:%f), avg_process_time:%f" % (best_acc, best_std, best_time))
        print("avg_acc:%f" % avg_acc)
        print("花费时间：", time.time() - start_time)
    return best_acc, avg_acc


# === random miss ==
# random_miss_CV(miss_percent, miss_attr_num)
# 1 [95.3333, 95.3333, 95.3333, 96.0000, 95.3333, 95.3333, 96.0000, 95.3333, 95.3333]
# 2 [95.3333, 95.3333, 95.3333, 96.0000, 96.0000, 94.6667, 94.6667, 96.0000, 95.3333]
# 3 [96.0000, 96.0000, 95.3333, 94.6667, 94.6667, 94.6667, 94.6667, 93.3333, 93.3333]

# === unify miss ===
# best_acc = []
# avg_acc = []
# for i in range(9):
#     best, avg = unify_miss_CV((i+1)/10)
#     best_acc.append(best)
#     avg_acc.append(avg)
# print(best_acc)
# print(avg_acc)


# === multiple miss ===
best_acc = []
avg_acc = []

for i in range(9):
    best, avg = random_miss_CV((i+1)/10, 8)
    best_acc.append(best)
    avg_acc.append(avg)
print(best_acc)
print(avg_acc)