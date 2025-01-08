import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from EBRB.liu_ebrb import LiuEBRBClassifier
from EBRB.dra_ebrb import DRAEBRBClassifier

from datasets.load_data import load_wine
from datasets.process_data import process_to_pieces
import random
import pandas as pd
from miss_data_method.random_process import random_array


def unify_single_miss(X, y, miss_percent):
    """
    统一缺失单个属性：
        数据集X中只缺失某一个属性值，缺失的位置是随机的
    :param X: 
    :param y: 
    :param miss_percent: 训练数据集的缺失率
    :return: 返回list格式，其中第i个位置代表缺失第i个属性的平均数据
    """
    N_SPLITS = 10
    A, D = process_to_pieces(X, y, 3)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    ebrb = LiuEBRBClassifier(A, D)

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
                # 均值
                # for c in range(np.shape(X_copy)[1]):
                #     mi = np.nanmean(X_copy[:, c])
                #     for r in range(np.shape(X_copy)[0]):
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
            ebrb = ebrb.fit(X_copy, train_y)
            y_predict = ebrb.predict(test_X)
            maes.append(accuracy_score(y_predict, test_y))
            times.append(ebrb.average_process_time)
        total_maes.append(np.mean(maes))
        total_std.append(np.std(maes))
        total_times.append(np.mean(times))
    return total_maes, total_std, total_times


def random_miss(X, y, miss_percent):
    """
    随机缺失多个属性值
    :param X:
    :param y:
    :param miss_percent: 训练数据集的缺失率
    :return: 返回十折实验后的平均数据
    """
    N_SPLITS = 10
    A, D = process_to_pieces(X, y, 3)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    ebrb = DRAEBRBClassifier(A, D)
    # ebrb = LiuEBRBClassifier(A, D)

    maes = []
    times = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        X_copy = random_array(train_X, miss_percent)
        # 均值
        # for c in range(np.shape(X_copy)[1]):
        #     mi = np.nanmean(X_copy[:, c])
        #     for r in range(np.shape(X_copy)[0]):
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
        ebrb = ebrb.fit(X_copy, train_y)
        y_predict = ebrb.predict(test_X)
        maes.append(accuracy_score(y_predict, test_y))
        times.append(ebrb.average_process_time)

    return np.mean(maes), np.std(maes), np.mean(times)


def unify_miss_CV(miss_percent):
    if miss_percent >= 1:
        raise RuntimeError("数据缺失率应该小于1")
    X, y = load_wine()
    total_acc = []
    total_std = []
    total_times = []
    best_acc, mean = 0, 0
    for tc in range(20):
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


def random_miss_CV(miss_percent):
    if miss_percent >= 1:
        raise RuntimeError("数据缺失率应该小于1")
    X, y = load_wine()

    best_acc, best_std, best_time = 0, 0, 0
    avg_acc = 0
    for tc in range(20):
        print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
        acc, std, times = random_miss(X, y, miss_percent)
        avg_acc = (tc * avg_acc + acc) / (tc + 1)
        if acc > best_acc:
            best_acc, best_std, best_time = acc, std, times
        print("acc:%f(std:%f), avg_process_time:%f" % (acc, std, times))
        print("")
        print("miss percent: %d%%" % (miss_percent * 100))
        print("acc:%f(std:%f), avg_process_time:%f" % (acc, std, times))
        print("best_acc:%f(std:%f), avg_process_time:%f" % (best_acc, best_std, best_time))
        print("avg_acc:%f" % avg_acc)
        print("")
    return best_acc, avg_acc


# === random miss ===
best_acc, avg_acc = [], []
for per in range(9):
    best, avg = random_miss_CV((per + 1) / 10)
    best_acc.append(best)
    avg_acc.append(avg)
print(best_acc)
print(avg_acc)
# 2 CV
# 均值 [0.7258426966292134, 0.6904494382022472, 0.6432584269662921, 0.6308988764044944, 0.5567415730337079, 0.5252808988764043, 0.505056179775281, 0.43258426966292135, 0.3780898876404495]
# 中位数 [0.7314606741573033, 0.6845505617977529, 0.6365168539325843, 0.5764044943820223, 0.5471910112359548, 0.4952247191011236, 0.4747191011235955, 0.44578651685393256, 0.37893258426966286]
# 众数 [0.6974719101123595, 0.6275280898876403, 0.556741573033708, 0.48904494382022473, 0.4528089887640451, 0.41095505617977535, 0.3837078651685394, 0.3584269662921349, 0.34466292134831467]
#
# 10 CV
# 均值[0.8178594771241832, 0.787450980392157, 0.7356372549019607, 0.6870751633986928, 0.6416993464052287, 0.6019607843137254, 0.5471241830065359, 0.4882679738562092, 0.41990196078431385]
# 中位数[0.8108006535947713, 0.7702614379084967, 0.7298039215686274, 0.6778267973856209, 0.6300000000000001, 0.5784150326797385, 0.5257843137254901, 0.4738562091503268, 0.4214705882352942]
# 众数 [0.7857026143790852, 0.7168954248366012, 0.6503431372549019, 0.569607843137255, 0.5109313725490195, 0.4731372549019607, 0.4176470588235294, 0.3805228758169934, 0.34955882352941164]
#
# 10CV - 零激活
# 均值[0.7926470588235295, 0.7318464052287581, 0.6759803921568628, 0.6140196078431372, 0.5445261437908497, 0.46083333333333326, 0.3733496732026143, 0.2806862745098039, 0.18589869281045757]
# 中位数[0.7832516339869282, 0.7236764705882353, 0.6566176470588235, 0.5826633986928103, 0.4984477124183006, 0.43457516339869284, 0.3420588235294118, 0.25864379084967315, 0.1640032679738562]
# 众数[0.7282189542483659, 0.615294117647059, 0.4941013071895425, 0.34936274509803916, 0.259297385620915, 0.1794607843137255, 0.10702614379084971, 0.05544117647058824, 0.026584967320261443]
# === single miss ===
# best_acc = []
# avg_acc = []
# for i in range(9):
#     best, avg = unify_miss_CV((i+1)/10)
#     best_acc.append(best)
#     avg_acc.append(avg)
# print(best_acc)
# print(avg_acc)
# 2CV [0.33146067415730324, 0.33146067415730324, 0.33146067415730324, 0.33146067415730324, 0.33146067415730324, 0.33146067415730324, 0.33146067415730324, 0.33146067415730324, 0.33146067415730324]
# ---
# 10CV - 零激活
# 均值[0.8264467068878835, 0.8201772247360484, 0.8164768728004023, 0.8098278029160384, 0.8001030668677729, 0.7892885872297638, 0.774089994972348, 0.7517760180995475, 0.7233697838109603]
# 中位数[0.8261023127199596, 0.8187694821518352, 0.8139328808446458, 0.8073453996983411, 0.796675465057818, 0.7846493212669684, 0.7667508798391152, 0.7405015082956259, 0.7035507792860736]
# 众数[0.8233107088989443, 0.8161148818501762, 0.8024157868275517, 0.7909627953745603, 0.7757931121166415, 0.7555769230769231, 0.7252777777777778, 0.6818677727501257, 0.616850175967823]