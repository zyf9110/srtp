import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from EBRB.dra_ebrb import DRAEBRBClassifier
from EBRB.liu_ebrb import LiuEBRBClassifier

from datasets.load_data import load_seeds
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
                for c in range(np.shape(X_copy)[1]):
                    mi = np.nanmean(X_copy[:, c])
                    for r in range(np.shape(X_copy)[0]):
                        if pd.isnull(X_copy[r][c]):
                            X_copy[r][c] = mi
                # 众数、中位数
                # for c in range(np.shape(X_copy)[1]):
                #     tmp = []
                #     for t in X_copy[:, c]:
                #         if not pd.isnull(t):
                #             tmp.append(t)
                #     # number = np.median(tmp)
                #     number = np.argmax(np.bincount(tmp))
                #     for r in range(np.shape(X_copy)[0]):
                #         if pd.isnull(X_copy[r][c]):
                #             X_copy[r][c] = number
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
    ebrb = LiuEBRBClassifier(A, D)
    maes = []
    times = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        X_copy = random_array(train_X, miss_percent)
        # 均值
        for c in range(np.shape(X_copy)[1]):
            mi = np.nanmean(X_copy[:, c])
            for r in range(np.shape(X_copy)[0]):
                if pd.isnull(X_copy[r][c]):
                    X_copy[r][c] = mi
        # 众数、中位数
        # for c in range(np.shape(X_copy)[1]):
        #     tmp = []
        #     for t in X_copy[:, c]:
        #         if not pd.isnull(t):
        #             tmp.append(t)
            # number = np.median(tmp)
            # number = np.argmax(np.bincount(tmp))
            # for r in range(np.shape(X_copy)[0]):
            #     if pd.isnull(X_copy[r][c]):
            #         X_copy[r][c] = number
        ebrb = ebrb.fit(X_copy, train_y)
        y_predict = ebrb.predict(test_X)
        maes.append(accuracy_score(y_predict, test_y))
        times.append(ebrb.average_process_time)

    return np.mean(maes), np.std(maes), np.mean(times)


def unify_miss_CV(miss_percent):
    if miss_percent >= 1:
        raise RuntimeError("数据缺失率应该小于1")
    X, y = load_seeds()
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
    X, y = load_seeds()
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
# print(best_acc)
print(avg_acc)
# 2 CV
# 均值 [0.9066666666666668, 0.8952380952380954, 0.8683333333333335, 0.8271428571428572, 0.7792857142857144, 0.7026190476190475, 0.6245238095238094, 0.5238095238095237, 0.41238095238095235]
# 中位数 [0.9076190476190475, 0.8923809523809526, 0.8833333333333335, 0.8488095238095239, 0.8040476190476193, 0.7297619047619046, 0.6166666666666667, 0.510952380952381, 0.42880952380952386]
# 众数 [0.9045238095238096, 0.8807142857142859, 0.8373809523809526, 0.7766666666666668, 0.6619047619047618, 0.6019047619047617, 0.4883333333333333, 0.4547619047619048, 0.37928571428571445]
# 10 CV
# 均值[0.9221428571428572, 0.9149999999999998, 0.8957142857142857, 0.8850000000000001, 0.8540476190476192, 0.8026190476190479, 0.7216666666666662, 0.6135714285714283, 0.46880952380952384]
# 中位数[0.9280952380952382, 0.9121428571428571, 0.9059523809523811, 0.8876190476190479, 0.8602380952380955, 0.8116666666666669, 0.7454761904761904, 0.6502380952380948, 0.5016666666666667]
# 众数 [0.9238095238095239, 0.901666666666667, 0.8771428571428574, 0.8266666666666669, 0.759047619047619, 0.6728571428571425, 0.5885714285714283, 0.5183333333333333, 0.40809523809523823]
# 10CV - 零激活
# 均值[0.9247619047619049, 0.9133333333333334, 0.8983333333333337, 0.8752380952380955, 0.8404761904761907, 0.7847619047619048, 0.6933333333333331, 0.5833333333333331, 0.4135714285714286]
# 中位数[0.9271428571428573, 0.9097619047619048, 0.9023809523809525, 0.885952380952381, 0.8616666666666669, 0.8083333333333336, 0.7297619047619047, 0.6135714285714283, 0.43380952380952387]
# 众数[0.9164285714285716, 0.9000000000000001, 0.860714285714286, 0.807857142857143, 0.7402380952380951, 0.6278571428571427, 0.5204761904761905, 0.40166666666666667, 0.24023809523809522]
# - dra_ebrb
# 均值[0.9088095238095238, 0.9042857142857145, 0.9033333333333335, 0.8980952380952381, 0.894761904761905, 0.8892857142857145, 0.8795238095238098, 0.8788095238095239, 0.8230952380952383]
# === unify miss ===
# best_acc = []
# avg_acc = []
# for i in range(9):
#     best, avg = unify_miss_CV((i+1)/10)
#     best_acc.append(best)
#     avg_acc.append(avg)
# print(best_acc)
# print(avg_acc)
# 2CV [0.9165986394557823, 0.9156122448979592, 0.9137074829931973, 0.9089115646258504, 0.90765306122449, 0.9037414965986396, 0.8942176870748302, 0.8769387755102044, 0.8368027210884356]
# 10CV - 零激活
# 均值[0.9345918367346938, 0.9320068027210885, 0.93, 0.9254761904761907, 0.9249659863945577, 0.9205102040816326, 0.9135714285714285, 0.9049319727891156, 0.8812244897959186]
# 中位数[0.9351020408163265, 0.933265306122449, 0.9315306122448981, 0.9286054421768707, 0.9282312925170068, 0.922721088435374, 0.9199659863945578, 0.9132993197278912, 0.8935714285714287]
# 众数[0.9347278911564627, 0.9323469387755102, 0.9279591836734695, 0.9231972789115648, 0.9209183673469388, 0.9134013605442178, 0.9060544217687075, 0.8904421768707484, 0.8487074829931974]
