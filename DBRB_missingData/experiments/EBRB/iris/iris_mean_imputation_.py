import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from EBRB.liu_ebrb import LiuEBRBClassifier

from datasets.load_data import load_iris
from datasets.process_data import process_to_pieces
import random
import pandas as pd
from miss_data_method.random_process import  random_array


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
    :param miss_feature_nums: 缺失属性个数
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
        # 填充均值
        # for c in range(np.shape(X_copy)[1]):
        #     mi = np.nanmean(X_copy[:, c])
        #     for r in range(np.shape(X_copy)[0]):
        #         if pd.isnull(X_copy[r][c]):
        #             X_copy[r][c] = mi
        # 使用中位数(众数)填充
        for c in range(np.shape(X_copy)[1]):
            tmp = []
            for t in X_copy[:, c]:
                if not pd.isnull(t):
                    tmp.append(t)
            number = np.median(tmp)
            # number = np.argmax(np.bincount(tmp))
            for r in range(np.shape(X_copy)[0]):
                if pd.isnull(X_copy[r][c]):
                    X_copy[r][c] = number
        # missingno.matrix(pd.DataFrame(X_copy))
        ebrb = ebrb.fit(X_copy, train_y)
        y_predict = ebrb.predict(test_X)
        maes.append(accuracy_score(y_predict, test_y))
        times.append(ebrb.average_process_time)

    return np.mean(maes), np.std(maes), np.mean(times)


def unify_miss_CV(miss_percent):
    if miss_percent >= 1:
        raise RuntimeError("数据缺失率应该小于1")
    X, y = load_iris()
    total_acc = []
    total_std = []
    total_times = []
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
    X, y = load_iris()
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
# best_acc, avg_acc = [], []
# for per in range(9):
#     best, avg = random_miss_CV((per + 1) / 10)
#     best_acc.append(best)
#     avg_acc.append(avg)
# print(best_acc)
# print(avg_acc)
# 2 折
# 均值 [0.9450000000000001, 0.9373333333333331, 0.9366666666666668, 0.9173333333333336, 0.914, 0.8726666666666667, 0.8543333333333333, 0.7866666666666666, 0.6419999999999999]
# 中位数[0.943, 0.9363333333333331, 0.933, 0.9146666666666666, 0.902666666666667, 0.8746666666666668, 0.8610000000000001, 0.7823333333333333, 0.659]
# 众数[0.9406666666666667, 0.9356666666666668, 0.9343333333333332, 0.9123333333333333, 0.893, 0.8656666666666666, 0.7703333333333331, 0.7059999999999998, 0.5849999999999999]
# 10 折
# 平均值[0.9530000000000001, 0.944, 0.9440000000000003, 0.9376666666666666, 0.9323333333333335, 0.9183333333333333, 0.8973333333333333, 0.8346666666666668, 0.754]
# 中位数 [0.9506666666666668, 0.9476666666666664, 0.9406666666666668, 0.9380000000000001, 0.9283333333333335, 0.9203333333333334, 0.9036666666666668, 0.8620000000000001, 0.7666666666666667]
# 众数 [0.9480000000000001, 0.95, 0.9423333333333332, 0.9396666666666664, 0.9259999999999999, 0.9129999999999999, 0.873, 0.8243333333333334, 0.6316666666666666]
# ---
# 10CV
# 均值
# 中位数
# 众数
# === unify miss ===
best_acc = []
avg_acc = []
for i in range(9):
    best, avg = unify_miss_CV((i+1)/10)
    best_acc.append(best)
    avg_acc.append(avg)
# print(best_acc)
print(avg_acc)
# 2CV [0.9457500000000001, 0.9448333333333334, 0.942, 0.9420833333333333, 0.9368333333333333, 0.9350833333333334, 0.9253333333333332, 0.9194166666666667, 0.8879166666666666]
# 10CV
# 均值[0.9514166666666666, 0.9508333333333332, 0.94875, 0.9484999999999999, 0.9465833333333333, 0.9434166666666667, 0.9382499999999999, 0.9345000000000001, 0.9196666666666669]
# 中位数[0.952, 0.9506666666666667, 0.9500833333333333, 0.9489166666666666, 0.9463333333333332, 0.944, 0.9414166666666667, 0.93625, 0.9193333333333333]
# 众数[0.9515, 0.9514166666666666, 0.9495, 0.9475833333333333, 0.9471666666666667, 0.9439166666666665, 0.9394166666666666, 0.9278333333333334, 0.898]
# ---
# 10CV
# 均值
# 中位数
# 众数
