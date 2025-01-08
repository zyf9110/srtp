import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from EBRB.liu_ebrb import LiuEBRBClassifier

from datasets.load_data import load_transfusion
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
    A, D = process_to_pieces(X, y, 2)
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
    A, D = process_to_pieces(X, y, 2)
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
        #     number = np.median(tmp)
        #     number = np.argmax(np.bincount(tmp))
        #     for r in range(np.shape(X_copy)[0]):
        #         if pd.isnull(X_copy[r][c]):
        #             X_copy[r][c] = number
        ebrb = ebrb.fit(X_copy, train_y)
        y_predict = ebrb.predict(test_X)
        maes.append(accuracy_score(y_predict, test_y))
        times.append(ebrb.average_process_time)

    return np.mean(maes), np.std(maes), np.mean(times)


def unify_miss_CV(miss_percent):
    if miss_percent >= 1:
        raise RuntimeError("数据缺失率应该小于1")
    X, y = load_transfusion()
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
    X, y = load_transfusion()
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
# 均值 [0.7665775401069518, 0.7648395721925135, 0.7633021390374333, 0.7632352941176471, 0.7610962566844919, 0.7577540106951872, 0.759090909090909, 0.7515374331550803, 0.7479278074866309]
# 中位数[0.7653074866310161, 0.7626336898395722, 0.7624331550802139, 0.7600267379679146, 0.7578877005347595, 0.7557486631016042, 0.7574197860962565, 0.752139037433155, 0.7448529411764705]
# 众数[0.7647058823529412, 0.7665106951871661, 0.7646390374331552, 0.7616978609625671, 0.7574866310160427, 0.7541443850267378, 0.755013368983957, 0.7497326203208556, 0.7439839572192513]
# 10 CV
# 均值[0.7641414414414416, 0.7643774774774775, 0.764436936936937, 0.7635063063063063, 0.7638432432432432, 0.7628945945945947, 0.7596909909909912, 0.7566558558558558, 0.7517270270270269]
# 中位数[0.7642486486486487, 0.7643882882882884, 0.7632909909909913, 0.7645783783783784, 0.7633351351351352, 0.7601855855855857, 0.7585774774774776, 0.7575882882882885, 0.7547306306306306]
# 众数[0.7651702702702703, 0.7644441441441442, 0.7638936936936939, 0.7629666666666667, 0.7639846846846848, 0.7608396396396395, 0.7584891891891894, 0.7584801801801803, 0.7513891891891892]
# 10 CV - 零激活
# 均值[0.7631657657657659, 0.7620108108108109, 0.7638558558558559, 0.7608522522522523, 0.7578756756756755, 0.7570063063063065, 0.7542747747747747, 0.7495702702702703, 0.7385729729729732]
# 中位数[0.7649225225225227, 0.7615945945945947, 0.7633711711711711, 0.7595738738738739, 0.7579162162162161, 0.7570837837837837, 0.7529621621621622, 0.7460270270270271, 0.7327162162162162]
# 众数[0.7668423423423423, 0.7668612612612613, 0.7631828828828829, 0.7622099099099098, 0.757472072072072, 0.756245045045045, 0.7479891891891893, 0.7350810810810811, 0.7157702702702704]
# === unify miss ===
best_acc = []
avg_acc = []
for i in range(9):
    best, avg = unify_miss_CV((i+1)/10)
    best_acc.append(best)
    avg_acc.append(avg)
# print(best_acc)
print(avg_acc)
# 2CV [0.768649732620321, 0.7674298128342246, 0.7666109625668449, 0.7655748663101605, 0.7650066844919787, 0.7644552139037433, 0.7633522727272728, 0.7623328877005349, 0.7598262032085561]
# 10CV - 零激活
# 均值[0.7675540540540539, 0.7665509009009009, 0.7654950450450451, 0.7647141891891893, 0.7641556306306307, 0.7633182432432433, 0.7618092342342342, 0.7607630630630631, 0.7568855855855856]
# 中位数[0.7673222972972973, 0.7666101351351353, 0.765370945945946, 0.7655761261261262, 0.7642020270270271, 0.763486936936937, 0.7628470720720721, 0.7602806306306307, 0.7558013513513513]
# 众数[0.7676157657657657, 0.7671193693693694, 0.7668918918918919, 0.7667153153153153, 0.766025, 0.7655207207207206, 0.7653898648648649, 0.7609385135135135, 0.7560477477477479]