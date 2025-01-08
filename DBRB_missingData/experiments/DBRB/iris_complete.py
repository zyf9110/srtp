import random

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from DBRB.dbrb import DBRBClassifier
from datasets.load_data import load_iris
from datasets.process_data import process_to_pieces


def cross_validation():
    N_SPLITS = 10
    X, y = load_iris()
    A, D = process_to_pieces(X, y, 3)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    dbrb = DBRBClassifier(A, D)
    maes = []
    times = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        dbrb = dbrb.fit(train_X, train_y)
        y_predict = dbrb.classify(test_X)
        maes.append(accuracy_score(y_predict, test_y))
        times.append(dbrb.average_process_time)
    return np.mean(maes), np.std(maes), np.mean(times)


'''
完整数据，做对比实验
'''
best_acc, best_std, best_time = 0, 0, 0
avg_acc = 0
for tc in range(10):
    print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
    acc, std, times = cross_validation()
    avg_acc = (tc * avg_acc + acc) / (tc + 1)
    if acc > best_acc:
        best_acc, best_std, best_time = acc, std, times
    print("acc:%f(std:%f), avg_process_time:%f" % (acc, std, times))
    print("best_acc:%f(std:%f), avg_process_time:%f" % (best_acc, best_std, best_time))
    print("avg_acc:%f" % avg_acc)
    print("")
