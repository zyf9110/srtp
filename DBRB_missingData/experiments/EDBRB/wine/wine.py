from EDBRB.ebrb import EDBRBClassifier
from datasets.process_data import process_to_pieces
from datasets.load_data import load_wine
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import math
import numpy as np
# import hdbscan


def cross_validation():
    N_SPLITS = 10
    X, y = load_wine()
    cluster = hdbscan.HDBSCAN()
    max_num = 0
    min_num = 10000
    for i in range(np.shape(X)[1]):
        label = cluster.fit(X[:, i].reshape(-1, 1)).labels_
        max_num = max(max_num, label.max() + 1)
        min_num = min(min_num, label.max() + 1)

    A, D = process_to_pieces(X, y, 3, 15)
    #  2 best_acc:0.583007(std:0.181284), avg_process_time:0.010215 avg_acc:0.555163
    #  4 best_acc:0.881373(std:0.082078), avg_process_time:0.012840 avg_acc:0.846895
    #  5 best_acc:0.909477(std:0.119771), avg_process_time:0.014689 avg_acc:0.885654
    # 10 best_acc:0.955556(std:0.041574), avg_process_time:0.021573 avg_acc:0.945948
    # 15 best_acc:0.966340(std:0.037061), avg_process_time:0.029561 avg_acc:0.951765
    # 20 best_acc:0.961111(std:0.043390), avg_process_time:0.034151 avg_acc:0.947255
    # 25 best_acc:0.955556(std:0.022222), avg_process_time:0.042106 avg_acc:0.952353
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    ebrb = EDBRBClassifier(A, D)
    maes = []
    times = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        ebrb = ebrb.fit(train_X, train_y)
        y_predict = ebrb.predict(test_X)
        maes.append(accuracy_score(y_predict, test_y))
        times.append(ebrb.average_process_time)
    return np.mean(maes), np.std(maes), np.mean(times)


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

