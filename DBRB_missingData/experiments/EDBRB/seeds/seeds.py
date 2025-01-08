from EDBRB.ebrb import EDBRBClassifier
from datasets.process_data import process_to_pieces
from datasets.load_data import load_seeds
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
    X, y = load_seeds()
    A, D = process_to_pieces(X, y, 3, 4)
    #  2 best_acc:0.904762(std:0.060234), avg_process_time:0.007026 avg_acc:0.878571
    #  4 best_acc:0.909524(std:0.044924), avg_process_time:0.009085 avg_acc:0.893333
    #  5 best_acc:0.880952(std:0.061168), avg_process_time:0.010094 avg_acc:0.871429
    # 10 best_acc:0.876190(std:0.057143), avg_process_time:0.013606 avg_acc:0.871429
    # 15 best_acc:0.871429(std:0.064065), avg_process_time:0.023698 avg_acc:0.864286
    # 20 best_acc:0.871429(std:0.060422), avg_process_time:0.021465 avg_acc:0.861905
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

