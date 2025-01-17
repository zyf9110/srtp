from EDBRB.ebrb import EDBRBClassifier
from datasets.process_data import process_to_pieces
from datasets.load_data import load_mammographic
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import math
import numpy as np
import hdbscan


def cross_validation():
    N_SPLITS = 10
    X, y = load_mammographic()
    A, D = process_to_pieces(X, y, 2, 4)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    #
    #  2 best_acc:0.802410(std:0.042494), avg_process_time:0.018233，avg_acc:0.797952
    #  4 best_acc:0.813253(std:0.041123), avg_process_time:0.023994，avg_acc:0.811446
    #  5 best_acc:0.810843(std:0.032797), avg_process_time:0.026250，avg_acc:0.808313
    # 10 best_acc:0.820482(std:0.047724), avg_process_time:0.037816，avg_acc:0.816867
    # 15 best_acc:0.830120(std:0.033842), avg_process_time:0.049202，avg_acc:0.825904
    # 20 best_acc:0.832530(std:0.037505), avg_process_time:0.059681，avg_acc:0.831084
    # 25 best_acc:0.828916(std:0.042358), avg_process_time:0.069701，avg_acc:0.824699
    # 30 best_acc:0.832530(std:0.032530), avg_process_time:0.087644，avg_acc:0.829398
    #
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
