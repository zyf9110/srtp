import random
import time

from EDBRB.ebrb import EDBRBClassifier
from datasets.process_data import process_to_pieces
from datasets.load_data import load_risk
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import math
import numpy as np
import hdbscan


def cross_validation(N):
    N_SPLITS = 10
    X, y = load_risk()
    X=np.array(X)
    print(X[0][0])

    A, D = process_to_pieces(X, y, 2, N)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)

    ebrb = EDBRBClassifier(A, D)
    maes = []
    times = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]

        train_vx,train_vy = [],[]
        test_vx, test_vy = [],[]
        t = 0
        for i in range(len(train_X)):
            if t < 200 and random.randint(0,5) == 0:
                train_vx.append(train_X[i])
                train_vy.append(train_y[i])
                t += 1
            else:
                test_vx.append(train_X[i])
                test_vy.append(train_y[i])
        for i in range(len(test_X)):
            test_vx.append(test_X[i])
            test_vy.append(test_y[i])

        ebrb = ebrb.fit(train_vx, train_vy)
        y_predict = ebrb.predict(test_vx)
        maes.append(accuracy_score(y_predict, test_vy))
        times.append(ebrb.average_process_time)
    return np.mean(maes), np.std(maes), np.mean(times)

def again(N):
    best_acc, best_std, best_time = 0, 0, 0
    avg_acc = 0
    for tc in range(3):
        start_time = time.time()
        print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
        acc, std, times = cross_validation(N)
        avg_acc = (tc * avg_acc + acc) / (tc + 1)
        if acc > best_acc:
            best_acc, best_std, best_time = acc, std, times
        print("acc:%f(std:%f), avg_process_time:%f" % (acc, std, times))
        print("best_acc:%f(std:%f), avg_process_time:%f" % (best_acc, best_std, best_time))
        print("avg_acc:%f" % avg_acc)
        print("花费时间: ", time.time() - start_time)
    return [best_acc,best_time]

result = []
# for i in [2,4,5,9,10,15,20]:
for i in [9]:
    acc = again(i)
    result.append(acc)

print(result)