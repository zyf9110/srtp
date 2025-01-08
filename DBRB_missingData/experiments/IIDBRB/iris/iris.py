# coding=utf-8
import random

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from DBRB.iidbrb2 import IIDBRBClassifier2
from datasets.load_data import load_iris
from datasets.process_data import process_to_pieces


def cross_validation():
    N_SPLITS = 10
    X, y = load_iris()

    A, D = process_to_pieces(X, y, 3)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    dbrb = IIDBRBClassifier2(A, D)
    maes = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        dbrb.fit(train_X, train_y)

        row = []
        for i in range(np.shape(A)[0]):
            tmp_X = np.copy(test_X)
            tmp_X[:, i] = np.full(len(tmp_X), np.nan)
            y_predict = dbrb.predict(tmp_X)
            row.append(accuracy_score(y_predict, test_y))
        maes.append(row)
    return np.mean(maes, 0), np.std(maes, 0)


total_acc = []
avg_acc = 0
for tc in range(10):
    print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
    acc, std = cross_validation()
    total_acc   .append(acc)
    print("mean acc is :")
    print(np.mean(total_acc, 0))
    print("best acc is:")
    print(np.max(total_acc, 0))
    print("")
