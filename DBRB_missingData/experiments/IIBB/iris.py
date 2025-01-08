from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
from datasets.load_data import load_iris
from datasets.process_data import process_to_pieces
from DBRB.incomplete_inference_belief_base import IIBB
from DBRB.iidbrb import transform_to_belief
import pandas as pd


X, y = load_iris()
kf = KFold(n_splits=10, shuffle=True)
maes = []
for train_index, test_index in kf.split(X):
    train_X, train_y = X[train_index, :], y[train_index]
    test_X, test_y = X[test_index], y[test_index]
    A, D = process_to_pieces(train_X, train_y, 2)
    iibb = IIBB(train_X, A)
    # 缺失第i个属性的数据
    row = []
    for i in range(np.shape(A)[0]):
        tmp_X = np.copy(test_X)
        tmp_X[:, i] = np.full(len(tmp_X), np.nan)
        y_predict = iibb.resolve(tmp_X)
        maes_tmp = []
        for j in range(len(y_predict)):
            y1 = np.array(transform_to_belief(test_X[j], A))[i]
            y2 = np.array(y_predict[j])[i]
            maes_tmp.append(np.sqrt(np.sum((y1 - y2) ** 2)))
        row.append(np.mean(maes_tmp))
    maes.append(row)

print(maes)
print('行均值：')
print(np.mean(maes, 1))
print('列均值：')
print(np.mean(maes, 0))
