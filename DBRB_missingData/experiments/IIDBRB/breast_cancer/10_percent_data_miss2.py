from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from DBRB.iidbrb2 import IIDBRB2, IIDBRBClassifier2
from datasets.load_data import load_breast_cancer
import numpy as np
import pandas as pd

from datasets.process_data import process_to_pieces

X, y = load_breast_cancer()
X = X[:, 1:]
incomplete_X = []
incomplete_y = []
complete_X = []
complete_y = []
for i in range(np.shape(X)[0]):
    if pd.isnull(X[i]).sum() != 0:
        incomplete_X.append(X[i])
        incomplete_y.append(y[i])
    else:
        complete_X.append(X[i])
        complete_y.append(y[i])

complete_X = np.array(complete_X)
complete_y = np.array(complete_y)
kf = KFold(n_splits=10, shuffle=True)
maes = []
for train_index, test_index in kf.split(complete_X):
    train_X, train_y = complete_X[train_index, :], complete_y[train_index]
    test_X, test_y = complete_X[test_index], complete_y[test_index]
    A, D = process_to_pieces(train_X, train_y, 2)
    dbrb = IIDBRBClassifier2(A, D)
    dbrb = dbrb.fit(train_X, train_y)
    # 缺失第i个属性的数据
    row = []
    for i in range(np.shape(A)[0]):
        tmp_X = np.copy(test_X)
        tmp_X[:, i] = np.full(len(tmp_X), np.nan)
        y_predict = dbrb.predict(tmp_X)
        row.append(accuracy_score(y_predict, test_y))
    maes.append(row)
    print(row)

print(maes)
print("")
print('行均值：')
print(np.mean(maes, 1))
print('列均值：')
print(np.mean(maes, 0))
