from DBRB.iidbrb import IIDBRBClassifier
from datasets.load_data import load_breast_cancer
from datasets.process_data import process_to_pieces
import numpy as np
import pandas as pd

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

A, D = process_to_pieces(np.array(complete_X), np.array(complete_y), 2)
dbrb = IIDBRBClassifier(A, D)
dbrb = dbrb.fit(complete_X, complete_y)
y_predict = dbrb.predict(incomplete_X)
print(y_predict)
print(incomplete_y)

