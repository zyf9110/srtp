import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from EDBRB.ebrb import EDBRBClassifier

from datasets.load_data import load_iris
from datasets.process_data import process_to_pieces
# import hdbscan


def cross_validation():
    N_SPLITS = 10
    X, y = load_iris()
    #  2  best_acc:0.933333(std:0.051640), avg_process_time:0.003088 avg_acc:0.896667
    #  4  best_acc:0.920000(std:0.058119), avg_process_time:0.003643 avg_acc:0.894000
    #  5  best_acc:0.960000(std:0.044222), avg_process_time:0.003999 avg_acc:0.946667
    # 10  best_acc:0.960000(std:0.053333), avg_process_time:0.005593 avg_acc:0.950000
    # 15  best_acc:0.940000(std:0.062893), avg_process_time:0.007309 avg_acc:0.928000
    # 20  best_acc:0.926667(std:0.046667), avg_process_time:0.029949 avg_acc:0.911333
    A, D = process_to_pieces(X, y, 3, 4)
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
