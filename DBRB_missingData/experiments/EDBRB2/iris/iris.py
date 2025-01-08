import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from datasets.load_data import load_iris
from EDBRB2.differential_evolution import run_DE_algorithm
from EDBRB2.pso import run_PSO_algorithm


def cross_validation():
    N_SPLITS = 10
    X, y = load_iris()
    # 初始方法 best_acc:0.960000(std:0.044222), avg_process_time:0.003999 avg_acc:0.946667
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    maes = []
    times = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        ebrb = run_DE_algorithm(train_X, train_y, 3, 20, 100)
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
# best_acc:0.96
# avg_acc:0.953333
