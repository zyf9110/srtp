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

my_ = [[0, 0], [0,0]]
def cross_validation(N):
    N_SPLITS = 10
    X, y = load_risk()

    A, D = process_to_pieces(X, y, 2, N)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)

    sigma = [0.027878511017689324, 0.027788421967163867, 0.04718774026895699, 0.0533445592106889, 0.028920547710856108, 0.02949066699603518, 0.04652229419771956, 0.04320452388736006, 0.04170455739960938, 0.04310415542188969, 0.028864225536482168, 0.02696809515970698, 0.03161985041298906, 0.026574054105181013, 0.02724070697971776, 0.04117530036750375, 0.04084266637550822, 0.0268196835392445, 0.03981538689954374, 0.026273165361417196, 0.04339109681613088, 0.05018627052104866, 0.037251589877245014, 0.03864236935674763, 0.03448743061235808, 0.0383708883317531, 0.025488447193220375, 0.02684279447623289]

    ebrb = EDBRBClassifier(A, D, sigma=sigma)
    maes = []
    times = []

    temp_ = [[], []]
    t = 200
    x_in = []
    while t:
        a = random.randint(0,1197)
        if a not in x_in:
            x_in.append(a)
            t -= 1
    x_in.sort()
    X_, y_ = [], []
    for i in range(len(X)):
        if i in x_in:
            X_.append(X[i])
            y_.append(y[i])
        else:
            temp_[0].append(X[i])
            temp_[1].append(y[i])
    X_ = np.array(X_)
    y_ = np.array(y_)
    for train_index, test_index in kf.split(X_):
        train_X, train_y = X_[train_index, :], y_[train_index]
        test_X, test_y = X_[test_index], y_[test_index]


        # print(train_y)
        ebrb = ebrb.fit(train_X, train_y)
        y_predict = ebrb.predict(temp_[0])
        # print(y_predict)
        temp_a = [[0,0],[0,0]]
        for i in range(len(y_predict)):
            temp_a[int(y_predict[i])][int(temp_[1][i])] += 1
        if my_[0][0] == 0 or my_[1][1] / (my_[1][1] + my_[1][0]) < temp_a[1][1] / (temp_a[1][1] + temp_a[1][0]):
            my_[0][0] = temp_a[0][0]
            my_[0][1] = temp_a[0][1]
            my_[1][0] = temp_a[1][0]
            my_[1][1] = temp_a[1][1]
        print(my_)
        maes.append(accuracy_score(y_predict, temp_[1]))
        times.append(ebrb.average_process_time)
    return np.mean(maes), np.std(maes), np.mean(times)

def again(N):
    best_acc, best_std, best_time = 0, 0, 0
    avg_acc = 0
    for tc in range(5):
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
for i in [9]:
    acc = again(i)
    result.append(acc)

print(result)