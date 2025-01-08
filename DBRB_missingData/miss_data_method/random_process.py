import random
import numpy as np
import pandas as pd


def random_array(X, percent):
    X_copy = np.copy(X).astype(np.float64)
    tmp = []
    for i in range(np.shape(X)[0]):
        tmp.append([0] * (len(X[i]) + 1))
        tmp[i][0] = i
        for j in range(np.shape(X)[1]):
            tmp[i][j + 1] = j

    index_set = set()
    flag = True
    while flag:
        row = random.randint(0, np.shape(tmp)[0] - 1)
        col = random.randint(1, np.shape(tmp[row])[0] - 1)
        index_set.add((tmp[row][0], tmp[row][col]))
        del tmp[row][col]
        if len(index_set) >= percent * np.shape(X)[0] * np.shape(X)[1]:
            flag = False
        if len(tmp[row]) == 1:
            del tmp[row]

    for (row, col) in index_set:
        X_copy[row][col] = None
    return X_copy


# def df_random_process(df_train, miss_percent):
#     """
#     传入一个df类型数据
#     :param df_train:
#     :param miss_percent:
#     :return:
#     """
#     X_copy = random_array(df_train[:-1].values, miss_percent)
#     X_copy = pd.DataFrame(X_copy, columns=[value for value in range(len(X_copy) - 1)])
#     X_copy['y'] = df_train[-1]
#     return X_copy

# X = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]]
# print(random_array(X, 0.6))
