import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sko.PSO import PSO
import numpy as np
from EDBRB.ebrb import EDBRBClassifier
from datasets.process_data import process_to_pieces


class Problem:
    def __init__(self, X, y, class_num):
        self.X = X
        self.y = y
        self.class_num = class_num

        self.Dim = np.shape(X)[1]
        self.lb = [0] * self.Dim  # 决策变量下界
        self.ub = [1] * self.Dim  # 决策变量上界

        self.Dim += 1  # 前提属性参考值个数优化
        self.lb.append(2)
        self.ub.append(15)

        self.w = 0.5
        self.c1 = 0.5
        self.c2 = 0.5

    def aimFunc(self, pop):
        res = []
        for p in pop:
            A, D = process_to_pieces(self.X, self.y, self.class_num)
            brb = EDBRBClassifier(A, D, p)
            kf = KFold(n_splits=2, shuffle=True)
            maes = []
            for train_index, test_index in kf.split(self.X):
                train_X, train_y = self.X[train_index, :], self.y[train_index]
                test_X, test_y = self.X[test_index], self.y[test_index]
                brb = brb.fit(train_X, train_y)
                y_predict = brb.predict(test_X)
                maes.append(1 - accuracy_score(y_predict, test_y))
            res.append(np.mean(maes))
        return np.reshape(res, (-1, 1))

    def solve(self, pop=20, max_iter=100):
        pso = PSO(self.aimFunc, self.Dim, pop, max_iter, self.lb, self.ub, self.w, self.c1, self.c2)
        pso.run()
        print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
        return pso.gbest_x, pso.gbest_y


def run_PSO_algorithm(X, y, class_num, pop=20, max_iter=100):
    problem = Problem(X, y, class_num)
    print("### 开始参数优化 ###")
    print("使用粒子群优化算法，种群数为%d，最大遗传代数为%s" % (pop, max_iter))
    time_start = time.time()
    gbest_x, gbest_y = problem.solve(pop, max_iter)
    time_end = time.time()
    print(gbest_x)
    print('最优的目标函数值为：%s' % gbest_y)
    print('耗时：%s 秒' % (time_end - time_start))
    print('### 结束参数优化 ###')
    print("")

    A, D = process_to_pieces(X, y, class_num)
    brb = EDBRBClassifier(A, D, gbest_x)
    brb = brb.fit(X, y)
    return brb
