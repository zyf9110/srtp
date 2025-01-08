import geatpy as ea
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from EDBRB.ebrb import EDBRBClassifier
from datasets.process_data import process_to_pieces

"""
# 文档地址：https://github.com/geatpy-dev/geatpy
"""


class MyProblem(ea.Problem):
    def __init__(self, X, y, class_num):
        M = 1  # M目标函数维度
        maxormins = np.array([1] * M)  # 初始化maxormins（目标最小值最大化标记列表，1：最小化目标；-1：最大化目标）
        Dim = np.shape(X)[1]  # 决策变量个数
        varTypes = [0] * Dim  # 初始化决策变量类型，元素为0表示对应的变量是连续的，1表示是离散的
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界 1：闭区间  0：开区间
        ubin = [1] * Dim  # 决策变量上边界 1：闭区间  0：开区间

        # Dim += 1
        # varTypes.append(1)  # 评价指标个数是离散变量
        # lb.append(2)  # 评价指标个数下界
        # ub.append(20)  # # 评价指标个数上界
        # lbin.append(1)
        # ubin.append(1)

        super().__init__("BRB_target_func", M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

        self.X = X
        self.y = y
        self.class_num = class_num

    def aimFunc(self, pop):
        """
        目标函数
        :param pop:  geatpy中的Population类型
        :return:
        """
        Vars = pop.Phen  # 决策变量矩阵
        res = []

        for i in range(Vars.shape[0]):
            A, D = process_to_pieces(self.X, self.y, self.class_num)
            brb = EDBRBClassifier(A, D, Vars[i])
            kf = KFold(n_splits=2, shuffle=True)
            maes = []
            for train_index, test_index in kf.split(self.X):
                train_X, train_y = self.X[train_index, :], self.y[train_index]
                test_X, test_y = self.X[test_index], self.y[test_index]
                brb = brb.fit(train_X, train_y)
                y_predict = brb.predict(test_X)
                maes.append(1 - accuracy_score(y_predict, test_y))
            res.append(np.mean(maes))
        res = np.array(res)
        pop.ObjV = res.reshape((-1, 1))


def run_DE_algorithm_only_params(X, y, class_num, NIND=50, MAXGEN=1000):
    """

    :param X:
    :param y:
    :param class_num: 分类数
    :param NIND: 种群规模
    :param MAXGEN: 最大遗传代数
    :return:
    """
    problem = MyProblem(X, y, class_num)
    Encoding = 'RI'
    field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, field, NIND)  # 实例化种群，但还没有被初始化
    algorithm = ea.soea_DE_rand_1_L_templet(problem, population)  # 实例化算法模版
    algorithm.MAXGEN = MAXGEN  # 最大遗传代数
    algorithm.mutOper.F = 0.5  # 设置变异缩放因子
    algorithm.recOper.XOVR = 0.5  # 设置交叉概率
    algorithm.drawing = 0  # 画出差分进化实验结果图，0：不画；1：画
    print("### 开始参数优化 ###")
    print("使用差分进化算法，种群数为%d，最大遗传代数为%s" % (NIND, MAXGEN))
    [population, obj_trace, var_trace] = algorithm.run()
    # population.save()  # 保存最后的种群信息
    best_gen = np.argmin(obj_trace[:, 1])
    best_ObjV = obj_trace[best_gen, 1]
    best_params = var_trace[best_gen, :]
    print(best_params)
    print('最优的目标函数值为：%s' % best_ObjV)
    print('最优的一代是第%s代' % (best_gen + 1))
    print('耗时：%s 秒' % algorithm.passTime)
    print('### 结束参数优化 ###')
    print("")
    A, D = process_to_pieces(X, y, class_num)
    brb = EDBRBClassifier(A, D, best_params)
    brb = brb.fit(X, y)
    return brb


# run_DE_algorithm()
