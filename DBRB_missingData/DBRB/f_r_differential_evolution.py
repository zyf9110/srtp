import geatpy as ea
import numpy as np
from sklearn.metrics import mean_squared_error

"""
# 文档地址：https://github.com/geatpy-dev/geatpy
"""


class MyProblem(ea.Problem):
    def __init__(self, brb_model, X, y):
        M = 1  # M目标函数维度
        maxormins = np.array([1] * M)  # 初始化maxormins（目标最小值最大化标记列表，1：最小化目标；-1：最大化目标）
        Dim = brb_model.get_params_num()  # 决策变量个数
        varTypes = [0] * Dim  # 初始化决策变量类型，元素为0表示对应的变量是连续的，1表示是离散的
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界 1：闭区间  0：开区间
        ubin = [1] * Dim  # 决策变量上边界 1：闭区间  0：开区间

        # 效用边界
        utility_start_index = len(brb_model.A) * len(brb_model.rules) + len(brb_model.D) * len(brb_model.rules)
        min = np.nanmin(X, 0)
        max = np.nanmax(X, 0)
        for t in range(len(brb_model.rules)):
            for j in range(len(brb_model.A)):
                lb[- utility_start_index + t * len(brb_model.A) + j] = min[j]
                ub[- utility_start_index + t * len(brb_model.A) + j] = max[j]
        super().__init__("BRB_target_func", M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

        self.brb = brb_model
        self.X = X
        self.y = y

    def aimFunc(self, pop):
        """
        目标函数
        :param pop:  geatpy中的Population类型
        :return:
        """
        Vars = pop.Phen  # 决策变量矩阵
        res = []

        for i in range(Vars.shape[0]):
            self.brb.set_params(Vars[i])
            if self.brb.is_classify:
                pre_y = self.brb.classify(self.X)
            else:
                pre_y = self.brb.predict(self.X)
            res.append(mean_squared_error(self.y, pre_y))
        res = np.array(res)
        pop.ObjV = res.reshape((-1, 1))

        len_D = len(self.brb.D)
        len_rules = len(self.brb.rules)

        col = np.zeros((len_rules, len(Vars))) * np.nan
        index = 0
        for i in range(len_rules):  # beta只和小于等于1
            col[index] = np.sum(Vars[:, np.shape(Vars)[1] - len_D * len_rules + i*len_D: np.shape(Vars)[1] - len_D * len_rules + (i+1)*len_D], axis=1) - 1
            index += 1

        pop.CV = col.T  # 违反约束程度， 每行代表一个个体，每列列代表一个约束条件


def run_DE_algorithm(brb_model, X, y, NIND=50, MAXGEN=1000):
    """
    :param brb_model:
    :param X:
    :param y:
    :param NIND: 种群规模
    :param MAXGEN: 最大遗传代数
    :return:
    """
    problem = MyProblem(brb_model, X, y)
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
    best_params = []
    for i in range(var_trace.shape[1]):
        best_params.append(var_trace[best_gen, i])
    #     print('第%s个参数：%s' % (i, var_trace[best_gen, i]))
    print(best_params)
    print('最优的目标函数值为：%s' % best_ObjV)
    print('最优的一代是第%s代' % (best_gen + 1))
    print('耗时：%s 秒' % algorithm.passTime)
    print('### 结束参数优化 ###')
    print("")
    brb_model.set_params(best_params)
    return brb_model

# run_DE_algorithm()
