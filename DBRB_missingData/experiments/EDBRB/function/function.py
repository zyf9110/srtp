import numpy as np
from sklearn.metrics import mean_absolute_error
from EDBRB.ebrb import EDBRBRegressor
from EBRB.liu_ebrb import LiuEBRBRegressor
from datasets.load_data import load_data_by
import matplotlib.pyplot as plt


N = [5, 10, 15, 30, 50, 70]
train_X, train_y = load_data_by("data.csv")
test_X, test_y = load_data_by("test.csv")
y = []
fig = plt.figure()

# plt.plot(train_X, train_y, '.')
# plt.xlabel('x', size=20)
# plt.ylabel('y', size=20)
# plt.show()

i = 0
for n in N:
    A = [list(np.linspace(0.0, 3.0, num=n))]
    D = [-2.5, -1, 1, 2, 3]  # 结果等级效用值

    debrb = EDBRBRegressor(A, D)
    debrb = debrb.fit(train_X, train_y)
    tmp = debrb.predict(test_X)
    y.append(tmp)
    print("avg_active_num: %d, mae: %f, time: %fs" % (debrb.average_active_rule_number, mean_absolute_error(tmp, test_y)
                                                      , debrb.average_process_time))
    ax = fig.add_subplot(3, 2, i+1)
    ax.plot(test_X, test_y, 'r-', label='real-values', linewidth=2)
    ax.plot(test_X, y[i], '--', label='N=' + str(n), linewidth=2)
    if i % 2 == 0:
        ax.set_ylabel('y', size=15)
    if i == len(N) - 2 or i == len(N) - 1:
        ax.set_xlabel('x', size=15)
    ax.legend(loc='upper left', fontsize=7)
    i += 1

plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.25,hspace=0.4)
# 二维输出
plt.show()
# plt.savefig(str(n)+'.png')
