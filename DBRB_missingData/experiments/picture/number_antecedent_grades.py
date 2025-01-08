import numpy as np
import matplotlib.pyplot as plt


x = [2, 4, 5, 10, 15, 20]
# ----- iris -----
iris = [0.933333,0.920000,0.960000,0.960000,0.940000,0.926667]
iris_time = [0.003088,0.003643,0.003999,0.005593,0.007309,0.029949]
# ----- seeds -----
seeds = [0.904762,0.909524,0.880952,0.876190,0.871429,0.871429]
seeds_time = [0.007026,0.009085,0.010094,0.013606,0.023698,0.021465]
# ----- transfusion -----
transfusion =[0.762162,0.762108,0.762054,0.762180,0.762180,0.762162]
transfusion_time = [0.014061,0.762022,0.761987,0.762014,0.762016,0.762007]
# ----- wine -----
wine = [0.583007,0.881373,0.909477,0.955556,0.966340,0.961111]
wine_time = [0.010215,0.012840,0.014689,0.021573,0.029561,0.034151]
# ----- mammographic -----
mammographic = [0.802410,0.813253,0.810843,0.820482,0.830120,0.832530]
mammographic_time = [0.018233,0.023994,0.026250,0.037816,0.049202,0.059681]

# 开启一个窗口
fig = plt.figure()
# 使用add_subplot在窗口加子图，其本质就是添加坐标系
# 三个参数分别为：行数，列数，本子图是所有子图中的第几个，最后一个参数设置错了子图可能发生重叠
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 2, 5)

a1 = ax1.plot(x, iris, '-', label='accuracy')
ax1.set_xlabel("(a)",size=15)
# ax1.set_ylabel("ratio of classes correctly classified", size=15)
# ax1.legend(loc='upper left', fontsize=10)
ax12 = ax1.twinx()
a12 = ax12.plot(x, iris_time, '-r', label='time')
# ax12.set_ylabel("inference process time(s)")
# ax12.legend(loc='upper right')
lns1 = a1 + a12
label1 = [l.get_label() for l in lns1]
ax1.legend(lns1, label1, loc=4, fontsize=8)


a2 = ax2.plot(x, seeds, '-', label='accuracy')
ax2.set_xlabel("(b)",size=15)
# ax2.set_ylabel("ratio of classes correctly classified")
# ax2.legend(loc='upper left', fontsize=10)
ax22 = ax2.twinx()
a22 = ax22.plot(x, seeds_time, '-r', label='time')
# ax22.set_ylabel("inference process time(s)", size=15)
# ax22.legend(loc='upper right', fontsize=10)
lns2 = a2 + a22
label2 = [l.get_label() for l in lns2]
ax2.legend(lns2, label2, loc=4, fontsize=8)


a3 = ax3.plot(x, transfusion, '-', label='accuracy')
ax3.set_xlabel("(c)",size=15)
# ax3.set_ylabel("ratio of classes correctly classified", size=15)
# ax3.legend(loc='upper left', fontsize=10)
ax32 = ax3.twinx()
a32 = ax32.plot(x, transfusion_time, '-r', label='time')
# ax32.set_ylabel("inference process time(s)")
# ax32.legend(loc='upper right', fontsize=10)
lns3 = a3 + a32
label3 = [l.get_label() for l in lns3]
ax3.legend(lns1, label1, fontsize=8)


a4 = ax4.plot(x, wine, '-', label='accuracy')
ax4.set_xlabel("(d)",size=15)
# ax4.set_ylabel("ratio of classes correctly classified")
# ax4.legend(loc='upper left', fontsize=10)
ax42 = ax4.twinx()
a42 = ax42.plot(x, wine_time, '-r', label='time')
# ax42.set_ylabel("inference process time(s)", size=15)
# ax42.legend(loc='upper right', fontsize=10)
lns4 = a4 + a42
label4 = [l.get_label() for l in lns4]
ax4.legend(lns4, label4, fontsize=8)

a5 = ax5.plot(x, mammographic, '-', label='accuracy')
ax5.set_xlabel("(e)",size=15)
# ax5.set_ylabel("accurcy", size=15)
# ax5.legend(loc='upper left', fontsize=10)
ax52 = ax5.twinx()
a52 = ax52.plot(x, mammographic_time, '-r', label='time')
# ax52.set_ylabel("inference process time(s)", size=15)
# ax52.legend(loc='upper right', fontsize=10)
lns5 = a5 + a52
label5 = [l.get_label() for l in lns5]
ax5.legend(lns5, label5, loc=4, fontsize=8)

plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.4,hspace=0.4)
plt.show()
# plt.savefig('0.png')
