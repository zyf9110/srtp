import scipy.stats as stats
import math
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('raw_risk_evaluation.csv')

row, col = df.shape
cor = []
a = 0
index = []
for i in range(col-1):
    pearson_corr, _ = stats.spearmanr(df[df.columns[i]], df[df.columns[col-1]])
    # cor.append([math.fabs(pearson_corr) ,pearson_corr, i])
    if math.fabs(pearson_corr) > 0.3:
        cor.append(math.fabs(pearson_corr))
        a += math.fabs(pearson_corr)
        index.append(i)
print(len(cor))
temp = []
for i in cor:
    temp.append(i/a)
print(temp)
data = {}
for i in index:
    data[df.columns[i]] = df[df.columns[i]]
data[df.columns[col-1]] = df[df.columns[col-1]]
data = pd.DataFrame(data)

# data.to_csv('risk_cor0.5.csv', index=False)



# cor.sort()
# print(cor)
# print(a)

# cor = []
# for i in range(len(x)):
#     print(cop[i])
#     pearson_corr, _ = stats.pearsonr(cop[i], y)
#     cor.append([i, pearson_corr])
#
# cor.sort()
# 计算皮尔逊相关系数
# pearson_corr, _ = stats.pearsonr(variable1, variable2)
# print("Pearson correlation coefficient:", pearson_corr)

# # 计算斯皮尔曼相关系数
# spearman_corr, _ = stats.spearmanr(variable1, variable2)
# print("Spearman correlation coefficient:", spearman_corr)
#
# # 计算肯德尔相关系数
# kendall_corr, _ = stats.kendalltau(variable1, variable2)
# print("Kendall correlation coefficient:", kendall_corr)