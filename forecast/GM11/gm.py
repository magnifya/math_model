import numpy as np
import pandas as pd
from scipy import io, integrate, linalg, signal
from scipy.sparse.linalg import eigs
from scipy.integrate import odeint
# %matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

# 参考：https://www.cnblogs.com/jjmg/p/grey_model_by_python.html
# 其他案例：https://github.com/dontLoveBugs/GM-1-1

# 线性平移预处理，确保数据级比在可容覆盖范围
def greyModelPreprocess(dataVec):
    "Set linear-bias c for dataVec"

    c = 0
    x0 = np.array(dataVec, float)
    n = x0.shape[0]  # 行数
  
    # 确定数值上下限
    L = np.exp(-2/(n+1))
    R = np.exp(2/(n+2))
    xmax = x0.max()
    xmin = x0.min()
    if xmin < 1:
        x0 += (1-xmin)
        c += (1-xmin)
    xmax = x0.max()
    xmin = x0.min()
    lambda_ = x0[0:-1] / x0[1:]  # 计算级比
    lambda_max = lambda_.max()
    lambda_min = lambda_.min()
    while (lambda_max > R or lambda_min < L):
        x0 += xmin
        c += xmin
        xmax = x0.max()
        xmin = x0.min()
        lambda_ = x0[0:-1] / x0[1:]
        lambda_max = lambda_.max()
        lambda_min = lambda_.min()
    return c

# 灰色预测模型
def greyModel(dataVec, predictLen):
    "Grey Model for exponential prediction"
    # dataVec = [1, 2, 3, 4, 5, 6]
    # predictLen = 5

    x0 = np.array(dataVec, float)
    n = x0.shape[0]
    x1 = np.cumsum(x0)
    B = np.array([-0.5 * (x1[0:-1] + x1[1:]), np.ones(n-1)]).T
    Y = x0[1:]
    u = linalg.lstsq(B, Y)[0]

    def diffEqu(y, t, a, b):
        return np.array(-a * y + b)

    t = np.arange(n + predictLen)
    sol = odeint(diffEqu, x0[0], t, args=(u[0], u[1]))
    sol = sol.squeeze()
    res = np.hstack((x0[0], np.diff(sol)))
    return res
    
    
# 输入数据
x = np.array([-18, 0.34, 4.68, 8.49, 29.84, 50.21, 77.65, 109.36])
c = greyModelPreprocess(x)
x_hat = greyModel(x+c, 5)-c

# 画图
t1 = range(x.size)
t2 = range(x_hat.size)
plt.plot(t1, x, color='r', linestyle="-", marker='*', label='True')
plt.plot(t2, x_hat, color='b', linestyle="--", marker='.', label="Predict")
plt.legend(loc='upper right')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.title('Prediction by Grey Model (GM(1,1))')
plt.show()