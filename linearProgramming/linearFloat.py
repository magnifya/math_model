#导入包
from scipy import optimize
import numpy as np

#确定c,A,b,Aeq,beq
c = np.array([2,3,-5]) #最值等式未知数系数矩阵

A = np.array([[-2,5,-1],[1,3,1]]) #<=不等式左侧未知数系数矩阵
b = np.array([-10,12]) #<=不等式右侧常数矩阵

Aeq = np.array([[1,1,1]]) #等式左侧未知数系数矩阵
beq = np.array([7]) #等式右侧常数矩阵

X1_bounds = [-1,None] #未知数取值范围
X2_bounds = [-1,None] #未知数取值范围
X3_bounds = [0,None] #未知数取值范围

#求解
# res = optimize.linprog(-c,A,b,Aeq,beq)
res = optimize.linprog(-c,A,b,Aeq,beq,bounds=(X1_bounds,X2_bounds,X3_bounds))
print(res)
print("-------------------------------")
print(res.fun)
print("-------------------------------")
print(res.x)
# reslult: 1
# 	fun: -14.571428571428571
# message: 'Optimization terminated successfully.'
#     nit: 2
# 	slack: array([3.85714286, 0.        ])
# 	status: 0
# success: True
#     x: array([6.42857143, 0.57142857, 0.        ])