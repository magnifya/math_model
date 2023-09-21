import pulp as pp
# pp.LpMinimize其实就是一个int，值为1
# pp.LpMaximize也是一个int，值为-1.
# 在参数后面也可以直接写1或者-1表示
# 其实默认的都是求最小值，在求最大值的时候两边都要乘以-1，转成最小值求解。
# 						所以最小值参数为1，最大值参数为-1
mylp = pp.LpProblem("lp1",pp.LpMinimize)

# 定义未知数，标记取值范围，cat为限制条件，Integer表示整数型，Continuous表示连续型
x1 = pp.LpVariable("x1",lowBound=0,upBound=9,cat="Integer")
x2 = pp.LpVariable("x2",lowBound=0,cat="Integer")
x3 = pp.LpVariable("x3",lowBound=0,cat="Integer")

# 在pulp中用+=符号，加约束和目标函数
# 只支持 = ，>= ， <= 不支持> , <
mylp += 3*x1+4*x2+x3
mylp +=(x1+6*x2+2*x3 >=5)
mylp +=(2*x1 >=3)

i = mylp.solve()
print(i)

print('ret = %s' %pp.value(mylp.objective))

print('x1 = %s' %pp.value(x1))
print('x2 = %s' %pp.value(x2))
print('x3 = %s' %pp.value(x3))