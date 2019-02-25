

# 用梯度下降计算函数的极小值
def argminf(x1, x2):
    r = ((x1+x2-4)**2 + (2*x1+3*x2 - 7)**2 + (4*x1+x2-9)**2)*0.5
    return r


# 全量计算一阶偏导的值
def deriv_x(x1, x2):
    r1 = (x1+x2-4) + (2*x1+3*x2-7)*2 + (4*x1+x2-9)*4
    r2 = (x1+x2-4) + (2*x1+3*x2-7)*3 + (4*x1+x2-9)
    return r1, r2


# 梯度下降算法
def gradient_decs(n):
    alpha = 0.01
    x1, x2 = 0, 0
    y1 = argminf(x1, x2)
    for i in range(n):
        deriv1, deriv2 = deriv_x(x1, x2)
        x1 = x1 - alpha * deriv1
        x2 = x2 - alpha * deriv2
        y2 = argminf(x1, x2)
        if y1 - y2 < 1e-6:
            return x1, x2, y2
        if y2 < y1:
            y1 = y2
    return x1, x2, y2


# 随机计算一阶偏导的值
def rand_deriv(x1, x2):
    import random
    r1, r2 = 0, 0
    rand = random.randint(0, 2)
    if rand == 0:
        r1 = x1+x2-4
        r2 = x1+x2-4
    elif rand == 1:
        r1 = (2*x1+3*x2-7)*2
        r2 = (2*x1+3*x2-7)*3
    else:
        r1 = (4*x1+x2-9)*4
        r2 = (4*x1+x2-9)
    return r1, r2


# 随机梯度下降
def random_grad_decs(n):
    alpha = 0.01
    x1, x2 = 0, 0
    y1 = argminf(x1, x2)
    y2 = 0
    for i in range(n):
        deriv1, deriv2 = rand_deriv(x1, x2)
        x1 = x1 - alpha * deriv1
        x2 = x2 - alpha * deriv2
        y2 = argminf(x1, x2)
        # if y1 - y2 < 1e-8:
        #     print("ok")
        #     return x1, x2, y2
        if y2 < y1:
            y1 = y2
    return x1, x2, y2


# 梯度下降法
print(gradient_decs(1000))

# 随机梯度下降法
print(random_grad_decs(1000))

# 随机梯度下降计算十次求平均
x1, x2, y = 0, 0, 0
for i in range(10):
    r1, r2, r3 = random_grad_decs(1000)
    x1 += r1
    x2 += r2
    y += r3
print(x1/10, x2/10, y/10)
