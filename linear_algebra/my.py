#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/11/20

def shape(M):
    """返回矩阵的行列"""
    if len(M) == 0:
        return 0, 0
    else:
        if M[0] is not list:
            return len(M), 1
        else:
            return len(M), len(M[0])


def matxRound(M, decPts=4):
    if not len(M):
        pass
    else:
        num_row,num_clo = shape(M)
        for r in range(num_row):
            for c in range(num_clo):
                M[r][c] = round(M[r][c], decPts)

# 计算矩阵的转置
def transpose(M):
    if len(M) == 0:
        return M
    else:
        Mt = []
        num_row, num_clo = shape(M)
        for x in range(num_clo):
            Mt.append([])
            for y in range(num_row):
                Mt[x].append(0)
        for x in range(num_row):
            for y in range(num_clo):
                Mt[y][x] = M[x][y]
        return Mt

def one_row_to_clo(M):
    ret = []
    for i in M:
        ret.append([i])
    return ret
# 计算矩阵乘法 AB，如果无法相乘则raise ValueError
def matxMultiply(A, B):
    row_a, clo_a = shape(A)
    row_b, clo_b = shape(B)
    if clo_b == 1:
        B = one_row_to_clo(B)
    if clo_a == row_b:
        res = []
        for i in range(row_a):
            res.append([])
            for j in range(clo_b):
                ele_sum = 0
                for s in range(clo_a):
                    ele_sum += A[i][s]*B[s][j]
                res[i].append(ele_sum)
        return res
    else:
        raise ValueError

# m, b = linearRegression(X, Y)

# 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    if len(A) != len(b):
        raise ValueError
    else:
        augment_mat = []
        for r in range(shape(A)[0]):
            augment_mat.append([])
            for c in range(shape(A)[1]):
                augment_mat[r].append(A[r][c])
            augment_mat[r].append(b[r][0])
        return augment_mat

# r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    if (0 <= r1 < len(M)) and (0 <= r2 < len(M)):
        M[r1], M[r2] = M[r2], M[r1]
    else:
        raise IndexError('list index out of range')

# r1 <--- r1 * scale
# scale为0是非法输入，要求 raise ValueError
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if not scale:
        raise ValueError('The parameter scale can not be zero')
    else:
        M[r] = [scale*i for i in M[r]]

# r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    if not scale:
        raise ValueError
    if (0 <= r1 < len(M)) and (0 <= r2 < len(M)):
        M[r1] = [M[r1][i] + scale * M[r2][i] for i in range(len(M[r2]))]
    else:
        raise IndexError('list index out of range')

 # 7,   5,   3,  -5 ||  1
 #  -4,   6,   2,  -2 ||  1
 #  -9,   4,  -5,   9 ||  1
 #  -9, -10,   5,  -4 ||  1
A = [
    [7,   5,   3,  -5],
    [-4,   6,   2,  -2],
    [-9,   4,  -5,   9],
    [-9, -10,   5,  -4]
]
b = [
    [1],
    [1],
    [1],
    [1]]
from fractions import Fraction
def to_fraction(M):
    for i in range(len(M)):
        for j in range(len(M[i])):
            M[i][j] = Fraction(float(M[i][j]))

# to_fraction(A)
# to_fraction(b)
# print(A)
# print(b)

def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    if len(A) != len(b):
        raise ValueError
    elif len(A) != len(A[0]):
        raise ValueError
    else:
        Ab = augmentMatrix(A, b)
        matxRound(Ab, decPts)
        num_row, num_clo = shape(Ab)
        for c in range(num_clo-1):
            current_max = 0.0
            current_row = c
            for r in range(c, num_row):
                if abs(Ab[r][c]) > current_max:
                    current_max = abs(Ab[r][c])
                    current_row = r
            if current_max < epsilon:
                return None
            else:
                swapRows(Ab, c, current_row)
                while abs((Ab[c][c]-1.0)) >= epsilon:
                    scaleRow(Ab, c, 1.0 / Ab[c][c])
                for j in range(c):
                    while abs(Ab[j][c]) >= epsilon:
                        addScaledRow(Ab, j, c, -Ab[j][c])
                for j in range(c + 1, num_row):
                    while abs(Ab[j][c]) >= epsilon:
                        addScaledRow(Ab, j, c, -Ab[j][c])
        res = []
        for row in range(num_row):
            res.append([Ab[row][-1]])
        return res
print(gj_Solve(A,b))

def calculateMSE(X,Y,m,b):
    if len(X) == len(Y) and len(X) != 0:
        n = len(X)
        square_li = [(Y[i]-m*X[i]-b)**2 for i in range(n)]
        return sum(square_li) / float(n)
    else:
        raise ValueError


def linearRegression(X, Y):
    mX = []
    for i in X:
        m = [i, 1]
        mX.append(m)
    X = mX

    y = []
    for i in Y:
        n = [i]
        y.append(n)
    Y = y
    XT = transpose(X)
    A = matxMultiply(XT, X)
    b = matxMultiply(XT, Y)
    ret = gj_Solve(A, b)
    return ret[0][0], ret[1][0]




