#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/11/20

B = [[1.12366,2,3,5],
     [2,3.325423,3,5],
     [1,2,5.666611,1]]
A = [[1,2,3],
     [4,5,6]]
def shape(M):
    if len(M) == 0:
        return 0,0
    else:
        return len(M),len(M[0])

def matxRound(M, decPts=4):
    if len(M) == 0:
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

# 计算矩阵乘法 AB，如果无法相乘则raise ValueError
def matxMultiply(A, B):
    row_a, clo_a = shape(A)
    row_b, clo_b = shape(B)
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

# N = transpose(A)
# print(N)
# matxRound(B)
# print(B)
A = [
    [7, 5, 3, -5],
    [-4, 6, 2, -2],
    [-9, 4, -5, 9],
    [-9, -10, 5, -4],
 ]
def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    if len(A) != len(b):
        raise ValueError
    elif len(A) != len(A[0]):
        raise ValueError
    else:
        Ab = augmentMatrix(A, b)
        print("start---->", Ab)
        matxRound(Ab, decPts)
        num_row, num_clo = shape(Ab)
        for c in range(num_clo-1):
            current_max = abs(Ab[c][c])
            print("初始值", current_max,c)
            current_row = c
            for r in range(c, num_row):
                if abs(Ab[r][c]) > current_max:
                    print("{}---{}".format(abs(Ab[r][c]), current_max))
                    current_max = abs(Ab[r][c])
                    current_row = r
                    print(current_max)
            if current_max == 0:
                print("----",Ab)
                print(current_max,current_row,c)
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
        print("结果：",res)
        return res

A = [
    [-2, -8, 9, -1, -1, -4, -6, -2, 0],
    [-2, 9, -2, -9, 2, -1, 7, 6, 1],
    [-2, 2, 9, -1, -4, 4, -5, -10, 6],
    [-3, 0, -4, -5, -9, 0, -10, 2, 5],
    [-4, 9, -9, 1, -2, 4, 2, 0, 0],
    [8, 4, 2, -4, 2, -4, -8, -9, 4],
    [4, -10, 1, -9, -4, 3, 1, 7, 7],
    [-7, 7, 5, -6, 2, -9, 1, -2, -6],
    [-4, -3, 6, 2, -10, 0, -9, -8, 6]]

b = [[0],[1],[2],[3],[4],[5],[6],[7],[8]]
print(gj_Solve(A, b))
# r = matxMultiply(X,Y)
# print(r)
# print(augmentMatrix(Y,b))
# # swapRows(Y,0,1)
# # print(Y)
# scaleRow(Y, 0, 2)
# print(Y)



