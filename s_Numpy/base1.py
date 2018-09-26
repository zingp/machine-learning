#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/8/31


"""
ndim：维度
shape：行数和列数
size：元素个数
"""
import numpy as np

li = [
    [1, 2, -4],
    [-2, 2, 1],
    [-3, 4, -2],
]

# dtype是指定将矩阵中的元素转换成什么类型:float, int, str
matrix = np.array(li, dtype=np.str)
print("matrix:", matrix)
print("matrix type:", type(matrix))   # <class 'numpy.ndarray'>  numpy的n维数组

print("matrix 维度::", matrix.ndim)
print("matrix 行列::", matrix.shape)
print("matrix 元素个数::", matrix.size)

print(matrix[0], matrix[0][0], type(matrix[0][0]))


# 创建0矩阵
zero = np.zeros([3, 4], dtype=np.int)
print("0矩阵：zero:", zero)

# 创建全1矩阵
one = np.ones([2, 4], dtype=int)
print(one)

# 创建全空数组，每个数都接近0
empty = np.empty([3, 3], )
print(empty)

# 创建连续数组
a = np.arange(1, 14, 2)
print("连续数组", a)

# 改变数组形状
b = np.arange(12).reshape((3, 4))
print(b)

# 生成线段型数据
# [1, 10]之间生成20个数据
line_array = np.linspace(1, 10, 20)
print(line_array)

# 线段型数据也可以改变形状
print(np.linspace(1, 10, 20).reshape(2, 10))
