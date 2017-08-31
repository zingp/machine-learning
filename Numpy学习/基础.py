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

matrix = np.array(li)
print("matrix:", matrix)
print("matrix 维度::", matrix.ndim)
print("matrix 行列::", matrix.shape)
print("matrix 元素个数::", matrix.size)
