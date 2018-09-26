#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/9/19

import numpy as np

# arange()指定[start, end)以及步长，生成数组
a = np.arange(1, 10, 0.5)
print(a)

# linsapce()指定[start, end]以及元素个数，均匀分散，生成数组
b = np.linspace(1, 10, 10)
print(b)
c = np.linspace(1, 10, 10, endpoint=False)   # 不包含终止值[start, end)
print(c)  # 等差数列

# logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
# 指定[10**1， 10**2]，按照等比数列分为10段，组成数组。
log = np.logspace(1, 2, 12, endpoint=True)
print(log)
print(log[1]/log[0], log[9]/log[8])

st = "abcdef"
res = np.fromstring(st, dtype=np.int8)
print(res)

res2 = res.reshape(-1, 1)
print(res2)

# 二维数组按索引切片
log2 = log.reshape(3, 4)
print(log2)
print(log2[1:, [1, 2]])
print(log2[1:, 2:])