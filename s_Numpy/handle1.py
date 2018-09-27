#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/9/19
import numpy as np

li = [1, 2, 3, 4]
a = np.array(li)
print(a, type(a))

print(a + 1)  # 广播 [2 3 4 5]

# 数组对应元素相加
b = np.array([2, 3, 4, 5])
print(a + b)       # [3 5 7 9]

# 对应元素相乘
print(a * b)       # [ 2  6 12 20]

# 对应元素乘方
print(a ** b)

# 访问
print(a[2])

arr = np.arange(0, 30, 2)    # [0, 30),差值为2的等差数列 array
print(arr)
