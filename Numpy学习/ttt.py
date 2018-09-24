#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/12/11

import math

# 向上取整
# print("math.ceil---")
# print("math.ceil(2.3) => ", math.ceil(2.3))
# print("math.ceil(2.6) => ", math.ceil(2.6))
#
# # 向下取整
# print("\nmath.floor---")
# print("math.floor(2.3) => ", math.floor(2.3))
# print("math.floor(2.6) => ", math.floor(2.6))
dic = {"a":"b", "c":"d"}
li = [1,2,3]
# print(**dic)

a = [
    "123",
    #"456",
]
print(a)

# 常量
# math.e
# math.pi

math.ceil(2.3)  # 向上取整 3
math.floor(2.3) # 向下取整 2
math.sqrt(4)    # 求平方根,返回浮点数
math.factorial(5)  # 求阶乘
math.log(100, 10)  # 以10为底，100的对数，如果不给10，默认为e
math.log10(1000)   # 以10为底1000的对数
math.pow(2, 3)     # 返回2 ** 3
math.fabs(-1.3456) # 返回浮点数的绝对值

# 角度和弧度的转换
math.degrees(math.pi/2)     # 弧度转为角度
math.radians(90)     # 角度转弧度

# 三角函数
math.sin(math.pi/2)        # 返回x的正弦，x 为弧度
math.cos(math.pi/3)        # 返回余弦
math.tan(math.pi/4)        # 返回正切