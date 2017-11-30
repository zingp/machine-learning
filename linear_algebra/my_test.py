#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/11/16

import math
from functools import reduce
# a = [1, 2, 3, ]
# b = [4, 5, 6, ]
# for x, y in zip(a, b):
#     print(x, y)
# print(math.sqrt(9.9))
# print(reduce(lambda x, y: x+y, [i**2 for i in a]))


# from decimal import Decimal, getcontext
#
# getcontext().prec = 30
#
#
# print(['0']*2)
# for i in range(5):
#     print(i)
# for i in range(6,10):
# from fractions import Fraction
# # print(-107*623)
# print(-107*35)
#
# a = Fraction('-66661/152')
# b = Fraction(-384.0)
# print(a+b)
# c = Fraction('-3745/152')
# print(c+Fraction(181.0))
# print(Fraction('-125029/23767'))
M = [
    [1, 2],
    [3, 4]
]
def transpose(M):
    return [list(col) for col in zip(*M)]
print(*M)
for i in zip(*M):
    print(i,type(i))