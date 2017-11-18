#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/11/16

import math
from functools import reduce
a = [1, 2, 3, ]
b = [4, 5, 6, ]
# for x, y in zip(a, b):
#     print(x, y)
# print(math.sqrt(9.9))
# print(reduce(lambda x, y: x+y, [i**2 for i in a]))


from decimal import Decimal, getcontext

getcontext().prec = 30


print(['0']*2)