#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/9/14
"""
不使用numpy创建单位矩阵。
"""

class UnitMatrix(object):

    def __init__(self, num):
        self.row_col = num
        self.shape = [num, num]

    def __str__(self):
        matrix = []
        for i in range(self.row_col):
            row_list = [1 if j == i else 0 for j in range(self.row_col)]
            matrix.append(row_list)

        res = ""
        for k in matrix:
            if k == matrix[0]:
                res += "{},\n".format(k)
            elif k == matrix[-1]:
                res += " {}".format(k)
            else:
                res += " {},\n".format(k)

        return """[{}]""".format(res)

obj = UnitMatrix(4)
print(obj)
print(obj.shape)
