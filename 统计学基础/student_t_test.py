#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/8/23

import math

class ArrayAbc(object):

    def __init__(self, array):
        self.array = [float(i) for i in array]
        self.count = self.count()
        self.sum = sum(self.array)
        self.avg = self.average()

    def count(self):
        return len(self.array)

    def average(self):
        """求平均数"""
        if self.count < 1:
            return 0
        else:
            return self.sum / self.count

    def median(self):
        """求中位数"""
        if self.count < 1:
            return None
        else:
            self.array.sort()
            return self.array[len(self.array) // 2]

    def variance(self):
        """求方差"""
        if self.count < 1:
            return None
        else:
            li = [(k-self.avg)**2 for k in self.array]
            s = sum(li)
            return s / self.count


class TTest(object):
    """两个正太总体均值差的检验（t检验）。"""
    def __init__(self, n1, n2, x, y, s1, s2):
        self.n1 = float(n1)
        self.s1 = float(s1)
        self.x = float(x)
        self.y = float(y)
        self.n2 = float(n2)
        self.s2 = float(s2)

    def get_sw(self):
        ss = (self.n1 - 1) * self.s1 + (self.n2 - 1) * self.s2
        ss_avg = ss / (self.n1 + self.n2 - 2)
        return math.sqrt(ss_avg)

    def get_t(self):
        a = self.x - self.y
        w = self.get_sw()
        b = w * math.sqrt(1 / self.n1 + 1 / self.n2)
        return abs(a / b)

    def get_r(self):
        """效应量"""
        t = self.get_t()
        return t**2 / (t**2 + self.n1 - 1)


if __name__ == '__main__':

    Congruent = [
        12.079, 16.791, 9.564, 8.63, 14.669, 12.238, 14.692, 8.987, 9.401, 14.48, 22.328, 15.298,
        15.073, 16.929, 18.2, 12.13, 18.495, 10.639, 11.344, 12.369, 12.944, 14.233, 19.71, 16.004,
    ]
    In_congruent = [
        19.278, 18.741, 21.214, 15.687, 22.803, 20.878, 24.572, 17.394, 20.762, 26.282, 24.524, 18.644,
        17.51, 20.33, 35.255, 22.158, 25.139, 20.429, 17.425, 34.288, 23.894, 17.96, 22.058, 21.157,
    ]

    obj_a = ArrayAbc(Congruent)
    obj_b = ArrayAbc(In_congruent)

    n1 = obj_a.count
    n2 = obj_b.count
    x = obj_a.avg
    y = obj_b.avg
    s1 = obj_a.variance()
    s2 = obj_b.variance()

    obj_t = TTest(n1, n2, x, y, s1, s2)
    print("X平均：", x)
    print("Y平均：", y)
    print("均值差值：", y-x)
    print("X方差：", s1)
    print("Y方差：", s2)
    print("t统计值:", obj_t.get_t())
    print("效应量r:", obj_t.get_r())


"""
X平均： 14.051125
Y平均： 22.01591666666667
均值差值： 7.9647916666666685
X方差： 12.141152859375003
Y方差： 22.05293382638889
t统计值: 6.672745133475093
效应量r: 0.6593880742304228
"""
