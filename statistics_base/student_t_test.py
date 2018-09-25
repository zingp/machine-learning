#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "LiuYouYuan"
# Date: 2017/8/23

import math


class SampleInfo(object):
    """计算样本基本统计量"""

    def __init__(self, array):
        self.array = [float(i) for i in array]
        self.counts = self.count()
        self.sum = sum(self.array)
        self.avg = self.average()

    def count(self):
        """求样本量"""
        return len(self.array)

    def f(self):
        """自由度"""
        return self.counts - 1

    def average(self):
        """求样本平均数"""
        if self.counts < 1:
            return 0
        else:
            return self.sum / self.counts

    def median(self):
        """求中位数"""
        if self.counts < 1:
            return None
        else:
            self.array.sort()
            return self.array[len(self.array) // 2]

    def variance(self):
        """求样本方差"""
        if self.counts < 1:
            return None
        else:
            li = [(k-self.avg)**2 for k in self.array]
            s = sum(li)
            return s / (self.counts - 1)

    def standard_dev(self):
        """求样本标准差"""
        return math.sqrt(self.variance())


class DiffArrayInfo(object):
    """获得样本差值序列"""

    def __init__(self, a, b):
        self.a = [i for i in a]
        self.b = [i for i in b]
        self.array = self.get_diff_array()

    def get_diff_array(self):
        """返回样本差值序列"""
        array = list()
        if len(self.a) == len(self.b):
            array = [self.a[i] - self.b[i] for i in range(len(self.a))]
        return array


def get_t(u_d, s_d, n):
    """
    计算双边t检验的统计量t；
    u_d:样本差值的均值；
    s_d:样本差值的标准差；
    n:样本量
    return:t值
    """
    return float(u_d) / (float(s_d) / math.sqrt(n))


def get_r(t, f):
    r = t**2 / (t**2 + f)
    return r


if __name__ == '__main__':

    Congruent = [
        12.079, 16.791, 9.564, 8.63, 14.669, 12.238, 14.692, 8.987, 9.401, 14.48, 22.328, 15.298,
        15.073, 16.929, 18.2, 12.13, 18.495, 10.639, 11.344, 12.369, 12.944, 14.233, 19.71, 16.004,
    ]
    In_congruent = [
        19.278, 18.741, 21.214, 15.687, 22.803, 20.878, 24.572, 17.394, 20.762, 26.282, 24.524, 18.644,
        17.51, 20.33, 35.255, 22.158, 25.139, 20.429, 17.425, 34.288, 23.894, 17.96, 22.058, 21.157,
    ]

    obj_a = SampleInfo(Congruent)
    obj_b = SampleInfo(In_congruent)
    obj_diff = DiffArrayInfo(In_congruent, Congruent)
    obj_c = SampleInfo(obj_diff.array)

    s3 = obj_c.standard_dev()
    t_val = get_t(obj_c.avg, s3, obj_c.count)
    r_val = get_r(t_val, obj_c.f())

    print("一致条件的样本的标准差：", obj_a.standard_dev())
    print("不一致条件的样本的标准差：", obj_b.standard_dev())
    print("样本差值的标准差:Sd = ", s3)
    print("t 统计量：t = ", t_val)
    print("r_2 效应量：r = ", r_val)

"""
一致条件的样本的标准差： 3.5593579576451955
不一致条件的样本的标准差： 4.797057122469138
样本差值的标准差:Sd =  4.864826910359054
t 统计量：t =  8.020706944109957
r_2 效应量：r =  0.736636416144506
"""
