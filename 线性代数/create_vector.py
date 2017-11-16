#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/11/15
import math

class Vector(object):

    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates)  # coordinates 坐标
            self.dimension = len(coordinates)      # 维度
        except ValueError:
            raise ValueError('The coordinate must be nonempty')
        except TypeError:
            raise TypeError('The coordinate must be iterable')

    def plus(self, v):
        """向量加法"""
        res = [x+y for x, y in zip(self.coordinates, v.coordinates)]
        return Vector(res)

    def minus(self, v):
        """向量减法"""
        res = [x-y for x, y in zip(self.coordinates, v.coordinates)]
        return Vector(res)

    def times_scalar(self, n):
        """标量×向量"""
        res = [n*x for x in self.coordinates]
        return Vector(res)

    def magnitude(self):
        """向量大小"""    # magnitude 大小
        # from functools import reduce
        # square_sum = reduce(lambda x, y: x+y, [i**2 for i in self.coordinates])
        square_sum = sum([i**2 for i in self.coordinates])
        return math.sqrt(square_sum)

    def normalized(self):
        """求单位向量"""
        # normalized 规格化
        try:
            return self.times_scalar(1 / self.magnitude())
        except ZeroDivisionError:
            raise Exception('Cannot normalize the zero vector')

    def __str__(self):
        return "Vector: {}".format(self.coordinates)

    def __eq__(self, v):
        return self.coordinates == v.coordinates

if __name__ == '__main__':

    my_vector = Vector([1, 2, 3])
    # print(my_vector)
    my_vector2 = Vector([1, 2, 3])
    my_vector3 = Vector([-1, 2, 3])
    # print(my_vector == my_vector3)
    # 测试向量相加
    vector1 = Vector([8.218, -9.341])
    vector2 = Vector([-1.129, 2.111])
    print(vector1.plus(vector2))

    # 测试向量相减
    vector3 = Vector([7.119, 8.215])
    vector4 = Vector([-8.223, 0.878])
    print(vector3.minus(vector4))
    # 测试标量×向量
    vector5 = Vector([1.671, -1.012, 0.318])
    print(vector5.times_scalar(7.41))

    # 测试向量大小、单位向量
    vector6 = Vector([-1, 1, 1])
    print(vector6.magnitude())
    print(vector6.normalized().magnitude())

    vector7 = Vector([-0.221, 7.437])
    print(vector7.magnitude())
    vector8 = Vector([8.813, -1.331, -6.247])
    print(vector8.magnitude())

    vector9 = Vector([5.581, -2.136])
    print(vector9.normalized())
    vector10 = Vector([1.996, 3.108, -4.554])
    print(vector10.normalized())

