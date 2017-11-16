#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/11/15
import math
from decimal import Decimal, getcontext

# Decimal 转换为小数，防止浮点数反三角时失去精度
getcontext().prec = 30


class Vector(object):

    CANNOT_NORMALIZE_ZERO_VECTOR_MSG = 'Cannot normalize the zero vector'

    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple([Decimal(x) for x in coordinates])  # coordinates 坐标
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
        res = [Decimal(n)*x for x in self.coordinates]
        return Vector(res)

    def magnitude(self):
        """向量大小"""
        # magnitude 大小
        # from functools import reduce
        # square_sum = reduce(lambda x, y: x+y, [i**2 for i in self.coordinates])
        square_sum = sum([i**2 for i in self.coordinates])
        return Decimal(math.sqrt(square_sum))

    def normalized(self):
        """求单位向量"""
        # normalized 规格化
        try:
            return self.times_scalar(Decimal('1.0') / self.magnitude())
        except ZeroDivisionError:
            raise Exception(self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG)

    def dot(self, v):
        """点积"""
        square_list = [x*y for x, y in zip(self.coordinates, v.coordinates)]
        return sum(square_list)

    def angle_with(self, v, in_degrees=False):
        # angle 角；degrees 度数, 默认弧度
        try:
            u1 = self.normalized()
            u2 = v.normalized()
            angle_in_radians = math.acos(u1.dot(u2))

            if in_degrees:
                angle_per_radians = 180. / math.pi
                return angle_in_radians * angle_per_radians
            else:
                return angle_in_radians
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception('Cannot product an angle with the zero vector')
            else:
                raise e

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

    # 测试点积
    v = Vector([7.887, 4.138])
    w = Vector([-8.802, 6.776])
    print("v.w:", v.dot(w))

    v = Vector([-5.955, -4.904, -1.874])
    w = Vector([-4.496, -8.755, 7.103])
    print("v.w:", v.dot(w))

    v = Vector([3.183, -7.627])
    w = Vector([-2.668, 5.319])
    print("v w 夹角 弧度：", v.angle_with(w))

    v = Vector([7.35, 0.221, 5.188])
    w = Vector([2.751, 8.259, 3.985])
    print("v w 夹角 度数：", v.angle_with(w, in_degrees=True))

