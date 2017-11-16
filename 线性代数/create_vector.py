#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/11/15

class Vector(object):

    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates) # coordinates 坐标
            self.dimension = len(coordinates)   # 维度
        except ValueError:
            raise ValueError('The coordinate must be nonempty')
        except TypeError:
            raise TypeError('The coordinate must be iterable')

    def plus(self, v):
        res = [x+y for x, y in zip(self.coordinates, v.coordinates)]
        return Vector(res)

    def minus(self, v):
        res = [x-y for x, y in zip(self.coordinates, v.coordinates)]
        return Vector(res)

    def scalar_multiply(self, n):
        res = [n*x for x in self.coordinates]
        return Vector(res)

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

    vector1 = Vector([8.218, -9.341])
    vector2 = Vector([-1.129, 2.111])
    print(vector1.plus(vector2))

    vector3 = Vector([7.119, 8.215])
    vector4 = Vector([-8.223, 0.878])
    print(vector3.minus(vector4))

    vector5 = Vector([1.671, -1.012, 0.318])
    print(vector5.scalar_multiply(7.41))
