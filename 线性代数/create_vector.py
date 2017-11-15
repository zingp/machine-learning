#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/11/15

class Vector(object):
    # coordinates 坐标
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates)
            self.dimension = len(coordinates)   # 维度
        except ValueError:
            raise ValueError('The coordinate must be nonempty.')
        except TypeError:
            raise TypeError('The coordinate must be iterable.')

    def __str__(self):
        return "Vector: {}".format(self.coordinates)

    def __eq__(self, v):
        return self.coordinates == v.coordinates

my_vector = Vector([1, 2, 3])
print(my_vector)
my_vector2 = Vector([1, 2, 3])
my_vector3 = Vector([-1, 2, 3])
print(my_vector == my_vector3)