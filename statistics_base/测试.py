#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/8/28
import math

def get_sd(a):
    s = sum(a)
    agv_d = float(s) / len(a)
    print(agv_d)
    m = 0
    for i in a:
        m += (i - agv_d)**2
    sd = math.sqrt(float(m) / (len(a)-1))
    e = math.sqrt(len(a))
    res = sd /e
    print(res)


a = [2, 4, 10, 12, 16, 15, 4, 27, 9, -1, 15]

get_sd(a)
# print(float(s) / 11)
