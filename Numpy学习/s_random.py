#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2018/09/24
import random as R

R.random()           # 返回[0, 1)之间的随机浮点数
# 返回[0, 2**4-1]之间的随机长整数，2 的4次方；比如R.getrandbits(4)，返回[0.15]之前的随机长整数
R.getrandbits(4)
R.uniform(1, 10)            # 返回[1, 10)之间的随机浮点数
R.randrange(1, 100, 5)      # 返回[1， 100)之间每隔5的随机整数
R.choice("abcdef")          # 返回序列中的一个随机元素
R.shuffle([1, 2, 3, 4])        # 返回一个乱序序列
R.sample("abcdefghijklmn", 3)  # 从序列中返回n个随机且不重复的元素
