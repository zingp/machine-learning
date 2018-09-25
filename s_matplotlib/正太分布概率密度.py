#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/9/19
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 加上这两句可以显示图像中文名称而不乱码
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']  #FangSong/黑体 FangSong/KaiTi
matplotlib.rcParams['axes.unicode_minus'] = False

mu = 0
sigma = 1
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 50)
y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
print(x.shape)
print('x = \n', x)
print(y.shape)
print('y = \n', y)
# plt.plot(x, y, 'ro-', linewidth=2)
# 'r-'红色的线；'go' 绿色的圆圈；先款2；圆圈大小6
plt.plot(x, y, 'r-', x, y, 'go', linewidth=2, markersize=6)
plt.grid(True)
plt.title("正态分布")     # 图像的名称
plt.savefig('1.png')    # 输出的图像存储
plt.show()
