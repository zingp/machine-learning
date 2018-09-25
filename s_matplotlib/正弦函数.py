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

x = np.linspace(-math.pi, math.pi, 50, endpoint=True)
y = np.sin(x)
# plt.plot(x, y, 'r--', x, y, 'go', linewidth=2, markersize=4, label='sin(x)')
# 'r-'实线；'r--'虚线
plt.plot(x, y, 'r-', linewidth=2, label='sin(x)')
plt.plot(x, y, 'bo', markersize=1, label='样本点')
plt.grid(True)  # 显示网格
# plt.legend(loc='upper right')  # 显示图例
plt.legend(loc=0)  # 显示图例
# 其中loc有很多参数
# 'best': 0,       自动分配图例到最佳位置
# 'upper right': 1,
# 'upper left': 2,
# 'lower left': 3,
# 'lower right': 4,
# 'right': 5,
# 'center left': 6,
# 'center right': 7,
# 'lower center': 8,
# 'upper center': 9,
# 'center': 10,
plt.show()
