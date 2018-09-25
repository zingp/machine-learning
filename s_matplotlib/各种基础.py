#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/9/25

import matplotlib.pyplot as plt
import numpy as np

# 使用np.linspace定义x：范围是(-3,3);个数是50. 仿真一维数据组(x ,y)表示曲线1.
x = np.linspace(-3, 3, 50)
y = x**2
y2 = 2*x + 1

# 使用plt.figure定义一个图像窗口. 使用plt.plot画(x ,y)曲线. 使用plt.show显示图像.
# plt.figure()
# 使用plt.figure定义一个图像窗口：编号为3；大小为(8, 5). 使用plt.plot画(x ,y2)曲线.
# 使用plt.plot画(x ,y)曲线，曲线的颜色属性(color)为红色;曲线的宽度(linewidth)为1.0；
# 曲线的类型(linestyle)为虚线. 使用plt.show显示图像.
plt.figure(num=1, figsize=(8, 5),)
plt.plot(x, y2)
plt.plot(x, y, color='red', linewidth=1.0, linestyle='--')
# plt.plot(x, y)
# 使用plt.xlim设置x坐标轴范围：(-1, 2)；
# 使用plt.ylim设置y坐标轴范围：(-2, 3)；
# 使用plt.xlabel设置x坐标轴名称：’x’；
# 使用plt.ylabel设置y坐标轴名称：’y’；
plt.xlim((-1, 2))
plt.ylim((-2, 3))
plt.xlabel('x')
plt.ylabel('y')
plt.show()