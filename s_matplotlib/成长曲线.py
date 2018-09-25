#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/11/3

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = np.arange(1, 50, 0.5)
    y1 = [10*(1+0.08)**a for a in x]
    plt.plot(x, y1, linewidth=2, color='#007500', label='chengzhang')

    y2 = [40 for i in x]
    plt.plot(x, y2, linewidth=2, color='#9F35FF', label='cengben')

    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
