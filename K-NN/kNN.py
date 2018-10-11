#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/9/11
"""
功能：将数据分类
k-邻近算法的伪代码：
1.计算已知类别数据中的每个点与当前点之间的距离；
2.按照距离递增次序排序；
3.选取与当前点距离最小的K个点；
4.确定k个点所在的类别出现的频率；
5.返回前k个点出现频率最高的类别作为当前点的预测分类；
"""

import numpy as np
import operator


def create_data_set():
    groups = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return groups, labels

def k_class(point, data_set, labels, k):
    """判断新输入的点属于哪个分类。"""

    # 计算距离
    data_set_size = data_set.shape[0]
    # np.tile([], shape)按照shape复制[]
    diff_data_set = np.tile(point, [data_set_size, 1]) - data_set
    square_diff = diff_data_set ** 2
    square_sum = square_diff.sum(axis=1)   # 矩阵行相加
    distance = square_sum ** 0.5
    sort_distance = np.argsort(distance)  # 返回的是数组值从小到大的索引值

    # 选择距离最小的k个点
    class_count = {}
    for i in range(k):
        vote_label = labels[sort_distance[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # 排序，并得到列表[('B', 2), ('A', 2)]
    sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sort_class_count[0][0]


if __name__ == '__main__':

    group, label = create_data_set()
    print(k_class([0.2, 0.2], group, label, 2))