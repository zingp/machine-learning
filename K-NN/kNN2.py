#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/9/11
"""
可视化data_set2.txt中的数据
"""

import numpy as np
import matplotlib.pyplot as plt
from kNN import k_class


def file_to_matrix(filename):
    """将文件中的数据转换为数组和分类标签"""
    f = open(filename, "r")
    lines = f.readlines()
    line_sum = len(lines)

    ret_matrix = np.zeros((line_sum, 3))    # 单个数组
    class_label_vector = []                 # 分类
    index = 0
    for line in lines:
        line = line.strip()
        line_list = line.split("\t")

        ret_matrix[index, :] = line_list[0: 3]
        class_label_vector.append(int(line_list[-1]))
        index += 1

    return ret_matrix, class_label_vector


def scatter_plot(data, label):
    """生成散点图来可视化数据"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], 15.0 *
               np.array(label), 15.0*np.array(label))
    plt.show()


def auto_norm(data):
    """数据归一化函数"""
    min_val = data.min(0)    # 0 应该是按照"列"计算
    max_val = data.max(0)

    ranges = max_val - min_val
    # norm_data_set = np.zeros(np.shape(data))
    m = data.shape[0]
    norm_data_set = data - np.tile(min_val, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))

    return norm_data_set, ranges, min_val


def error_rate_test():
    """
    测试分类算法的错误率，通常0.9的数据作为训练样本，0.1的数据作为测试。
    """

    test_ratio = 0.10   # 测试比值，测试数据占所有数据的比值
    data_set, data_label = file_to_matrix("data_set2.txt")
    norm_set, ranges, min_val = auto_norm(data_set)
    m = norm_set.shape[0]
    test_vector_num = int(m*test_ratio)       # 测试数据的大小

    error_count = 0.0
    for i in range(test_vector_num):
        result = k_class(norm_set[i, :], norm_set[test_vector_num: m, :],
                         data_label[test_vector_num: m], 4)
        if result != data_label[i]:
            print("Test result is:{}, real answer is: {}".format(
                result, data_label[i]))
            error_count += 1.0
    print("The error rate is: %f" % (error_count / float(test_vector_num)))


if __name__ == '__main__':

    # data_set, data_label = file_to_matrix("data_set2.txt")
    # norm_data = auto_norm(data_set)         # 归一化之前的散点图可视化
    # scatter_plot(data_set, data_label)
    # scatter_plot(norm_data, data_label)     # 归一化之后的散点图可视化

    error_rate_test()
