#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/11/14

import numpy as np
import os
from kNN import k_class

def img2vector(filename):
    """循环读取文件前32行，每行的前32位字符串。转换为1行1024列的numpy数组"""
    vector = np.zeros((1, 1024))
    f = open(filename, "r")
    for i in range(32):
        line_str = f.readline()
        for j in range(32):
            vector[0, 32*i + j] = int(line_str[j])
    f.close()
    return vector


def hand_writing_class():

    # 读取trainingDigits目录下的文件，解析文件名，获取分类，添加到hw_labels
    hw_labels = []
    training_file_list = os.listdir("trainingDigits")
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name = training_file_list[i]
        file_prefix = file_name.split(".")[0]
        class_num_str = file_prefix.split("_")[0]
        hw_labels.append(class_num_str)
        # 得到原始参照集合
        training_mat[i, :] = img2vector("trainingDigits/{}".format(file_name))

    test_file_list = os.listdir("testDigits")
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name = test_file_list[i]
        file_prefix = file_name.split(".")[0]
        class_num_str = file_prefix.split("_")[0]
        vector_test = img2vector("testDigits/{}".format(file_name))

        result = k_class(vector_test, training_mat, hw_labels, 3)
        # print("result is {}, in fact is {}".format(result, class_num_str))
        if result != class_num_str:
            error_count += 1.0
    print("error count:", error_count)
    print("error rate:", error_count / float(m_test))


if __name__ == '__main__':

    hand_writing_class()
