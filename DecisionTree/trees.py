#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/11/15

from math import log

def calc_shannon_ent(data_set):
    """计算给定数据集的香农熵"""
    num_entries = len(data_set)
    label_counts = {}
    # feat_vec 特征向量
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        # prob 分类概率
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent
