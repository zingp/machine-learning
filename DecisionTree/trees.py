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

def create_date_set():
    """创造测试的数据集和分类"""
    data_set = [
        # [1, 1, 'maybe'],  #[feature1, feature2, class]
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

def split_data_set(data_set, axis, value):
    """按照给定特征划分数据集"""
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set

def choose_best_feature_to_split(data_set):
    """选择最好的数据集划分方式"""
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain, best_feature = 0.0, -1
    for i in range(num_features):
        # 数据集的每个向量的第i个特征的列表
        feat_list = [feat_vec[i] for feat_vec in data_set]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            # 这步没想明白
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

test_data_set, test_labels = create_date_set()
print(test_data_set)
entropy = calc_shannon_ent(test_data_set)
print("Entropy:", entropy)    # 熵越高，混合的数据集越多
print(split_data_set(test_data_set, 0, 1))
print(split_data_set(test_data_set, 0, 0))

# 计算第几个特征为最佳特征值
print(choose_best_feature_to_split(test_data_set))
