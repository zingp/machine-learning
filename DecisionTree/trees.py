#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/11/15

from math import log
# 首先明确data_set的数据结构
'''
dataset = [
    [1, 1, 'yes'],
    [1, 0, 'no'],
    [0, 1, 'no'],
    [0, 1, 'yes'],
    [0, 1, 'no],
]
'''

# 计算给定数据集的香农熵
# 思路是先统计每个label出现的次数，算出概率，然后按照香浓熵公式计算。
def calc_shannon_ent(data_set):
    # 数据集长度
    num_entries = len(data_set)
    # 用于统计每个label的个数
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

# 创造测试的数据集和分类
def create_date_set():
    data_set = [
        # [1, 1, 'maybe'],  #[feature1, feature2, class]
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
        # [0, 1, 'nn'],
        # [0, 1, 'nx'],
        # [0, 1, 'gg'],
    ]
    labels = ['no surfacing', 'flippers']     # filppers 是 脚蹼的意思
    return data_set, labels

# 按照给定特征划分数据集
def split_data_set(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set

# 选择最好的数据集划分方式
# 逻辑是： 分别计算按照每一个特征【label】划分后的数据集的信息熵，选出信息熵最小的对应特征
def choose_best_feature_to_split(data_set):
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
            # 子数据集站总数据集概率*子数据集香浓熵 求和
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


# 当数据集已经处理了所有label，但是class 仍然不是唯一的，就只有按照概率高的来分类了
# ["yes", "no", "yes"] 以概率高德为分类标准
def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    # 按照维度为1排序，降序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # print(sorted_class_count)
    return sorted_class_count[0][0]

# 递归创建决策树：就是一个嵌套字典
def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(data_set):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)
    best_label = labels[best_feat]
    del(labels[best_feat])
    my_trees = {best_label: {}}
    feat_value = [e[best_feat] for e in data_set]
    uniq_val = set(feat_value)
    for v in uniq_val:
        sublabels = labels[:]
        my_trees[best_label][v] = create_tree(split_data_set(data_set, best_feat, v), sublabels)
    return my_trees


'''
test_data_set, test_labels = create_date_set()
print(test_data_set)
entropy = calc_shannon_ent(test_data_set)
print("Entropy:", entropy)    # 熵越高，混合的数据集越多
print(split_data_set(test_data_set, 0, 1))
print(split_data_set(test_data_set, 0, 0))

# 计算第几个特征为最佳特征值
print(choose_best_feature_to_split(test_data_set))
'''

if __name__ == '__main__':
    dataset, labels = create_date_set()
    shannon_ent = calc_shannon_ent(dataset)
    print("香浓熵：", shannon_ent)

    tree = create_tree(dataset, labels)
    print(tree)
