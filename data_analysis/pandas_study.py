#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/12/16

import pandas as pd

# 10天中不同5个站的地铁乘客量，创建pandas的DataFrame
ridership_df = pd.DataFrame(
    data=[[0, 0, 2, 5, 0],
          [1478, 3877, 3674, 2328, 2539],
          [1613, 4088, 3991, 6461, 2691],
          [1560, 3392, 3826, 4787, 2613],
          [1608, 4802, 3932, 4477, 2705],
          [1576, 3933, 3909, 4979, 2685],
          [95, 229, 255, 496, 201],
          [2, 0, 1, 27, 0],
          [1438, 3785, 3589, 4174, 2215],
          [1342, 4043, 4009, 4665, 3033]],
    index=['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
           '05-06-11', '05-07-11', '05-08-11', '05-09-11', '05-10-11'],
    columns=['R003', 'R004', 'R005', 'R006', 'R007']
)

if 0:
    print(ridership_df)
    # 通过字典映射的方式创建dataFrame,单个列表实际就是列向量, 列向量必须等长
    df_1 = pd.DataFrame({'A': [0, 1, 2], 'B': [3, 4, 5]})
    print(df_1)

    # 也可以通过2维numpy数组创建，得指明列的名称
    df_2 = pd.DataFrame([[0, 1, 2], [3, 4, 5]], columns=['A', 'B', 'C'])
    print(df_2)


# 访问DataFrame中的元素
if 0:
    print(ridership_df.iloc[0])        # 访问第一行
    print(ridership_df.loc['05-05-11'])  # 通过访问某行
    print(ridership_df['R003'])          # 访问某列
    print(ridership_df.iloc[1, 3])


if 0:
    # 访问多行
    print(ridership_df.iloc[1:4])  # 访问1,2,3行
    # 访问多列
    print(ridership_df[['R003', 'R005']])

# Pandas axis（轴）
if 0:
    df = pd.DataFrame({'A': [0, 1, 2], 'B': [3, 4, 5]})
    print(df.sum())         # 分别求各列数据的和
    print(df.sum(axis=1))   # 分别求各行的和
    print(df.values.sum())  # 数据总和 .value是所有数据

if 0:
    max_station = ridership_df.iloc[0].argmax()    # 得到第0行，数字最大的那列列名
    print(ridership_df[max_station].mean())
    print(ridership_df.mean())    # DataFrame.mean()提供了每列的平均值

def mean_riders_for_max_station(ridership):
    '''
    Fill in this function to find the station with the maximum riders on the
    first day, then return the mean riders per day for that station. Also
    return the mean ridership overall for comparsion.

    This is the same as a previous exercise, but this time the
    input is a Pandas DataFrame rather than a 2D NumPy array.
    '''

    max_station = ridership_df.iloc[0].argmax()      # 得到第0行，数字最大的那列列名
    overall_mean = ridership_df[max_station].mean()  # 计算max_station这列的平均值
    mean_for_max = ridership_df.values.mean()        # 所有数据的平均值

    return (overall_mean, mean_for_max)

if 0:
    print(mean_riders_for_max_station(ridership_df))

titanic_df = pd.read_csv("titanic-data.csv")
# print(titanic_df.head())   # 打印前5行
print(titanic_df.describe())  # 整体数据的统计信息，平均值，标准差等

