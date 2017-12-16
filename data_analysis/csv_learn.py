#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "Zing-p"
# Date: 2017/12/1

import csv

#get csv data set
def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

titanic_data = read_csv('titanic-data.csv')
print(titanic_data)
for i in range(len(titanic_data)):
    for k, v in titanic_data[i].items():
        print(k, v)
    print('-------------------------')
# PassengerId 1     乘客序号
# Survived 0        是否生还： 0否，1是
# Pclass 3          船票类别：1
# Name Braund, Mr. Owen Harris   姓名
# Sex male          性别
# Age 22            年龄
# SibSp 1           同乘的兄弟姐妹人数
# Parch 0           同乘的父母孩子人数
# Ticket A/5 21171  船票号码
# Fare 7.25         票价
# Cabin             舱位
# Embarked S        上船地点：C = Cherbourg, Q = Queenstown, S = Southampton
                    # C = 瑟堡港，Q =皇后镇, S=南安普顿

