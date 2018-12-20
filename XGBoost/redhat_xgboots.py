#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import random
from operator import itemgetter
import time
pd.set_option('display.max_columns', 90) # 用于列显示不全的情况
random.seed(0)

def intersect(a, b):
    return list(set(a) & set(b))


def get_features(train, test):
    # 取出数据集列名
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    # 取交集的作用是取出共同特征，如果不是共同拥有的特征，没法预测
    output = intersect(trainval, testval)
    output.remove('people_id')
    output.remove('activity_id')
    return sorted(output)


def data_preprocessing():
    # 按列指定数据类型
    print("Load act_train.csv...")
    train = pd.read_csv('act_train.csv',
                        dtype={'people_id': np.str,   # 默认是float会出现小数
                            'activity_id':np.str,
                            'outcome': np.int8},
                        parse_dates=['date'])         # 自主设置列数据格式，包含时间用parse_dates进行解析
    print("Load act_test.csv...")
    test = pd.read_csv("act_test.csv",
                        dtype={'people_id': np.str,
                                'activity_id': np.str},
                        parse_dates=['date'])
    print("Load people.csv...")
    people = pd.read_csv("people.csv",
                        dtype={'people_id': np.str,
                                'activity_id': np.str,
                                'char_38': np.int32},  # 注意到char_38时数值类型
                        parse_dates=['date'])

    # dadaframe 最好不要单个值操作，而要分块如某列、某行这样操作
    for table in [train, test]:
        table['year'] = table['date'].dt.year
        table['month'] = table['date'].dt.month
        table['day'] = table['date'].dt.day
        table.drop('date', axis=1, inplace=True)
        # axis按列操作，inplace=true直接替换原来数组，默认False生成新数组
        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)
        for i in range(1, 11):
            table['char_'+str(i)].fillna('type -999', inplace=True,)  # -999 在xgboost视为空值
            table['char_'+str(i)] = table['char_'+str(i)].str.lstrip('type ').astype(np.int32)  

    # 处理people
    people['year'] = people['date'].dt.year
    people['month'] = people['date'].dt.month
    people['day'] = people['date'].dt.day
    people.drop('date', axis=1, inplace=True)
    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
    for i in range(1, 10):
        people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    for i in range(10, 38):
        people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)  # 强制转换

    # merge数据 填充空值
    train = pd.merge(train, people, how='left', on='people_id', left_index=True)
    train.fillna(-999, inplace=True)
    test = pd.merge(test, people, how='left', on='people_id', left_index=True)
    test.fillna(-999, inplace=True)

    features = get_features(train, test)
    return train, test, features


def train_adjust(train, features):
    target = 'outcome'
    random_state = 0  # 设定一个随机数
    eta = 0.6         # 学习率
    max_depth = 5     # 最大学习深度，如果过拟合，也就是模型复杂度复杂了，应该调小一些
    subsample = 0.5   # 子采样(chouyang)
    colsample_bytree = 1    # 列采样
    start_time = time.time()

    print("XGBoost trian start...")
    params = {
        "objective": "binary:logistic",   # 目标二分类
        "booster": "gbtree",
        "eval_metric": "auc",  #  评价方式AUC
        "eta": eta,
        "tree_method": "exact",   # 构造树时候的方法，exact是一个基本方法（贪心遍历），数据量不大可采用。
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,    # 静默的输出，不希望输出设为1
        "seed": random_state,
    }
    num_boost_round = 1000     # 最重要的参数轮次，需要迭代多少轮次，有多少颗树
    early_stopping_rounds = 10   # 如果测试集10次都不提高，就停止；也就是训练误差开始上升时结束
    test_size = 0.5

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)  # 必须转换成xgboost数组
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    end_time = time.time()
    print("cost_time:", end_time - start_time)
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration+1)
    score = roc_auc_score(X_valid[target].values, check)
    print('Check error value: {:.6f}'.format(score))

def train_all_data(train, features):
    target = 'outcome'
    random_state = 0
    eta = 0.3
    max_depth = 8
    subsample = 0.5
    colsample_bytree = 1
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 465
    early_stopping_rounds = 10
    test_size = 0.5

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    # dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    model_name = "redhat_bussiness_{}.model".format(time.strftime("%Y%m%d%H%M"))
    gbm.save_model(model_name)
    end_time = time.time()
    print("Cost:", end_time - start_time)
    print("create submission:", model_name)

def create_submission(score, test, prediction):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('activity_id,outcome\n')
    total = 0
    for id in test['activity_id']:
        str1 = str(id) + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def gene_submission(test, features, model_name):
    gbm = xgb.Booster(model_file=model_name)
    score = '0.99'
    test_prediction = gbm.predict(xgb.DMatrix(test[features]))
    create_submission(score, test, test_prediction)

if __name__ == "__main__":
    ## 读取数据  训练模型
    train, test, features = data_preprocessing()
    train_adjust(train, features)