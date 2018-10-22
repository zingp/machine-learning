from numpy import *


# 创建数据集 [[1.0, -0.017612, 14.053064],]  [0, 1, ...]
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # 转换为numpy矩阵
    labelMat = mat(classLabels).transpose()  # 转换为numpy矩阵,列向量
    m, n = shape(dataMatrix)    # 100行 3列
    print(m, n)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):  # heavy on matrix operations
        h = sigmoid(dataMatrix*weights)  # matrix mult
        error = (labelMat - h)  # vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    return weights


if __name__ == '__main__':
    data_set, label_mat = loadDataSet()
    w = gradAscent(data_set, label_mat)
    print(w)
