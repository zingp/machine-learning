import numpy as np
from time import sleep
import random


def load_data_set(filename):
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(linea_rr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    fr.close()
    return data_mat, label_mat


def selectJrand(i, m):
    j = i  # we want to select any J not equal to i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T *
                        (dataMatrix*dataMatrix[i, :].T)) + b
            # if checks if an example violates KKT conditions
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T *
                            (dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0 * dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[i,
                                                                             :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                # update i by the same amount as j
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                # the update is in the oppostie direction
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[i,
                                                                                            :].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[j,
                                                                                            :].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j, :]*dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" %
                      (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


if __name__ == '__main__':
    data_arr, label_arr = load_data_set('testSet.txt')
    print(data_arr, label_arr)
    # b, al = smoSimple(data_arr, label_arr, 0.6, 0.001, 40)
    # print(b)
    for i in range(100):
        if al[i] > 0:
            print("alpha:", al[i])
            print(data_arr[i], "分类：", label_arr[i])
