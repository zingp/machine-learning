# 计算L1， L2以及余弦相似度
import numpy as np


# L1距离
def l1dt(vec1, vec2):
    v1 = np.mat(vec1)
    v2 = np.mat(vec2)
    return np.sum(abs(v1-v2))


# L2距离
def l2dt(vec1, vec2):
    v1 = np.mat(vec1)
    v2 = np.mat(vec2)
    return np.sqrt((v1-v2)*(v1-v2).T)


## 余弦相似度
def cosdt(vec1, vec2):
    v1 = np.mat(vec1)
    v2 = np.mat(vec2)
    # v2是列向量
    return np.dot(v1,v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2))

if __name__ == "__main__":
    # v1 = [12, 34, 5, 6, 17, 8]
    # v2 = [21, 3,114, 12, 4, 5]
    # v1 = [100, 50]
    # v2 = [1000, 500]
    v1 = [2, 10]
    v2 = [10, 50]
    print(l1dt(v1, v2))
    print(l2dt(v1, v2))
    print(cosdt(v1, v2))