import numpy as np


# 生成数据集
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak',
                       'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字, 0 代表正常
    return postingList, classVec


# 获取单词具有唯一性的列表
def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


# 出现单词标注为1
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # p0Num = np.ones(numWords)
    # p1Num = np.ones(numWords)  # change to ones()
    # p0Denom = 2.0
    # p1Denom = 2.0  # change to 2.0
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)  # change to ones()
    p0Denom = 0.0
    p1Denom = 0.0  # change to 2.0
    for i in range(numTrainDocs):
        print(trainCategory[i], i)
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
            print("p1", p1Denom)
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            print("p0", p0Denom)
    # p1Vect = np.log(p1Num/p1Denom)  # change to log()
    # p0Vect = np.log(p0Num/p0Denom)  # change to log()
    p1Vect = p1Num/p1Denom  # change to log()
    p0Vect = p0Num/p0Denom  # change to log()
    return p0Vect, p1Vect, pAbusive


if __name__ == '__main__':
    list_posts, list_class = loadDataSet()
    # print("list_posts:", list_posts)
    print("list_class:", list_class)
    my_vec_list = createVocabList(list_posts)
    # print("vec_list:", my_vec_list)
    train_mat = []
    for post_doc in list_posts:
        train_mat.append(setOfWords2Vec(my_vec_list, post_doc))
    print("trainmat::", len(train_mat[0]))

    p0v, p1v, pab = trainNB0(train_mat, list_class)
    print(p0v)
    print(p1v)
    print(pab)
