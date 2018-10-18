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


# 词集模型 ：出现单词标注为1
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# 词袋模型：每出现一次单词+1


def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)     # 评论条数
    numWords = len(trainMatrix[0])      # 总词汇量
    pAbusive = sum(trainCategory)/float(numTrainDocs)     # 整个文档出现侮辱性评论的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)  # change to ones()
    p0Denom = 2.0
    p1Denom = 2.0  # change to 2.0
    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)  # change to ones()
    # p0Denom = 0.0
    # p1Denom = 0.0  # change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)  # change to log()
    p0Vect = np.log(p0Num/p0Denom)  # change to log()
    # p1Vect = p1Num/p1Denom  # change to log()    # 给定评论是1情况下，该条评论中出现词汇表中单词的概率（有的单词没出现则为0）
    # p0Vect = p0Num/p0Denom  # change to log()    # 概率有一个为0，乘积均为0，所以转换为log
    return p0Vect, p1Vect, pAbusive


# 比较概率大小，返回概率大的类别
def classify_bayes(vec2classify, p0vec, p1vec, pclass1):
    p1 = sum(vec2classify * p1vec) + np.log(pclass1)
    p0 = sum(vec2classify * p0vec) + np.log(1-pclass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 测试函数
def testing_bayes():
    list_posts, list_class = loadDataSet()
    my_vec_list = createVocabList(list_posts)
    train_mat = []
    for post_doc in list_posts:
        train_mat.append(setOfWords2Vec(my_vec_list, post_doc))
    p0v, p1v, pab = trainNB0(train_mat, list_class)

    test_list = ['love', 'my', 'dalmation']
    doc_set_vec = np.array(setOfWords2Vec(my_vec_list, test_list))
    print(test_list, "classified as:",
          classify_bayes(doc_set_vec, p0v, p1v, pab))
    test_list = ['stupid', 'garbage', 'dalmation']
    doc_set_vec = np.array(setOfWords2Vec(my_vec_list, test_list))
    print(test_list, "classified as:",
          classify_bayes(doc_set_vec, p0v, p1v, pab))


if __name__ == '__main__':

    testing_bayes()
