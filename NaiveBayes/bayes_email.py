import os
import re
import random
import numpy as np
from bayes import *

base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)


# 将文本装换成单词list，并去除长度小于3的单词
def txt_parse(txt):
    data_list = []
    if len(txt) != 0:
        cont = re.split(r"\W*", txt)
        data_list = [i.lower() for i in cont if len(i) > 2]
    return data_list


def word_list(name):
    word_vecs = []
    word_lists = []
    for i in range(1, 26):
        filename = os.path.join(base_dir, "email", name, "{}.txt".format(i))
        with open(filename, "r") as f:
            content = txt_parse(f.read())
            word_vecs.append(content)
    return word_vecs


def test_email_bayes():
    ham_vec = word_list("ham")
    spam_vec = word_list("spam")
    dataset = ham_vec + spam_vec
    label_list = [1]*25 + [0]*25
    print("-" * 100)
    # print(ham_list + spam_list)
    my_vec_list = createVocabList(dataset)  # 还是要用所有词集

    rand_dataset = []
    rand_lebel = []
    for i in range(10):
        x = random.randint(0, len(dataset) - 1)
        rand_dataset.append(dataset[x])
        rand_lebel.append(label_list[x])
        del(dataset[x])
        del(label_list[x])

    train_mat = []
    for post_doc in dataset:
        train_mat.append(setOfWords2Vec(my_vec_list, post_doc))
    p0v, p1v, pab = trainNB0(train_mat, label_list)

    error_num = 0
    for i in range(len(rand_dataset)):
        doc_set_vec = np.array(setOfWords2Vec(my_vec_list, rand_dataset[i]))
        this_class = classify_bayes(doc_set_vec, p0v, p1v, pab)
        if this_class != rand_lebel[i]:
            error_num += 1
            print(rand_dataset[i])
    return error_num / len(rand_dataset)


if __name__ == "__main__":
    sum_error_rate = 0
    for i in range(10):
        sum_error_rate += test_email_bayes()
    print("rvg_error_rate:", sum_error_rate / 10)
    # 多运行几次求平均 5%
