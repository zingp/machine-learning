import feedparser
import operator
import random
import numpy as np
from bayes import *

# url = "http://newyork.craigslist.org/stp/index.rss"
url = "http://www.nasa.gov/rss/dyn/image_of_the_day.rss"
ny = feedparser.parse(url)
print(ny['entries'])
print(len(ny['entries']))


def clac_most_freq(vocab_list, fulltext):
    freq_dic = {}
    for tocken in vocab_list:
        freq_dic[tocken] = fulltext.count(tocken)
    sorted_freq = sorted(
        freq_dic.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:30]


def local_words(feed1, feed0):
    doc_list, class_list, full_text = [], [], []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        word_list = feedparser.textParse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = feedparser.textParse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = createVocabList(doc_list)
    top30_words = clac_most_freq(vocab_list, full_text)
    for pair in top30_words:
        if pair[0] in vocab_list:
            vocab_list.remove(pair[0])
    training_set = list(range(2*min_len))
    test_set = []
    for i in range(20):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])

    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(bagOfWords2Vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0v, p1v, pspam = trainNB0(np.array(train_mat), np.array(train_classes))
    error_count = 0
    for doc_index in test_set:
        word_vector = bagOfWords2Vec(vocab_list, doc_list[doc_index])
        if classify_bayes(np.array(word_vector), p0v, p1v, pspam) != class_list[doc_index]:
            error_count += 1
    print("error rate is :%f", float(error_count)/len(test_set))
    return vocab_list, p0v, p1v


if __name__ == '__main__':
    pass
