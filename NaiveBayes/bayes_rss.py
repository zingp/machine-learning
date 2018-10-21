import feedparser

# url = "http://newyork.craigslist.org/stp/index.rss"
url = "http://www.nasa.gov/rss/dyn/image_of_the_day.rss"
ny = feedparser.parse(url)
print(ny['entries'])
print(len(ny['entries']))


def clac_most_freq(vocab_list, fulltext):
    import operator
    freq_dic = {}
    for tocken in vocab_list:
        freq_dic[tocken] = fulltext.count(tocken)
    sorted_freq = sorted(freq_dic.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:30]

def local_words(feed1, feed0):
    import feedparser
    doc_list, class_list, full_text = [], [], []
    
