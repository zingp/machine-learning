import operator

a = ['a', 'b', 'c', 'a', 'b', 'b']

def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    # 按照维度为1排序，降序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_class_count)
    return sorted_class_count[0][0]

r = majority_cnt(a)
print(r)

print({'a':1, 'b':2}.items())    # 编程元素为元组的列表