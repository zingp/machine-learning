import matplotlib
import matplotlib.pyplot as plt

# FangSong/黑体 FangSong/KaiTi
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction', va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node('决策节点', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('叶节点', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


# 递归获取叶节点的数目
def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for k in second_dict.keys():
        if type(second_dict[k]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[k])
        else:
            num_leafs += 1
    return num_leafs


# 递归获树的最大层数
def get_tree_depth(my_tree):
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for k in second_dict.keys():
        if type(second_dict[k]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[k])
        else:
            this_depth = 1
    max_depth = max_depth if max_depth > this_depth else this_depth
    return max_depth


if __name__ == "__main__":
    create_plot()
