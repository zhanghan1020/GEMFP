import networkx as nx
import igraph as ig
from math import log
from sklearn import metrics


def get_network_file():
    return "E:/LFR/LFR 5000 5 50 20 200/3/network10.dat"
    # return "network500-0.25equal.dat"


def get_community_file():
    return "E:/LFR/LFR 5000 5 50 20 200/3/community10.dat"


def my_graph(file_name):
    # file = open(get_network_file())
    file = open(file_name)
    g_nx = nx.Graph()
    edge_list = []
    tag = 0
    ind = 0
    nodes = set()
    # edges = []
    for line in file:
        # if ind == 0:
        #     ind +=1
        #     continue
        line = line.strip('\n')
        num_str = line.split('\t')
        num = [0, 0]
        num[0] = int(num_str[0])-1#网络从1开始要-1
        num[1] = int(num_str[1])-1
        nodes.add(num[0])
        nodes.add(num[1])
        edge_list.append((num[0], num[1]))
        if num[1] > tag + 1:
            # for I in range(tag + 1, num[1]):
            #     g_nx.add_node(I)
            tag = num[1]
        # g_nx.add_edge(num[0], num[1])
    g_ig = ig.Graph(n=tag, edges=edge_list, directed=False)
    nodes = list(nodes)
    nodes = sorted(nodes)
    for i in range(len(nodes)):
        g_nx.add_node(nodes[i])
    for e in edge_list:
        g_nx.add_edge(e[0],e[1])
    return g_nx, g_ig


def graph_comm_dict(file_name):
    # file = open(get_community_file())
    file = open(file_name)
    comm_dict = {}
    for line in file:
        line = line.strip('\n')
        num_str = line.split('\t')
        num = [0, 0]
        num[0] = int(num_str[0])-1
        num[1] = int(num_str[1])-1
        comm_dict[num[0]] = num[1]
    return comm_dict


def graph_comm_list_array(file_name):
    comm_dict = graph_comm_dict(file_name)
    keys = comm_dict.keys()
    length = len(keys)
    list_array = [[] * 1 for i in range(length)]
    for i in keys:
        list_array[comm_dict[i] - 1].append(i)
    res_list_array = []
    for i in list_array:
        if [] != i:
            res_list_array.append(i)
            continue
        break
    return res_list_array


def graph_comm_list(file_name):
    comm_dict = graph_comm_dict(file_name)
    keys = comm_dict.keys()
    length = len(keys)
    list = [-1] * length
    for i in keys:
        list[i] = comm_dict[i]#网络从1开始要-1
    return list


def change_to_mutualdict(A):
    res = dict()
    for index in A.keys():
        if A[index] not in res.keys():
            res[A[index]] = set()
        res[A[index]].add(index)
    return res


def mutual_info(c_A):
    B = graph_comm_dict()
    S = len(B)
    c_B = change_to_mutualdict(B)
    # print  "CA",c_A
    # print  "CB",c_B
    I_num = 0
    for i in c_A:
        for j in c_B:
            n_i = len(c_A[i])
            n_j = len(c_B[j])
            n_ij = len(c_A[i] & c_B[j])
            if n_ij == 0:
                continue
            log_term = log((n_ij * S * 1.0) / (n_i * n_j))

            I_num += n_ij * log_term
    I_num *= -2

    I_den = 0
    for i in c_A:
        n_i = len(c_A[i])
        I_den += n_i * log(n_i * 1.0 / S)

    for j in c_B:
        n_j = len(c_B[j])
        I_den += n_j * log(n_j * 1.0 / S)

    I = I_num / I_den
    return I


# a=graph_comm_dict()
# b=graph_comm_list_array()
# c=graph_comm_list()
#
# print a

if __name__ == '__main__':

    # G = my_graph_igraph()
    # ig.plot(G)
    # print("aa")

    # for test change_to_mutualdict(A):
    B = graph_comm_dict()
    res = change_to_mutualdict(B)

    # for test mutual_info

    # a={0:{1,2},1:{3}}
    # b={0:{1,2},1:{3}}
    # print(mutual_info(a,b))
    a = dict()
    for i in range(1000):
        a[i] = set([i + 1])
    val = mutual_info(a)
    metrics.normalized_mutual_info_score()
    print(val)
